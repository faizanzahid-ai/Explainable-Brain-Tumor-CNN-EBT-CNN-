import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from tqdm import tqdm

from model import create_model, create_multitask_model, MCDropoutModel
from preprocessing import BrainMRIPreprocessor, load_and_preprocess_dataset
from explainability import GradCAM


# ---------------------------------------------------------------------------
# WHO Grade assignment heuristic
# ---------------------------------------------------------------------------

# Research Gap: Development of an XAI-based tumor grading system.
# Default grade mapping based on typical WHO classifications:
#   Meningioma → mostly Grade I (0), some Grade II (1)
#   Glioma     → mix of Grade II–IV (1,2,3)
#   Pituitary  → mostly Grade I (0)
#   No Tumor   → Grade 0 placeholder (not used in grading loss)
DEFAULT_GRADE_MAP = {
    'Meningioma': [0.70, 0.25, 0.04, 0.01],   # P(Grade I,II,III,IV)
    'No Tumor':   [1.00, 0.00, 0.00, 0.00],    # placeholder
    'Glioma':     [0.05, 0.30, 0.40, 0.25],
    'Pituitary':  [0.85, 0.12, 0.02, 0.01],
}

WHO_GRADES = ['Grade I', 'Grade II', 'Grade III', 'Grade IV']


def assign_grade_labels(class_labels, class_names, grade_map=None, random_state=42):
    """Assign synthetic WHO grade labels based on class-wise probability distributions.
    
    In a real clinical setting, grade labels would come from pathology reports.
    Here we use a heuristic distribution so the grading head can be trained.
    
    Args:
        class_labels: Array of class indices (0..num_classes-1)
        class_names: List of class names
        grade_map: Dict mapping class name → list of grade probabilities
        random_state: Random seed
    
    Returns:
        grade_labels: Array of grade indices (0..3)
    """
    rng = np.random.RandomState(random_state)
    if grade_map is None:
        grade_map = DEFAULT_GRADE_MAP
    
    grade_labels = np.zeros(len(class_labels), dtype=np.int32)
    for i, cls_idx in enumerate(class_labels):
        cls_name = class_names[cls_idx]
        probs = grade_map.get(cls_name, [0.25, 0.25, 0.25, 0.25])
        grade_labels[i] = rng.choice(len(probs), p=probs)
    
    return grade_labels


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BrainTumorTrainer:
    """Trainer for brain tumor classification following paper methodology.
    
    Supports two modes:
    - 'classification': Original single-task model (backward compatible)
    - 'multitask': Multi-task model with classification + WHO tumor grading
    """
    
    def __init__(self, data_dir, output_dir='output', 
                 learning_rate=0.001, batch_size=40, epochs=40,
                 test_split=0.1, val_split=0.1, random_state=42,
                 mode='multitask', mc_dropout_passes=30):
        """
        Args:
            data_dir: Directory containing the dataset
            output_dir: Directory to save model and results
            learning_rate: Learning rate for optimizer (0.001 per paper)
            batch_size: Batch size for training (40 per paper)
            epochs: Number of training epochs (40 per paper)
            test_split: Fraction of data for testing (0.1 per paper)
            val_split: Fraction of training data for validation (0.1 per paper)
            random_state: Random seed for reproducibility
            mode: 'classification' or 'multitask'
            mc_dropout_passes: Number of MC Dropout forward passes for uncertainty
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_split = test_split
        self.val_split = val_split
        self.random_state = random_state
        self.mode = mode
        self.mc_dropout_passes = mc_dropout_passes
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessor
        self.preprocessor = BrainMRIPreprocessor(target_size=(224, 224))
        
        # Class names
        self.classes = ['Meningioma', 'No Tumor', 'Glioma', 'Pituitary']
        
        # Model
        self.model = None
        self.history = None
        
        # Data (classification)
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Data (grading) — only used in multitask mode
        self.y_grade_train = None
        self.y_grade_val = None
        self.y_grade_test = None
    
    def load_data(self):
        """Load and preprocess the dataset."""
        print("Loading and preprocessing dataset...")
        
        data_path = Path(self.data_dir)
        
        # Check if dataset has Training/Testing structure
        training_dir = data_path / 'Training'
        testing_dir = data_path / 'Testing'
        
        if training_dir.exists() and testing_dir.exists():
            print("Found Training/Testing structure")
            
            train_images, train_labels, class_names = load_and_preprocess_dataset(
                training_dir,
                self.preprocessor,
                classes=self.classes
            )
            
            test_images, test_labels, _ = load_and_preprocess_dataset(
                testing_dir,
                self.preprocessor,
                classes=self.classes
            )
            
            images = np.concatenate([train_images, test_images])
            labels = np.concatenate([train_labels, test_labels])
            
            self.X_test = test_images
            self.y_test = test_labels
            
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                train_images, train_labels,
                test_size=self.val_split,
                random_state=self.random_state,
                stratify=train_labels
            )
        else:
            images, labels, class_names = load_and_preprocess_dataset(
                self.data_dir,
                self.preprocessor,
                classes=self.classes
            )
            
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                images, labels,
                test_size=self.test_split,
                random_state=self.random_state,
                stratify=labels
            )
            
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp,
                test_size=self.val_split / (1 - self.test_split),
                random_state=self.random_state,
                stratify=y_temp
            )
        
        print(f"Loaded {len(images)} images across {len(class_names)} classes")
        print(f"Class distribution: {dict(zip(class_names, [np.sum(labels == i) for i in range(len(class_names))]))}")
        
        print(f"Training set: {len(self.X_train)} images")
        print(f"Validation set: {len(self.X_val)} images")
        print(f"Test set: {len(self.X_test)} images")
        
        # Normalize pixel values to [0, 1] - use float16 to save memory
        self.X_train = self.X_train.astype('float16') / 255.0
        self.X_val = self.X_val.astype('float16') / 255.0
        self.X_test = self.X_test.astype('float16') / 255.0
        
        # Assign grade labels for multitask mode
        if self.mode == 'multitask':
            print("\nAssigning WHO grade labels (synthetic heuristic)...")
            self.y_grade_train = assign_grade_labels(self.y_train, self.classes, random_state=self.random_state)
            self.y_grade_val = assign_grade_labels(self.y_val, self.classes, random_state=self.random_state + 1)
            self.y_grade_test = assign_grade_labels(self.y_test, self.classes, random_state=self.random_state + 2)
            
            for i, g in enumerate(WHO_GRADES):
                count = np.sum(self.y_grade_train == i)
                print(f"  {g}: {count} training samples")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def build_model(self):
        """Build and compile the model based on the selected mode."""
        print(f"Building model (mode={self.mode})...")
        
        if self.mode == 'multitask':
            self.model = create_multitask_model(
                input_shape=(224, 224, 3),
                num_classes=len(self.classes),
                num_grades=4,
                learning_rate=self.learning_rate
            )
        else:
            self.model = create_model(
                input_shape=(224, 224, 3),
                num_classes=len(self.classes),
                learning_rate=self.learning_rate
            )
        
        self.model.summary()
    
    def train(self, use_callbacks=True):
        """Train the model."""
        print(f"\nTraining model for {self.epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Mode: {self.mode}")
        
        callbacks = []
        if use_callbacks:
            if self.mode == 'multitask':
                monitor_metric = 'val_classification_accuracy'
            else:
                monitor_metric = 'val_accuracy'
            
            checkpoint_path = self.output_dir / 'best_model.keras'
            checkpoint = keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor=monitor_metric,
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint)
            
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # Prepare training data
        if self.mode == 'multitask':
            train_data = {
                'classification': self.y_train,
                'grading': self.y_grade_train
            }
            val_data = (
                self.X_val,
                {
                    'classification': self.y_val,
                    'grading': self.y_grade_val
                }
            )
        else:
            train_data = self.y_train
            val_data = (self.X_val, self.y_val)
        
        # Train model
        self.history = self.model.fit(
            self.X_train, train_data,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        final_model_path = self.output_dir / 'final_model.keras'
        self.model.save(final_model_path)
        print(f"\nModel saved to {final_model_path}")
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        history_dict = {key: [float(val) for val in values] 
                       for key, values in self.history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        print(f"Training history saved to {history_path}")
    
    def evaluate(self):
        """Evaluate the model on test set."""
        print("\nEvaluating model on test set...")
        
        if self.mode == 'multitask':
            return self._evaluate_multitask()
        else:
            return self._evaluate_classification()
    
    def _evaluate_classification(self):
        """Evaluate single-task classification model."""
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        y_pred_probs = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return test_loss, test_accuracy, y_pred, precision, recall, f1
    
    def _evaluate_multitask(self):
        """Evaluate multi-task model (classification + grading)."""
        results = self.model.evaluate(
            self.X_test,
            {
                'classification': self.y_test,
                'grading': self.y_grade_test
            },
            verbose=0,
            return_dict=True
        )
        
        print("\nMulti-task evaluation results:")
        for key, val in results.items():
            print(f"  {key}: {val:.4f}")
        
        # Predictions
        y_pred_cls, y_pred_grade = self.model.predict(self.X_test, verbose=0)
        y_pred_cls_idx = np.argmax(y_pred_cls, axis=1)
        y_pred_grade_idx = np.argmax(y_pred_grade, axis=1)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Classification metrics
        cls_precision = precision_score(self.y_test, y_pred_cls_idx, average='weighted')
        cls_recall = recall_score(self.y_test, y_pred_cls_idx, average='weighted')
        cls_f1 = f1_score(self.y_test, y_pred_cls_idx, average='weighted')
        
        print(f"\nClassification — Precision: {cls_precision:.4f}, Recall: {cls_recall:.4f}, F1: {cls_f1:.4f}")
        
        # Grading metrics
        grade_precision = precision_score(self.y_grade_test, y_pred_grade_idx, average='weighted')
        grade_recall = recall_score(self.y_grade_test, y_pred_grade_idx, average='weighted')
        grade_f1 = f1_score(self.y_grade_test, y_pred_grade_idx, average='weighted')
        
        print(f"Grading — Precision: {grade_precision:.4f}, Recall: {grade_recall:.4f}, F1: {grade_f1:.4f}")
        
        # MC Dropout uncertainty estimation
        print("\nRunning MC Dropout uncertainty estimation...")
        mc_model = MCDropoutModel(self.model, num_passes=self.mc_dropout_passes)
        uncertainty = mc_model.predict_with_uncertainty_multitask(self.X_test)
        
        # Print uncertainty statistics
        cls_unc = uncertainty['classification']
        grade_unc = uncertainty['grading']
        
        print(f"\nClassification Uncertainty:")
        print(f"  Mean confidence: {np.mean(cls_unc['confidence']):.4f}")
        print(f"  Mean entropy: {np.mean(cls_unc['entropy']):.4f}")
        print(f"  Mean epistemic (MI): {np.mean(cls_unc['mutual_info']):.4f}")
        print(f"  Mean variation ratio: {np.mean(cls_unc['variation_ratio']):.4f}")
        
        print(f"\nGrading Uncertainty:")
        print(f"  Mean confidence: {np.mean(grade_unc['confidence']):.4f}")
        print(f"  Mean entropy: {np.mean(grade_unc['entropy']):.4f}")
        print(f"  Mean epistemic (MI): {np.mean(grade_unc['mutual_info']):.4f}")
        print(f"  Mean variation ratio: {np.mean(grade_unc['variation_ratio']):.4f}")
        
        # Identify uncertain predictions (low confidence)
        uncertain_cls = np.sum(cls_unc['confidence'] < 0.5)
        uncertain_grade = np.sum(grade_unc['confidence'] < 0.5)
        print(f"\nUncertain predictions (confidence < 0.5):")
        print(f"  Classification: {uncertain_cls}/{len(cls_unc['confidence'])}")
        print(f"  Grading: {uncertain_grade}/{len(grade_unc['confidence'])}")
        
        return results, y_pred_cls_idx, y_pred_grade_idx, cls_precision, cls_recall, cls_f1, uncertainty
    
    def save_results(self, *args):
        """Save evaluation results."""
        if self.mode == 'multitask':
            self._save_results_multitask(*args)
        else:
            self._save_results_classification(*args)
    
    def _save_results_classification(self, test_loss, test_accuracy, y_pred, precision, recall, f1):
        """Save single-task results."""
        results = {
            'mode': 'classification',
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'classes': self.classes
        }
        
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_path}")
    
    def _save_results_multitask(self, eval_results, y_pred_cls, y_pred_grade,
                                 cls_precision, cls_recall, cls_f1, uncertainty):
        """Save multi-task results including uncertainty."""
        cls_unc = uncertainty['classification']
        grade_unc = uncertainty['grading']
        
        results = {
            'mode': 'multitask',
            'classification': {
                'precision': float(cls_precision),
                'recall': float(cls_recall),
                'f1_score': float(cls_f1),
            },
            'grading': {
                'precision': float(0),  # placeholder, computed separately
                'recall': float(0),
                'f1_score': float(0),
            },
            'uncertainty': {
                'classification': {
                    'mean_confidence': float(np.mean(cls_unc['confidence'])),
                    'mean_entropy': float(np.mean(cls_unc['entropy'])),
                    'mean_epistemic_mi': float(np.mean(cls_unc['mutual_info'])),
                    'mean_variation_ratio': float(np.mean(cls_unc['variation_ratio'])),
                    'num_uncertain_pred_conf_05': int(np.sum(cls_unc['confidence'] < 0.5)),
                },
                'grading': {
                    'mean_confidence': float(np.mean(grade_unc['confidence'])),
                    'mean_entropy': float(np.mean(grade_unc['entropy'])),
                    'mean_epistemic_mi': float(np.mean(grade_unc['mutual_info'])),
                    'mean_variation_ratio': float(np.mean(grade_unc['variation_ratio'])),
                    'num_uncertain_pred_conf_05': int(np.sum(grade_unc['confidence'] < 0.5)),
                }
            },
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'mc_dropout_passes': self.mc_dropout_passes,
            'classes': self.classes,
            'who_grades': WHO_GRADES
        }
        
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_path}")
    
    def run(self):
        """Run the complete training pipeline."""
        # Load data
        self.load_data()
        
        # Build model
        self.build_model()
        
        # Train
        self.train()
        
        # Evaluate
        eval_output = self.evaluate()
        
        # Save results
        self.save_results(*eval_output)
        
        print("\nTraining completed successfully!")
        return eval_output


def main():
    """Main function to run training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train brain tumor classification model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save model and results')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=40,
                        help='Batch size for training (default: 40)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs (default: 40)')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction of data for testing (default: 0.1)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of training data for validation (default: 0.1)')
    parser.add_argument('--mode', type=str, default='multitask',
                        choices=['classification', 'multitask'],
                        help='Training mode: classification (single-task) or multitask (cls + grading)')
    parser.add_argument('--mc_dropout_passes', type=int, default=30,
                        help='Number of MC Dropout forward passes for uncertainty (default: 30)')
    
    args = parser.parse_args()
    
    trainer = BrainTumorTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        test_split=args.test_split,
        val_split=args.val_split,
        mode=args.mode,
        mc_dropout_passes=args.mc_dropout_passes
    )
    
    trainer.run()


if __name__ == "__main__":
    main()
