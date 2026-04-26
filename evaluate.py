import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from model import create_model, create_multitask_model, MCDropoutModel
from preprocessing import BrainMRIPreprocessor, load_and_preprocess_dataset
from explainability import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, IntegratedGradientsExplainer,
    RISEExplainer, XAIFusion, XAIBenchmark,
    visualize_explanation, visualize_multi_xai, visualize_fusion,
    visualize_uncertainty, visualize_grading_uncertainty
)
from train import assign_grade_labels, WHO_GRADES


class ModelEvaluator:
    """Evaluator for brain tumor classification model.
    
    Extended with:
    - Multi-XAI explanations (Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, RISE)
    - XAI fusion and benchmarking
    - MC Dropout uncertainty estimation
    - WHO tumor grading evaluation (for multi-task models)
    """
    
    WHO_GRADES = ['Grade I', 'Grade II', 'Grade III', 'Grade IV']
    
    def __init__(self, model_path, data_dir, output_dir='evaluation_results',
                 mc_dropout_passes=30, is_multitask=False):
        """
        Args:
            model_path: Path to the trained model
            data_dir: Directory containing the test dataset
            output_dir: Directory to save evaluation results
            mc_dropout_passes: Number of MC Dropout forward passes
            is_multitask: Whether the model is a multi-task model (classification + grading)
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.mc_dropout_passes = mc_dropout_passes
        self.is_multitask = is_multitask
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
        
        # Detect multi-task automatically
        if isinstance(self.model.output, list):
            self.is_multitask = True
            print("Detected multi-task model (classification + grading)")
        
        # Initialize preprocessor
        self.preprocessor = BrainMRIPreprocessor(target_size=(224, 224))
        
        # Class names
        self.classes = ['Meningioma', 'No Tumor', 'Glioma', 'Pituitary']
    
    def load_test_data(self):
        """Load and preprocess test data."""
        print("Loading test data...")
        
        images, labels, class_names = load_and_preprocess_dataset(
            self.data_dir,
            self.preprocessor,
            classes=self.classes
        )
        
        print(f"Loaded {len(images)} test images")
        
        # Normalize pixel values to [0, 1] - use float16 to save memory
        images = images.astype('float16') / 255.0
        
        # Assign grade labels if multi-task
        self.y_grade_test = None
        if self.is_multitask:
            self.y_grade_test = assign_grade_labels(labels, self.classes)
        
        return images, labels
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        print("\nEvaluating model...")
        
        if self.is_multitask:
            return self._evaluate_multitask(X_test, y_test)
        else:
            return self._evaluate_classification(X_test, y_test)
    
    def _evaluate_classification(self, X_test, y_test):
        """Evaluate single-task classification model."""
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': dict(zip(self.classes, precision_per_class)),
            'recall_per_class': dict(zip(self.classes, recall_per_class)),
            'f1_per_class': dict(zip(self.classes, f1_per_class))
        }
        
        print(f"\nOverall Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for class_name in self.classes:
            print(f"{class_name}:")
            print(f"  Precision: {metrics['precision_per_class'][class_name]:.4f}")
            print(f"  Recall: {metrics['recall_per_class'][class_name]:.4f}")
            print(f"  F1 Score: {metrics['f1_per_class'][class_name]:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.classes))
        
        return metrics, y_pred, y_pred_probs
    
    def _evaluate_multitask(self, X_test, y_test):
        """Evaluate multi-task model (classification + grading)."""
        y_pred_cls, y_pred_grade = self.model.predict(X_test, verbose=0)
        y_pred_cls_idx = np.argmax(y_pred_cls, axis=1)
        y_pred_grade_idx = np.argmax(y_pred_grade, axis=1)
        
        # Classification metrics
        cls_accuracy = accuracy_score(y_test, y_pred_cls_idx)
        cls_precision = precision_score(y_test, y_pred_cls_idx, average='weighted')
        cls_recall = recall_score(y_test, y_pred_cls_idx, average='weighted')
        cls_f1 = f1_score(y_test, y_pred_cls_idx, average='weighted')
        
        print(f"\nClassification Metrics:")
        print(f"Accuracy: {cls_accuracy:.4f}")
        print(f"Precision: {cls_precision:.4f}")
        print(f"Recall: {cls_recall:.4f}")
        print(f"F1 Score: {cls_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_cls_idx, target_names=self.classes))
        
        # Grading metrics
        grade_accuracy = accuracy_score(self.y_grade_test, y_pred_grade_idx)
        grade_precision = precision_score(self.y_grade_test, y_pred_grade_idx, average='weighted')
        grade_recall = recall_score(self.y_grade_test, y_pred_grade_idx, average='weighted')
        grade_f1 = f1_score(self.y_grade_test, y_pred_grade_idx, average='weighted')
        
        print(f"\nWHO Grading Metrics:")
        print(f"Accuracy: {grade_accuracy:.4f}")
        print(f"Precision: {grade_precision:.4f}")
        print(f"Recall: {grade_recall:.4f}")
        print(f"F1 Score: {grade_f1:.4f}")
        print("\nGrading Classification Report:")
        print(classification_report(self.y_grade_test, y_pred_grade_idx, target_names=self.WHO_GRADES))
        
        metrics = {
            'classification': {
                'accuracy': cls_accuracy,
                'precision': cls_precision,
                'recall': cls_recall,
                'f1_score': cls_f1,
            },
            'grading': {
                'accuracy': grade_accuracy,
                'precision': grade_precision,
                'recall': grade_recall,
                'f1_score': grade_f1,
            }
        }
        
        return metrics, y_pred_cls_idx, y_pred_cls
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title='Confusion Matrix', filename='confusion_matrix.png'):
        """Plot and save confusion matrix."""
        if labels is None:
            labels = self.classes
        
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n{title}:")
        header = "" + " " * 14 + "  ".join(f"{c:>10s}" for c in labels)
        print(header)
        print("-" * len(header))
        for i, label in enumerate(labels):
            row = f"{label:>12s} | " + "  ".join(f"{cm[i][j]:>10d}" for j in range(len(labels)))
            print(row)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = self.output_dir / filename
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {cm_path}")
        plt.close()
    
    def generate_multi_xai_explanations(self, X_test, y_test, y_pred, num_samples=5):
        """Generate explanations using multiple XAI methods + fusion.
        
        Research Gap: Comprehensive benchmarking and fusion of multiple XAI
        techniques (Score-CAM, Integrated Gradients, Grad-CAM++, RISE) to
        improve reliability.
        """
        print(f"\nGenerating multi-XAI explanations for {num_samples} samples...")
        
        # Initialize all explainers
        explainers = {
            'Grad-CAM': GradCAM(self.model),
            'Grad-CAM++': GradCAMPlusPlus(self.model),
            'Score-CAM': ScoreCAM(self.model, num_masks=64),
            'Integrated-Grad': IntegratedGradientsExplainer(self.model, num_steps=30),
            'RISE': RISEExplainer(self.model, num_masks=200, mask_resolution=8),
        }
        
        # Directories
        multi_xai_dir = self.output_dir / 'multi_xai_explanations'
        multi_xai_dir.mkdir(exist_ok=True)
        fusion_dir = self.output_dir / 'xai_fusion'
        fusion_dir.mkdir(exist_ok=True)
        
        # Fusion engine
        fusion = XAIFusion(fusion_method='mean')
        consensus_fusion = XAIFusion(fusion_method='consensus')
        
        # Select samples
        indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        
        for idx in indices:
            image = X_test[idx]
            pred_label = y_pred[idx]
            true_label = y_test[idx]
            
            image_255 = (image * 255).astype(np.uint8)
            
            # Generate heatmaps from all methods
            heatmaps_dict = {}
            for name, explainer in explainers.items():
                print(f"  [{idx}] Generating {name}...")
                heatmap, overlayed = explainer.explain(image, pred_label, alpha=0.4)
                heatmaps_dict[name] = (heatmap, overlayed)
            
            # Save multi-XAI comparison
            class_name = self.classes[pred_label]
            save_path = multi_xai_dir / f'multi_xai_{idx}_true_{self.classes[true_label]}_pred_{class_name}.png'
            visualize_multi_xai(image_255, heatmaps_dict, class_name, save_path=save_path)
            
            # Fuse heatmaps
            heatmap_list = [h for h, _ in heatmaps_dict.values()]
            
            fused_mean = fusion.fuse(heatmap_list)
            fused_overlay = _overlay_heatmap_helper(image_255, fused_mean)
            
            fused_consensus = consensus_fusion.fuse(heatmap_list)
            fused_cons_overlay = _overlay_heatmap_helper(image_255, fused_consensus)
            
            fusion_save = fusion_dir / f'fusion_{idx}_pred_{class_name}.png'
            visualize_fusion(image_255, heatmaps_dict, fused_mean, fused_overlay,
                             class_name, save_path=fusion_save)
        
        print(f"Multi-XAI explanations saved to {multi_xai_dir}")
        print(f"XAI fusion results saved to {fusion_dir}")
        return explainers
    
    def run_xai_benchmark(self, X_test, y_test, y_pred, num_samples=3):
        """Run quantitative XAI benchmark (deletion/insertion/consistency).
        
        Research Gap: Comprehensive benchmarking of multiple XAI techniques.
        """
        print(f"\nRunning XAI benchmark on {num_samples} samples...")
        
        explainers = {
            'Grad-CAM': GradCAM(self.model),
            'Grad-CAM++': GradCAMPlusPlus(self.model),
            'Score-CAM': ScoreCAM(self.model, num_masks=64),
            'Integrated-Grad': IntegratedGradientsExplainer(self.model, num_steps=30),
            'RISE': RISEExplainer(self.model, num_masks=200, mask_resolution=8),
        }
        
        benchmark = XAIBenchmark(self.model, explainers, num_steps=15)
        
        indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        
        all_results = {}
        for idx in indices:
            image = X_test[idx]
            pred_label = y_pred[idx]
            print(f"\nBenchmarking sample {idx} (class: {self.classes[pred_label]})...")
            
            results = benchmark.benchmark(image, pred_label)
            all_results[idx] = results
        
        # Aggregate and print summary
        print("\n" + "=" * 80)
        print("AGGREGATED XAI BENCHMARK RESULTS")
        print("=" * 80)
        
        method_names = list(explainers.keys())
        for method in method_names:
            del_aucs = [all_results[idx][method]['deletion_auc'] for idx in all_results]
            ins_aucs = [all_results[idx][method]['insertion_auc'] for idx in all_results]
            consistencies = []
            for idx in all_results:
                consistencies.extend(list(all_results[idx][method]['consistency'].values()))
            
            print(f"{method:<20} Del-AUC: {np.mean(del_aucs):.4f}  Ins-AUC: {np.mean(ins_aucs):.4f}  Avg-IoU: {np.mean(consistencies):.4f}")
        
        # Save benchmark results
        benchmark_path = self.output_dir / 'xai_benchmark.json'
        serializable = {}
        for idx, res in all_results.items():
            serializable[str(idx)] = {}
            for method, data in res.items():
                serializable[str(idx)][method] = {
                    'deletion_auc': data['deletion_auc'],
                    'insertion_auc': data['insertion_auc'],
                    'consistency': {k: float(v) for k, v in data['consistency'].items()}
                }
        with open(benchmark_path, 'w') as f:
            json.dump(serializable, f, indent=4)
        print(f"\nBenchmark results saved to {benchmark_path}")
        
        return all_results
    
    def estimate_uncertainty(self, X_test, y_pred):
        """Run MC Dropout uncertainty estimation.
        
        Research Gap: Integration of uncertainty-aware models (Bayesian CNN / MC Dropout).
        """
        print(f"\nRunning MC Dropout uncertainty estimation ({self.mc_dropout_passes} passes)...")
        
        mc_model = MCDropoutModel(self.model, num_passes=self.mc_dropout_passes)
        
        if self.is_multitask:
            uncertainty = mc_model.predict_with_uncertainty_multitask(X_test)
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
            
            # Visualise a few samples
            unc_dir = self.output_dir / 'uncertainty'
            unc_dir.mkdir(exist_ok=True)
            
            indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
            for idx in indices:
                image_255 = (X_test[idx] * 255).astype(np.uint8)
                cls_name = self.classes[y_pred[idx]] if not isinstance(y_pred, dict) else self.classes[y_pred[idx]]
                grade_idx = grade_unc['pred_class'][idx]
                grade_name = self.WHO_GRADES[grade_idx]
                
                cls_single = {k: v[idx:idx+1] if isinstance(v, np.ndarray) else v for k, v in cls_unc.items()}
                grade_single = {k: v[idx:idx+1] if isinstance(v, np.ndarray) else v for k, v in grade_unc.items()}
                
                save_path = unc_dir / f'uncertainty_{idx}.png'
                visualize_grading_uncertainty(
                    image_255,
                    {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 else v) for k, v in cls_unc.items()},
                    {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 else v) for k, v in grade_unc.items()},
                    cls_name, grade_name, save_path=save_path
                )
            
            return uncertainty
        
        else:
            uncertainty = mc_model.predict_with_uncertainty(X_test)
            
            print(f"\nUncertainty Statistics:")
            print(f"  Mean confidence: {np.mean(uncertainty['confidence']):.4f}")
            print(f"  Mean entropy: {np.mean(uncertainty['entropy']):.4f}")
            print(f"  Mean epistemic (MI): {np.mean(uncertainty['mutual_info']):.4f}")
            print(f"  Mean variation ratio: {np.mean(uncertainty['variation_ratio']):.4f}")
            
            # Flag uncertain predictions
            uncertain_mask = uncertainty['confidence'] < 0.5
            n_uncertain = np.sum(uncertain_mask)
            print(f"  Uncertain predictions (confidence < 0.5): {n_uncertain}/{len(uncertainty['confidence'])}")
            
            # Visualise a few samples
            unc_dir = self.output_dir / 'uncertainty'
            unc_dir.mkdir(exist_ok=True)
            
            indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
            for idx in indices:
                image_255 = (X_test[idx] * 255).astype(np.uint8)
                cls_name = self.classes[y_pred[idx]]
                
                save_path = unc_dir / f'uncertainty_{idx}.png'
                visualize_uncertainty(
                    image_255,
                    {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 else v) for k, v in uncertainty.items()},
                    cls_name, save_path=save_path
                )
            
            return uncertainty
    
    def save_metrics(self, metrics):
        """Save evaluation metrics to JSON file."""
        if self.is_multitask:
            serializable = {
                'classification': {k: float(v) for k, v in metrics['classification'].items()},
                'grading': {k: float(v) for k, v in metrics['grading'].items()},
            }
        else:
            serializable = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'precision_per_class': {k: float(v) for k, v in metrics['precision_per_class'].items()},
                'recall_per_class': {k: float(v) for k, v in metrics['recall_per_class'].items()},
                'f1_per_class': {k: float(v) for k, v in metrics['f1_per_class'].items()}
            }
        
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(serializable, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
    
    def run(self, generate_explanations=True, num_explanations=5,
            run_benchmark=True, num_benchmark_samples=3,
            run_uncertainty=True):
        """Run complete evaluation pipeline.
        
        Args:
            generate_explanations: Whether to generate multi-XAI explanations
            num_explanations: Number of explanation samples
            run_benchmark: Whether to run XAI benchmark
            num_benchmark_samples: Number of benchmark samples
            run_uncertainty: Whether to run MC Dropout uncertainty estimation
        """
        # Load test data
        X_test, y_test = self.load_test_data()
        
        # Evaluate
        eval_output = self.evaluate(X_test, y_test)
        metrics = eval_output[0]
        y_pred = eval_output[1]
        
        # Plot confusion matrix (classification)
        self.plot_confusion_matrix(y_test, y_pred, labels=self.classes,
                                   title='Classification Confusion Matrix',
                                   filename='confusion_matrix.png')
        
        # Plot grading confusion matrix if multi-task
        if self.is_multitask and self.y_grade_test is not None:
            y_pred_grade = np.argmax(self.model.predict(X_test, verbose=0)[1], axis=1)
            self.plot_confusion_matrix(self.y_grade_test, y_pred_grade, labels=self.WHO_GRADES,
                                       title='WHO Grading Confusion Matrix',
                                       filename='grading_confusion_matrix.png')
        
        # Generate multi-XAI explanations
        if generate_explanations:
            self.generate_multi_xai_explanations(X_test, y_test, y_pred, num_explanations)
        
        # Run XAI benchmark
        if run_benchmark:
            self.run_xai_benchmark(X_test, y_test, y_pred, num_benchmark_samples)
        
        # Run uncertainty estimation
        if run_uncertainty:
            self.estimate_uncertainty(X_test, y_pred)
        
        # Save metrics
        self.save_metrics(metrics)
        
        print("\nEvaluation completed successfully!")
        return metrics


def _overlay_heatmap_helper(image, heatmap, alpha=0.4):
    """Helper to overlay heatmap on image."""
    import cv2
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    if len(image.shape) == 3 and image.shape[2] == 3:
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed


def main():
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate brain tumor classification model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the test dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--generate_explanations', action='store_true',
                        help='Whether to generate multi-XAI explanations')
    parser.add_argument('--num_explanations', type=int, default=5,
                        help='Number of explanation samples to generate')
    parser.add_argument('--run_benchmark', action='store_true',
                        help='Whether to run XAI benchmark')
    parser.add_argument('--num_benchmark_samples', type=int, default=3,
                        help='Number of benchmark samples')
    parser.add_argument('--run_uncertainty', action='store_true',
                        help='Whether to run MC Dropout uncertainty estimation')
    parser.add_argument('--mc_dropout_passes', type=int, default=30,
                        help='Number of MC Dropout forward passes')
    parser.add_argument('--is_multitask', action='store_true',
                        help='Whether the model is multi-task (auto-detected)')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mc_dropout_passes=args.mc_dropout_passes,
        is_multitask=args.is_multitask
    )
    
    evaluator.run(
        generate_explanations=args.generate_explanations,
        num_explanations=args.num_explanations,
        run_benchmark=args.run_benchmark,
        num_benchmark_samples=args.num_benchmark_samples,
        run_uncertainty=args.run_uncertainty
    )


if __name__ == "__main__":
    main()
