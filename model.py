import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class BrainTumorCNN:
    """CNN architecture for brain tumor classification as described in the paper."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the CNN architecture as described in the paper with regularization.
        
        Architecture:
        - Conv layers with filters: 8, 16, 32, 64, 128, 256
        - Kernel size: 3x3
        - Activation: ReLU
        - Padding: same
        - Max pooling after each conv layer (factor of 2)
        - Batch normalization after each conv block
        - Dropout for regularization
        - Average pooling
        - Flatten
        - 2 Dense layers with 512 neurons each (ReLU) with Dropout
        - Output Dense layer with softmax
        """
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # Convolutional blocks with increasing filters
        filters = [8, 16, 32, 64, 128, 256]
        
        for num_filters in filters:
            # Convolutional layer
            model.add(layers.Conv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                padding='same',
                activation='relu'
            ))
            
            # Max pooling layer (downsampling by factor of 2)
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
            # Batch normalization
            model.add(layers.BatchNormalization())
        
        # Dropout after conv blocks
        model.add(layers.Dropout(0.3))
        
        # Average pooling layer
        model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        
        # Flatten layer
        model.add(layers.Flatten())
        
        # First dense layer with 512 neurons and dropout
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        
        # Second dense layer with 512 neurons and dropout
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        
        # Output layer with softmax activation
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer, loss function, and metrics.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def summary(self):
        """Print model summary."""
        self.model.summary()
    
    def get_model(self):
        """Return the Keras model."""
        return self.model


class BrainTumorMultiTaskCNN:
    """Multi-task CNN for brain tumor classification AND WHO tumor grading (I-IV).
    
    Research Gap: Development of an XAI-based tumor grading system to assist in treatment planning.
    
    Architecture:
    - Shared convolutional backbone (same as BrainTumorCNN)
    - Two task-specific heads:
      1. Classification head: tumor type (Meningioma, No Tumor, Glioma, Pituitary)
      2. Grading head: WHO Grade I-IV (only for tumor-positive images)
    """
    
    WHO_GRADES = ['Grade I', 'Grade II', 'Grade III', 'Grade IV']
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4, num_grades=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_grades = num_grades
        self.model = self._build_model()
    
    def _build_shared_backbone(self, inputs):
        """Build shared convolutional backbone."""
        x = inputs
        filters = [8, 16, 32, 64, 128, 256]
        
        for num_filters in filters:
            x = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.BatchNormalization()(x)
        
        x = layers.Dropout(0.3)(x)
        x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        
        # Shared dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        return x
    
    def _build_model(self):
        """Build multi-task model with classification and grading heads."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Shared backbone
        shared = self._build_shared_backbone(inputs)
        
        # Classification head (tumor type)
        cls_head = layers.Dense(256, activation='relu')(shared)
        cls_head = layers.Dropout(0.3)(cls_head)
        cls_output = layers.Dense(self.num_classes, activation='softmax', name='classification')(cls_head)
        
        # Grading head (WHO Grade I-IV)
        grade_head = layers.Dense(256, activation='relu')(shared)
        grade_head = layers.Dropout(0.3)(grade_head)
        grade_output = layers.Dense(self.num_grades, activation='softmax', name='grading')(grade_head)
        
        model = models.Model(inputs=inputs, outputs=[cls_output, grade_output])
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile multi-task model with separate losses for each head."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'classification': 'sparse_categorical_crossentropy',
                'grading': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'classification': 1.0,
                'grading': 0.5
            },
            metrics={
                'classification': 'accuracy',
                'grading': 'accuracy'
            }
        )
    
    def summary(self):
        """Print model summary."""
        self.model.summary()
    
    def get_model(self):
        """Return the Keras model."""
        return self.model


class MCDropoutModel:
    """Monte Carlo Dropout wrapper for uncertainty estimation.
    
    Research Gap: Integration of uncertainty-aware models (Bayesian CNN / Monte Carlo Dropout).
    
    At inference time, dropout layers are kept active to produce stochastic forward passes.
    Multiple passes yield a distribution of predictions from which we compute:
    - Mean prediction (expected class probabilities)
    - Predictive entropy (total uncertainty)
    - Mutual information (epistemic uncertainty)
    - Variation ratio (aleatoric uncertainty proxy)
    """
    
    def __init__(self, model, num_passes=30):
        """
        Args:
            model: Trained Keras model with Dropout layers
            num_passes: Number of stochastic forward passes (T in MC Dropout)
        """
        self.model = model
        self.num_passes = num_passes
    
    def predict_with_uncertainty(self, X, batch_size=32):
        """
        Run MC Dropout inference and compute uncertainty metrics.
        
        Args:
            X: Input images (numpy array)
            batch_size: Batch size for prediction
            
        Returns:
            dict with keys:
                'mean_probs': Mean predicted probabilities across passes (n_samples, n_classes)
                'pred_class': Predicted class from mean probs
                'confidence': Max probability from mean probs
                'entropy': Predictive entropy per sample
                'mutual_info': Mutual information (epistemic uncertainty) per sample
                'variation_ratio': Variation ratio per sample
                'all_probs': All T forward pass probability arrays (T, n_samples, n_classes)
        """
        # Enable dropout at inference time
        predictions = []
        
        for i in range(self.num_passes):
            # Use training=True to keep dropout active
            if isinstance(self.model.output, list):
                # Multi-task model: use classification output only
                pred = self.model(X, training=True)[0]
            else:
                pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        all_probs = np.array(predictions)  # (T, n_samples, n_classes)
        
        # Mean prediction
        mean_probs = np.mean(all_probs, axis=0)  # (n_samples, n_classes)
        pred_class = np.argmax(mean_probs, axis=1)
        confidence = np.max(mean_probs, axis=1)
        
        # Predictive entropy: H[y|x] = - sum_c p_c * log(p_c)
        eps = 1e-10
        entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)
        
        # Mutual information: I[y;w|x] = H[y|x] - E_w[H[y|x,w]]
        # = entropy of mean - mean of entropies of each pass
        per_pass_entropy = -np.sum(all_probs * np.log(all_probs + eps), axis=2)  # (T, n_samples)
        mean_per_pass_entropy = np.mean(per_pass_entropy, axis=0)  # (n_samples,)
        mutual_info = entropy - mean_per_pass_entropy
        
        # Variation ratio: 1 - (max frequency of predicted class across passes)
        per_pass_pred = np.argmax(all_probs, axis=2)  # (T, n_samples)
        variation_ratio = np.zeros(mean_probs.shape[0])
        for s in range(mean_probs.shape[0]):
            counts = np.bincount(per_pass_pred[:, s], minlength=mean_probs.shape[1])
            mode_count = np.max(counts)
            variation_ratio[s] = 1.0 - mode_count / self.num_passes
        
        return {
            'mean_probs': mean_probs,
            'pred_class': pred_class,
            'confidence': confidence,
            'entropy': entropy,
            'mutual_info': mutual_info,
            'variation_ratio': variation_ratio,
            'all_probs': all_probs
        }
    
    def predict_with_uncertainty_multitask(self, X, batch_size=32):
        """
        MC Dropout inference for multi-task model (classification + grading).
        
        Returns:
            dict with classification uncertainty + grading uncertainty + combined metrics
        """
        cls_predictions = []
        grade_predictions = []
        
        for i in range(self.num_passes):
            outputs = self.model(X, training=True)
            cls_predictions.append(outputs[0].numpy())
            grade_predictions.append(outputs[1].numpy())
        
        all_cls_probs = np.array(cls_predictions)
        all_grade_probs = np.array(grade_predictions)
        
        # Classification uncertainty
        cls_mean = np.mean(all_cls_probs, axis=0)
        cls_pred = np.argmax(cls_mean, axis=1)
        cls_confidence = np.max(cls_mean, axis=1)
        eps = 1e-10
        cls_entropy = -np.sum(cls_mean * np.log(cls_mean + eps), axis=1)
        
        cls_per_pass_entropy = -np.sum(all_cls_probs * np.log(all_cls_probs + eps), axis=2)
        cls_mutual_info = cls_entropy - np.mean(cls_per_pass_entropy, axis=0)
        
        # Grading uncertainty
        grade_mean = np.mean(all_grade_probs, axis=0)
        grade_pred = np.argmax(grade_mean, axis=1)
        grade_confidence = np.max(grade_mean, axis=1)
        grade_entropy = -np.sum(grade_mean * np.log(grade_mean + eps), axis=1)
        
        grade_per_pass_entropy = -np.sum(all_grade_probs * np.log(all_grade_probs + eps), axis=2)
        grade_mutual_info = grade_entropy - np.mean(grade_per_pass_entropy, axis=0)
        
        # Variation ratio for classification
        cls_per_pass_pred = np.argmax(all_cls_probs, axis=2)
        cls_var_ratio = np.zeros(cls_mean.shape[0])
        for s in range(cls_mean.shape[0]):
            counts = np.bincount(cls_per_pass_pred[:, s], minlength=cls_mean.shape[1])
            cls_var_ratio[s] = 1.0 - np.max(counts) / self.num_passes
        
        # Variation ratio for grading
        grade_per_pass_pred = np.argmax(all_grade_probs, axis=2)
        grade_var_ratio = np.zeros(grade_mean.shape[0])
        for s in range(grade_mean.shape[0]):
            counts = np.bincount(grade_per_pass_pred[:, s], minlength=grade_mean.shape[1])
            grade_var_ratio[s] = 1.0 - np.max(counts) / self.num_passes
        
        return {
            'classification': {
                'mean_probs': cls_mean,
                'pred_class': cls_pred,
                'confidence': cls_confidence,
                'entropy': cls_entropy,
                'mutual_info': cls_mutual_info,
                'variation_ratio': cls_var_ratio,
                'all_probs': all_cls_probs
            },
            'grading': {
                'mean_probs': grade_mean,
                'pred_class': grade_pred,
                'confidence': grade_confidence,
                'entropy': grade_entropy,
                'mutual_info': grade_mutual_info,
                'variation_ratio': grade_var_ratio,
                'all_probs': all_grade_probs
            }
        }


def create_model(input_shape=(224, 224, 3), num_classes=4, learning_rate=0.001):
    """
    Create and compile the brain tumor classification model.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    cnn = BrainTumorCNN(input_shape=input_shape, num_classes=num_classes)
    cnn.compile_model(learning_rate=learning_rate)
    return cnn.get_model()


def create_multitask_model(input_shape=(224, 224, 3), num_classes=4, num_grades=4, learning_rate=0.001):
    """
    Create and compile the multi-task model for classification + tumor grading.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of tumor type classes
        num_grades: Number of WHO grade classes (default 4 for Grade I-IV)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras multi-task model
    """
    cnn = BrainTumorMultiTaskCNN(input_shape=input_shape, num_classes=num_classes, num_grades=num_grades)
    cnn.compile_model(learning_rate=learning_rate)
    return cnn.get_model()


if __name__ == "__main__":
    # Test model creation
    print("=== Single-task model ===")
    model = create_model()
    model.summary()
    
    print("\n=== Multi-task model (classification + grading) ===")
    mt_model = create_multitask_model()
    mt_model.summary()
