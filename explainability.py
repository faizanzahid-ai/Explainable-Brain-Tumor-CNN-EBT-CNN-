import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import models


# ---------------------------------------------------------------------------
# Base utilities
# ---------------------------------------------------------------------------

def _find_last_conv_layer(model):
    """Find the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in the model")


def _overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay a heatmap on the original image."""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    if len(image.shape) == 3 and image.shape[2] == 3:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlayed


# ---------------------------------------------------------------------------
# 1. Grad-CAM (original)
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM implementation for model explainability as described in the paper."""
    
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        if self.layer_name is None:
            self.layer_name = _find_last_conv_layer(self.model)
        self.grad_model = self._build_grad_model()
    
    def _build_grad_model(self):
        # Handle multi-task models: use classification output only
        model_output = self.model.output
        if isinstance(model_output, list):
            model_output = model_output[0]  # Use classification head
        return models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, model_output]
        )
    
    def generate_heatmap(self, image, class_idx, eps=1e-8):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + eps)
        return heatmap.numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        return _overlay_heatmap(image, heatmap, alpha, colormap)
    
    def explain(self, image, class_idx, alpha=0.4):
        heatmap = self.generate_heatmap(image, class_idx)
        overlayed = self.overlay_heatmap(image, heatmap, alpha)
        return heatmap, overlayed
    
    def analyze_layer_importance(self, images, labels):
        layer_importance = {}
        conv_layers = [layer for layer in self.model.layers if isinstance(layer, keras.layers.Conv2D)]
        for layer in conv_layers:
            temp_grad_cam = GradCAM(self.model, layer.name)
            heatmaps = []
            for img, label in zip(images, labels):
                heatmap = temp_grad_cam.generate_heatmap(img, label)
                heatmaps.append(heatmap)
            mean_importance = np.mean([np.mean(h) for h in heatmaps])
            layer_importance[layer.name] = mean_importance
        return layer_importance


# ---------------------------------------------------------------------------
# 2. Grad-CAM++  (improved Grad-CAM with better object localisation)
# ---------------------------------------------------------------------------

class GradCAMPlusPlus:
    """Grad-CAM++ implementation for improved visual explanations.
    
    Research Gap: Comprehensive benchmarking of multiple XAI techniques.
    Grad-CAM++ uses a weighted combination of positive partial derivatives
    of the last conv layer w.r.t. the class score, yielding sharper and
    better-localised heatmaps, especially for multi-instance images.
    """
    
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        if self.layer_name is None:
            self.layer_name = _find_last_conv_layer(self.model)
        self.grad_model = self._build_grad_model()
    
    def _build_grad_model(self):
        return models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
    
    def generate_heatmap(self, image, class_idx, eps=1e-8):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)  # (1, h, w, C)
        
        # First derivative
        first_deriv = grads
        # Second derivative
        second_deriv = grads * grads
        # Third derivative (approximate)
        third_deriv = grads * grads * grads
        
        # Sum over spatial dims
        global_sum = tf.reduce_sum(conv_outputs[0], axis=(0, 1))
        
        # Alpha weights (Grad-CAM++ formula)
        alpha_num = second_deriv[0]
        alpha_denom = 2.0 * second_deriv[0] + third_deriv[0] * global_sum + eps
        alpha = alpha_num / (alpha_denom + eps)
        
        # Weights: sum of alpha * ReLU(first derivative)
        weights = tf.reduce_sum(alpha * tf.maximum(first_deriv[0], 0), axis=(0, 1))
        
        # Weighted combination of feature maps
        heatmap = tf.reduce_sum(weights * conv_outputs[0], axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + eps)
        return heatmap.numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        return _overlay_heatmap(image, heatmap, alpha, colormap)
    
    def explain(self, image, class_idx, alpha=0.4):
        heatmap = self.generate_heatmap(image, class_idx)
        overlayed = self.overlay_heatmap(image, heatmap, alpha)
        return heatmap, overlayed


# ---------------------------------------------------------------------------
# 3. Score-CAM  (gradient-free, score-weighted CAM)
# ---------------------------------------------------------------------------

class ScoreCAM:
    """Score-CAM: Score-Weighted Visual Explanations for CNNs.
    
    Research Gap: Comprehensive benchmarking of multiple XAI techniques.
    Score-CAM is gradient-free — it uses the forward pass scores of
    masked input images to weight the activation maps, avoiding the
    noise introduced by gradient-based methods.
    """
    
    def __init__(self, model, layer_name=None, num_masks=None):
        self.model = model
        self.layer_name = layer_name
        if self.layer_name is None:
            self.layer_name = _find_last_conv_layer(self.model)
        self.num_masks = num_masks  # None = use all channels
    
    def generate_heatmap(self, image, class_idx, eps=1e-8):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        img_tensor = tf.cast(image, tf.float32)
        
        # Get activation maps
        activation_model = models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output]
        )
        activations = activation_model(img_tensor)  # (1, h, w, C)
        activations = activations[0]  # (h, w, C)
        
        h, w, C = activations.shape
        if self.num_masks is not None and self.num_masks < C:
            channel_indices = np.linspace(0, C - 1, self.num_masks, dtype=int)
        else:
            channel_indices = range(C)
        
        # Upsample each activation map to input size and mask the input
        scores = np.zeros(len(channel_indices), dtype=np.float32)
        
        for i, c in enumerate(channel_indices):
            act_map = activations[:, :, c].numpy()
            act_map_norm = (act_map - act_map.min()) / (act_map.max() - act_map.min() + eps)
            act_map_resized = cv2.resize(act_map_norm, (image.shape[2], image.shape[1]))
            mask = np.expand_dims(act_map_resized, axis=-1)  # (H, W, 1)
            
            # Masked input
            masked_input = img_tensor * mask
            
            # Score for target class
            if isinstance(self.model.output, list):
                pred = self.model(masked_input, training=False)[0]
            else:
                pred = self.model(masked_input, training=False)
            scores[i] = pred[0, class_idx].numpy()
        
        # Softmax over scores (normalise weights)
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / (exp_scores.sum() + eps)
        
        # Weighted sum of activation maps
        heatmap = np.zeros((h, w), dtype=np.float32)
        for i, c in enumerate(channel_indices):
            heatmap += weights[i] * activations[:, :, c].numpy()
        
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + eps)
        return heatmap
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        return _overlay_heatmap(image, heatmap, alpha, colormap)
    
    def explain(self, image, class_idx, alpha=0.4):
        heatmap = self.generate_heatmap(image, class_idx)
        overlayed = self.overlay_heatmap(image, heatmap, alpha)
        return heatmap, overlayed


# ---------------------------------------------------------------------------
# 4. Integrated Gradients  (path-attribution method)
# ---------------------------------------------------------------------------

class IntegratedGradientsExplainer:
    """Integrated Gradients explainer for pixel-level attribution.
    
    Research Gap: Comprehensive benchmarking of multiple XAI techniques.
    Integrated Gradients computes the integral of gradients along a
    straight-line path from a baseline (black image) to the input,
    producing a fine-grained attribution map that satisfies axiomatic
    properties (sensitivity, implementation invariance).
    """
    
    def __init__(self, model, baseline=None, num_steps=50):
        """
        Args:
            model: Trained Keras model
            baseline: Baseline input (defaults to black image)
            num_steps: Number of interpolation steps along the path
        """
        self.model = model
        self.baseline = baseline
        self.num_steps = num_steps
    
    def _get_baseline(self, image):
        if self.baseline is not None:
            return self.baseline
        return tf.zeros_like(image)
    
    def generate_attribution(self, image, class_idx, eps=1e-8):
        """
        Generate Integrated Gradients attribution map.
        
        Returns:
            Attribution map (H, W) — aggregated across channels
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        image_tensor = tf.cast(image, tf.float32)
        baseline = self._get_baseline(image_tensor)
        
        # Interpolate between baseline and input
        alphas = tf.linspace(0.0, 1.0, self.num_steps + 1)
        
        accumulated_grads = tf.zeros_like(image_tensor)
        
        for alpha in alphas[1:]:  # skip alpha=0 (baseline)
            with tf.GradientTape() as tape:
                tape.watch(image_tensor)
                interpolated = baseline + alpha * (image_tensor - baseline)
                if isinstance(self.model.output, list):
                    pred = self.model(interpolated, training=False)[0]
                else:
                    pred = self.model(interpolated, training=False)
                loss = pred[:, class_idx]
            
            grads = tape.gradient(loss, image_tensor)
            accumulated_grads = accumulated_grads + grads
        
        avg_grads = accumulated_grads / tf.cast(self.num_steps, tf.float32)
        
        # IG = (input - baseline) * avg_grads
        attributions = (image_tensor - baseline) * avg_grads
        attributions = attributions[0].numpy()  # (H, W, C)
        
        # Aggregate channels: sum of absolute values
        attr_map = np.sum(np.abs(attributions), axis=-1)
        attr_map = np.maximum(attr_map, 0)
        attr_map = attr_map / (attr_map.max() + eps)
        return attr_map
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        return _overlay_heatmap(image, heatmap, alpha, colormap)
    
    def explain(self, image, class_idx, alpha=0.4):
        heatmap = self.generate_attribution(image, class_idx)
        overlayed = self.overlay_heatmap(image, heatmap, alpha)
        return heatmap, overlayed


# ---------------------------------------------------------------------------
# 5. RISE  (Randomized Input Sampling for Explanation)
# ---------------------------------------------------------------------------

class RISEExplainer:
    """RISE: Randomized Input Sampling for Explanation of black-box models.
    
    Research Gap: Comprehensive benchmarking of multiple XAI techniques.
    RISE is model-agnostic: it probes the model with many randomly masked
    versions of the input and weights each mask by the model's output score
    for the target class, producing a saliency map without any gradient
    computation.
    """
    
    def __init__(self, model, num_masks=1000, mask_resolution=8, p=0.5):
        """
        Args:
            model: Trained Keras model
            num_masks: Number of random masks to generate
            mask_resolution: Resolution of the random mask (upsampled to image size)
            p: Probability of keeping a cell unmasked
        """
        self.model = model
        self.num_masks = num_masks
        self.mask_resolution = mask_resolution
        self.p = p
        self.masks = None
    
    def _generate_masks(self, input_size):
        """Generate random binary masks at low resolution then upsample."""
        cell_size = input_size[0] // self.mask_resolution
        up_size = (cell_size * self.mask_resolution, cell_size * self.mask_resolution)
        
        grid = np.random.rand(self.num_masks, self.mask_resolution, self.mask_resolution) < self.p
        grid = grid.astype('float32')
        
        masks = np.empty((self.num_masks, *up_size), dtype='float32')
        for i in range(self.num_masks):
            mask = cv2.resize(grid[i], up_size, interpolation=cv2.INTER_LINEAR)
            masks[i] = mask
        
        # Pad / crop to exact input size
        masks_padded = np.zeros((self.num_masks, input_size[0], input_size[1]), dtype='float32')
        for i in range(self.num_masks):
            h, w = masks[i].shape
            masks_padded[i, :h, :w] = masks[i]
        
        return masks_padded  # (N, H, W)
    
    def generate_heatmap(self, image, class_idx, eps=1e-8):
        """
        Generate RISE saliency map.
        
        Args:
            image: Input image (H, W, 3) or (1, H, W, 3)
            class_idx: Target class index
            
        Returns:
            Heatmap (H, W) normalised to [0, 1]
        """
        if len(image.shape) == 4:
            img = image[0]
        else:
            img = image.copy()
        
        input_h, input_w = img.shape[0], img.shape[1]
        
        # Generate masks
        masks = self._generate_masks((input_h, input_w))  # (N, H, W)
        self.masks = masks
        
        # Apply masks and get predictions
        img_float = img.astype('float32')
        scores = np.zeros(self.num_masks, dtype='float32')
        
        # Batch predictions for efficiency
        batch_size = 64
        for start in range(0, self.num_masks, batch_size):
            end = min(start + batch_size, self.num_masks)
            batch_masks = masks[start:end]  # (B, H, W)
            masked_images = batch_masks[:, :, :, np.newaxis] * img_float[np.newaxis, :, :, :]  # (B, H, W, 3)
            
            if isinstance(self.model.output, list):
                preds = self.model(masked_images, training=False)[0]
            else:
                preds = self.model(masked_images, training=False)
            
            scores[start:end] = preds[:, class_idx].numpy()
        
        # Weighted sum of masks
        saliency = np.zeros((input_h, input_w), dtype='float32')
        for i in range(self.num_masks):
            saliency += scores[i] * masks[i]
        
        saliency = np.maximum(saliency, 0)
        saliency = saliency / (saliency.max() + eps)
        return saliency
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        return _overlay_heatmap(image, heatmap, alpha, colormap)
    
    def explain(self, image, class_idx, alpha=0.4):
        heatmap = self.generate_heatmap(image, class_idx)
        overlayed = self.overlay_heatmap(image, heatmap, alpha)
        return heatmap, overlayed


# ---------------------------------------------------------------------------
# 6. XAI Fusion  — combine heatmaps from multiple methods
# ---------------------------------------------------------------------------

class XAIFusion:
    """Fuse heatmaps from multiple XAI methods to improve reliability.
    
    Research Gap: A comprehensive benchmarking and fusion of multiple XAI
    techniques to improve reliability.
    
    Supports:
    - Mean fusion: average of normalised heatmaps
    - Max fusion: element-wise maximum
    - Consensus fusion: intersection of top-k% regions across methods
    - Weighted fusion: user-specified or learned weights
    """
    
    def __init__(self, fusion_method='mean', weights=None):
        """
        Args:
            fusion_method: One of 'mean', 'max', 'consensus', 'weighted'
            weights: Dict mapping method names to weights (used only if fusion_method='weighted')
        """
        self.fusion_method = fusion_method
        self.weights = weights
    
    def fuse(self, heatmaps, top_k_percent=20):
        """
        Fuse a list of heatmaps into a single combined heatmap.
        
        Args:
            heatmaps: List of (H, W) numpy arrays, each normalised to [0, 1]
            top_k_percent: For 'consensus' method, the % of top pixels to consider
        
        Returns:
            Fused heatmap (H, W) normalised to [0, 1]
        """
        if not heatmaps:
            raise ValueError("No heatmaps provided for fusion")
        
        # Ensure all heatmaps have the same shape (resize to first heatmap's size)
        ref_shape = heatmaps[0].shape
        aligned = []
        for h in heatmaps:
            if h.shape != ref_shape:
                h = cv2.resize(h, (ref_shape[1], ref_shape[0]))
            aligned.append(h)
        
        stacked = np.stack(aligned, axis=0)  # (N, H, W)
        eps = 1e-8
        
        if self.fusion_method == 'mean':
            fused = np.mean(stacked, axis=0)
        
        elif self.fusion_method == 'max':
            fused = np.max(stacked, axis=0)
        
        elif self.fusion_method == 'consensus':
            # Binary mask per method: top-k% pixels
            masks = []
            for h in aligned:
                threshold = np.percentile(h, 100 - top_k_percent)
                masks.append((h >= threshold).astype('float32'))
            mask_stack = np.stack(masks, axis=0)
            # Consensus: fraction of methods that agree each pixel is important
            consensus = np.mean(mask_stack, axis=0)
            # Weight by mean heatmap intensity
            mean_heatmap = np.mean(stacked, axis=0)
            fused = consensus * mean_heatmap
        
        elif self.fusion_method == 'weighted':
            if self.weights is None:
                # Equal weights if not provided
                n = len(aligned)
                w = np.ones(n) / n
            else:
                # Map weights by order; assume heatmaps are passed in known order
                w = np.array([self.weights.get(i, 1.0 / len(aligned)) for i in range(len(aligned))])
                w = w / w.sum()
            fused = np.tensordot(w, stacked, axes=([0], [0]))
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        fused = np.maximum(fused, 0)
        fused = fused / (fused.max() + eps)
        return fused


# ---------------------------------------------------------------------------
# 7. XAI Benchmark  — quantitative comparison of XAI methods
# ---------------------------------------------------------------------------

class XAIBenchmark:
    """Quantitative benchmarking of multiple XAI methods.
    
    Research Gap: A comprehensive benchmarking and fusion of multiple XAI
    techniques to improve reliability.
    
    Metrics:
    - Deletion AUC: remove most important pixels first → lower AUC = better
    - Insertion AUC: add most important pixels first → higher AUC = better
    - Mean Increase in Confidence (when keeping important pixels)
    - Consistency score: pairwise agreement between methods (intersection over union)
    """
    
    def __init__(self, model, explainers_dict, num_steps=20):
        """
        Args:
            model: Trained Keras model
            explainers_dict: Dict mapping method name → explainer instance
                             e.g. {'Grad-CAM': GradCAM(model), 'Score-CAM': ScoreCAM(model), ...}
            num_steps: Number of steps for deletion/insertion curves
        """
        self.model = model
        self.explainers = explainers_dict
        self.num_steps = num_steps
    
    def _get_prediction_fn(self):
        """Return a function that takes images and returns class probabilities."""
        def predict(images):
            if isinstance(self.model.output, list):
                return self.model(images, training=False)[0].numpy()
            return self.model(images, training=False).numpy()
        return predict
    
    def deletion_score(self, image, heatmap, class_idx):
        """
        Deletion metric: progressively remove (grey out) the most important
        pixels according to the heatmap and measure the drop in class
        probability. Lower AUC = better explanation.
        """
        if len(image.shape) == 3:
            img = image.copy()
        else:
            img = image[0].copy()
        
        h, w = heatmap.shape
        predict = self._get_prediction_fn()
        
        # Flatten and sort pixel indices by importance (descending)
        flat = heatmap.flatten()
        order = np.argsort(flat)[::-1]
        
        total_pixels = len(order)
        step_size = max(1, total_pixels // self.num_steps)
        
        probs = []
        img_mod = img.copy().astype('float32')
        
        for step in range(self.num_steps + 1):
            pred = predict(np.expand_dims(img_mod, 0))
            probs.append(pred[0, class_idx])
            
            # Remove top pixels
            end = min((step + 1) * step_size, total_pixels)
            for idx in order[step * step_size:end]:
                r, c = divmod(idx, w)
                img_mod[r, c] = 0  # grey out
        
        return np.mean(probs)  # AUC approximation
    
    def insertion_score(self, image, heatmap, class_idx):
        """
        Insertion metric: start from a grey image and progressively add
        the most important pixels. Higher AUC = better explanation.
        """
        if len(image.shape) == 3:
            img = image.copy()
        else:
            img = image[0].copy()
        
        h, w = heatmap.shape
        predict = self._get_prediction_fn()
        
        flat = heatmap.flatten()
        order = np.argsort(flat)[::-1]
        
        total_pixels = len(order)
        step_size = max(1, total_pixels // self.num_steps)
        
        probs = []
        img_mod = np.zeros_like(img, dtype='float32')
        
        for step in range(self.num_steps + 1):
            pred = predict(np.expand_dims(img_mod, 0))
            probs.append(pred[0, class_idx])
            
            # Add top pixels
            end = min((step + 1) * step_size, total_pixels)
            for idx in order[step * step_size:end]:
                r, c = divmod(idx, w)
                img_mod[r, c] = img[r, c]
        
        return np.mean(probs)  # AUC approximation
    
    def consistency_score(self, heatmap1, heatmap2, top_k_percent=20):
        """
        Compute IoU of top-k% important regions between two heatmaps.
        Higher = more consistent.
        """
        t1 = np.percentile(heatmap1, 100 - top_k_percent)
        t2 = np.percentile(heatmap2, 100 - top_k_percent)
        
        mask1 = heatmap1 >= t1
        mask2 = heatmap2 >= t2
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        return intersection / union
    
    def benchmark(self, image, class_idx, top_k_percent=20):
        """
        Run full benchmark across all registered explainers.
        
        Args:
            image: Input image (H, W, 3)
            class_idx: Target class index
            top_k_percent: % of top pixels for consistency
        
        Returns:
            dict mapping method name → {
                'heatmap': np.array,
                'deletion_auc': float,
                'insertion_auc': float,
                'consistency': dict (pairwise IoU with other methods)
            }
        """
        results = {}
        heatmaps = {}
        
        # Generate heatmaps
        for name, explainer in self.explainers.items():
            print(f"  Generating heatmap: {name}...")
            heatmap = explainer.generate_heatmap(image, class_idx)
            heatmaps[name] = heatmap
        
        # Compute deletion / insertion for each
        for name, heatmap in heatmaps.items():
            print(f"  Computing deletion/insertion: {name}...")
            del_auc = self.deletion_score(image, heatmap, class_idx)
            ins_auc = self.insertion_score(image, heatmap, class_idx)
            
            # Pairwise consistency
            consistency = {}
            for other_name, other_hm in heatmaps.items():
                if other_name != name:
                    consistency[other_name] = self.consistency_score(
                        heatmap, other_hm, top_k_percent
                    )
            
            results[name] = {
                'heatmap': heatmap,
                'deletion_auc': float(del_auc),
                'insertion_auc': float(ins_auc),
                'consistency': consistency
            }
        
        return results
    
    def benchmark_summary(self, benchmark_results):
        """Print a formatted summary table of benchmark results."""
        print("\n" + "=" * 80)
        print("XAI Method Benchmark Summary")
        print("=" * 80)
        header = f"{'Method':<20} {'Deletion AUC':<15} {'Insertion AUC':<15} {'Avg Consistency':<15}"
        print(header)
        print("-" * 80)
        
        for name, res in benchmark_results.items():
            avg_cons = np.mean(list(res['consistency'].values())) if res['consistency'] else 0.0
            print(f"{name:<20} {res['deletion_auc']:<15.4f} {res['insertion_auc']:<15.4f} {avg_cons:<15.4f}")
        
        print("=" * 80)


# ---------------------------------------------------------------------------
# SHAP & LIME (kept from original)
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """SHAP explainer for model interpretability."""
    
    def __init__(self, model, background_samples=100):
        self.model = model
        self.background_samples = background_samples
        self.explainer = None
    
    def initialize(self, background_data):
        try:
            import shap
            if len(background_data) > self.background_samples:
                indices = np.random.choice(len(background_data), self.background_samples, replace=False)
                background_subset = background_data[indices]
            else:
                background_subset = background_data
            self.explainer = shap.GradientExplainer(self.model, background_subset)
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
    
    def explain(self, image, class_idx):
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize() first.")
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        shap_values = self.explainer.shap_values(image)
        return shap_values


class LIMEExplainer:
    """LIME explainer for model interpretability."""
    
    def __init__(self, model):
        self.model = model
    
    def predict_fn(self, images):
        predictions = self.model.predict(images, verbose=0)
        return predictions
    
    def explain(self, image, class_idx, num_samples=1000):
        try:
            from lime import lime_image
            from skimage.segmentation import mark_boundaries
            
            explainer = lime_image.LimeImageExplainer()
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            explanation = explainer.explain_instance(
                image[0].astype('double'),
                self.predict_fn,
                top_labels=5,
                hide_color=0,
                num_samples=num_samples
            )
            
            temp, mask = explanation.get_image_and_mask(
                class_idx,
                positive_only=True,
                num_features=10,
                hide_rest=False
            )
            
            return temp, mask, explanation
            
        except ImportError:
            print("LIME not installed. Install with: pip install lime")
            return None, None, None


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def visualize_explanation(original_image, heatmap, overlayed, class_name, save_path=None):
    """Visualize a single XAI explanation (original, heatmap, overlay)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlayed)
    axes[2].set_title(f'Overlayed (Class: {class_name})')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_multi_xai(original_image, heatmaps_dict, class_name, save_path=None):
    """Visualise heatmaps from multiple XAI methods side by side.
    
    Args:
        original_image: Original image (H, W, 3)
        heatmaps_dict: Dict mapping method name → (heatmap, overlayed)
        class_name: Name of predicted class
        save_path: Optional save path
    """
    n_methods = len(heatmaps_dict)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
    
    # Top row: heatmaps
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    for i, (name, (heatmap, _)) in enumerate(heatmaps_dict.items()):
        axes[0, i + 1].imshow(heatmap, cmap='jet')
        axes[0, i + 1].set_title(name)
        axes[0, i + 1].axis('off')
    
    # Bottom row: overlays
    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')
    
    for i, (name, (_, overlayed)) in enumerate(heatmaps_dict.items()):
        axes[1, i + 1].imshow(overlayed)
        axes[1, i + 1].set_title(f'{name} Overlay')
        axes[1, i + 1].axis('off')
    
    plt.suptitle(f'XAI Comparison — {class_name}', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_fusion(original_image, heatmaps_dict, fused_heatmap, fused_overlay,
                     class_name, save_path=None):
    """Visualise individual heatmaps + fused result."""
    n_methods = len(heatmaps_dict)
    n_cols = n_methods + 2  # original + methods + fused
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for i, (name, (heatmap, _)) in enumerate(heatmaps_dict.items()):
        axes[i + 1].imshow(heatmap, cmap='jet')
        axes[i + 1].set_title(name)
        axes[i + 1].axis('off')
    
    axes[-1].imshow(fused_overlay)
    axes[-1].set_title('Fused')
    axes[-1].axis('off')
    
    plt.suptitle(f'XAI Fusion — {class_name}', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_uncertainty(image, uncertainty_results, class_name, save_path=None):
    """Visualise uncertainty metrics alongside the prediction.
    
    Args:
        image: Original image (H, W, 3)
        uncertainty_results: Dict from MCDropoutModel.predict_with_uncertainty
        class_name: Predicted class name
        save_path: Optional save path
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title(f'Prediction: {class_name}')
    axes[0].axis('off')
    
    # Confidence bar
    confidence = uncertainty_results['confidence']
    axes[1].barh(['Confidence'], [confidence], color='green' if confidence > 0.8 else 'orange' if confidence > 0.5 else 'red')
    axes[1].set_xlim(0, 1)
    axes[1].set_title(f'Confidence: {confidence:.3f}')
    
    # Entropy
    entropy = uncertainty_results['entropy']
    max_entropy = np.log(uncertainty_results['mean_probs'].shape[-1])
    axes[2].barh(['Entropy'], [entropy], color='salmon')
    axes[2].set_xlim(0, max_entropy)
    axes[2].set_title(f'Entropy: {entropy:.3f} / {max_entropy:.3f}')
    
    # Epistemic uncertainty (mutual info)
    mi = uncertainty_results['mutual_info']
    axes[3].barh(['Mutual Info'], [mi], color='mediumpurple')
    axes[3].set_xlim(0, max_entropy)
    axes[3].set_title(f'Epistemic: {mi:.3f}')
    
    plt.suptitle('Uncertainty Estimation (MC Dropout)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_grading_uncertainty(image, cls_unc, grade_unc, cls_name, grade_name, save_path=None):
    """Visualise uncertainty for both classification and grading heads."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Classification
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'Class: {cls_name}')
    axes[0, 0].axis('off')
    
    axes[0, 1].barh(['Conf'], [cls_unc['confidence']], color='green')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_title(f'Cls Confidence: {cls_unc["confidence"]:.3f}')
    
    axes[0, 2].barh(['Entropy'], [cls_unc['entropy']], color='salmon')
    axes[0, 2].set_title(f'Cls Entropy: {cls_unc["entropy"]:.3f}')
    
    axes[0, 3].barh(['MI'], [cls_unc['mutual_info']], color='mediumpurple')
    axes[0, 3].set_title(f'Cls Epistemic: {cls_unc["mutual_info"]:.3f}')
    
    # Row 2: Grading
    axes[1, 0].imshow(image)
    axes[1, 0].set_title(f'Grade: {grade_name}')
    axes[1, 0].axis('off')
    
    axes[1, 1].barh(['Conf'], [grade_unc['confidence']], color='green')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_title(f'Grade Confidence: {grade_unc["confidence"]:.3f}')
    
    axes[1, 2].barh(['Entropy'], [grade_unc['entropy']], color='salmon')
    axes[1, 2].set_title(f'Grade Entropy: {grade_unc["entropy"]:.3f}')
    
    axes[1, 3].barh(['MI'], [grade_unc['mutual_info']], color='mediumpurple')
    axes[1, 3].set_title(f'Grade Epistemic: {grade_unc["mutual_info"]:.3f}')
    
    plt.suptitle('Multi-Task Uncertainty (Classification + Grading)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    from model import create_model
    
    model = create_model()
    grad_cam = GradCAM(model)
    print(f"Grad-CAM initialized with layer: {grad_cam.layer_name}")
    
    grad_cam_pp = GradCAMPlusPlus(model)
    print(f"Grad-CAM++ initialized with layer: {grad_cam_pp.layer_name}")
    
    score_cam = ScoreCAM(model)
    print(f"Score-CAM initialized with layer: {score_cam.layer_name}")
    
    ig = IntegratedGradientsExplainer(model)
    print("Integrated Gradients explainer initialized")
    
    rise = RISEExplainer(model, num_masks=100)
    print("RISE explainer initialized")
