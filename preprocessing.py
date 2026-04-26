import cv2
import numpy as np
from pathlib import Path


class BrainMRIPreprocessor:
    """Preprocessing pipeline for brain MRI images as described in the paper."""
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target size for resizing (width, height)
        """
        self.target_size = target_size
    
    def crop_brain_region(self, image):
        """
        Crop the brain region from the MRI image.
        
        Steps:
        1. Convert to grayscale
        2. Apply thresholding to separate object from background
        3. Apply morphological operations (erosion and dilation) to remove noise
        4. Find the largest contour
        5. Get extreme points for cropping
        6. Crop with extra pixels
        
        Args:
            image: Input MRI image (RGB or grayscale)
            
        Returns:
            Cropped image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours found, return original image
            return image
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get extreme points
        x_coords = largest_contour[:, 0, 0]
        y_coords = largest_contour[:, 0, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add extra pixels (padding)
        padding = 10
        x_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Crop the image
        if len(image.shape) == 3:
            cropped = image[y_min:y_max, x_min:x_max]
        else:
            cropped = image[y_min:y_max, x_min:x_max]
        
        return cropped
    
    def normalize(self, image):
        """
        Normalize pixel values to range [0, 255].
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        # Normalize to [0, 1]
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    
    def resize(self, image):
        """
        Resize image to target size.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        return resized
    
    def preprocess(self, image):
        """
        Apply full preprocessing pipeline.
        
        Args:
            image: Input MRI image
            
        Returns:
            Preprocessed image
        """
        # Step 1: Crop brain region
        cropped = self.crop_brain_region(image)
        
        # Step 2: Normalize
        normalized = self.normalize(cropped)
        
        # Step 3: Resize
        resized = self.resize(normalized)
        
        return resized
    
    def preprocess_batch(self, images):
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of preprocessed images
        """
        return [self.preprocess(img) for img in images]


def load_and_preprocess_dataset(data_dir, preprocessor, classes=None, class_mapping=None):
    """
    Load and preprocess dataset from directory structure.
    
    Expected structure:
    data_dir/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            ...
    
    Args:
        data_dir: Path to dataset directory
        preprocessor: BrainMRIPreprocessor instance
        classes: List of class names (if None, inferred from subdirectories)
        class_mapping: Dictionary mapping folder names to standard class names
        
    Returns:
        images: List of preprocessed images
        labels: List of corresponding labels
        class_names: List of class names
    """
    data_path = Path(data_dir)
    
    # Default class mapping for different dataset naming conventions
    if class_mapping is None:
        class_mapping = {
            'glioma': 'Glioma',
            'glioma_tumor': 'Glioma',
            'meningioma': 'Meningioma',
            'meningioma_tumor': 'Meningioma',
            'notumor': 'No Tumor',
            'no_tumor': 'No Tumor',
            'pituitary': 'Pituitary',
            'pituitary_tumor': 'Pituitary'
        }
    
    if classes is None:
        # Get subdirectories and map to standard names
        raw_classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        classes = [class_mapping.get(c, c) for c in raw_classes]
        # Remove duplicates while preserving order
        seen = set()
        classes = [c for c in classes if not (c in seen or seen.add(c))]
    
    images = []
    labels = []
    
    # Create reverse mapping from standard name to folder names
    folder_to_class = {}
    for folder_name, standard_name in class_mapping.items():
        if standard_name in classes:
            folder_to_class[folder_name] = standard_name
    
    # Load images from all matching folders
    for folder_name, standard_name in folder_to_class.items():
        folder_path = data_path / folder_name
        if not folder_path.exists():
            continue
        
        class_idx = classes.index(standard_name)
        image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png')) + list(folder_path.glob('*.jpeg'))
        
        for img_file in image_files:
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preprocess
            preprocessed = preprocessor.preprocess(img)
            
            images.append(preprocessed)
            labels.append(class_idx)
    
    return np.array(images), np.array(labels), classes
