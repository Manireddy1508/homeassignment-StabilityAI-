import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import logging
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Constants
MIN_FACE_SIZE = 100
EYEGLASSES_CLASS_ID = 837  # ImageNet class for eyeglasses
SUNGLASSES_CLASS_ID = 835  # ImageNet class for sunglasses
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

class EyeglassesClassifier:
    def __init__(self, model_name='resnet18', device=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        """
        Initialize the eyeglasses classifier.
        
        Args:
            model_name: Name of the model to use ('resnet18' or 'mobilenet_v3_small')
            device: Device to run the model on (default: auto-detect)
            confidence_threshold: Minimum confidence threshold for eyeglasses detection
        """
        self.confidence_threshold = confidence_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        if model_name == 'resnet18':
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        elif model_name == 'mobilenet_v3_small':
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"Initialized {model_name} on {self.device}")
    
    def preprocess_face(self, image_path, bbox):
        """
        Preprocess a face crop from the original image.
        
        Args:
            image_path: Path to the original image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Preprocessed tensor or None if preprocessing fails
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract face region
            x1, y1, x2, y2 = map(int, bbox)
            face = img[y1:y2, x1:x2]
            
            # Check face size
            if face.shape[0] < MIN_FACE_SIZE or face.shape[1] < MIN_FACE_SIZE:
                logger.debug(f"Face too small: {face.shape}")
                return None
            
            # Convert to PIL Image
            face_pil = Image.fromarray(face)
            
            # Apply transforms
            tensor = self.transform(face_pil).unsqueeze(0)
            return tensor
            
        except Exception as e:
            logger.warning(f"Error preprocessing face from {image_path}: {str(e)}")
            return None
    
    def classify_face(self, face_tensor):
        """
        Classify if a face is wearing eyeglasses.
        
        Args:
            face_tensor: Preprocessed face tensor
            
        Returns:
            Dictionary with classification results
        """
        try:
            with torch.no_grad():
                face_tensor = face_tensor.to(self.device)
                outputs = self.model(face_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get probabilities for eyeglasses and sunglasses
                glasses_prob = probabilities[EYEGLASSES_CLASS_ID].item()
                sunglasses_prob = probabilities[SUNGLASSES_CLASS_ID].item()
                
                # Determine if wearing eyeglasses (not sunglasses)
                is_glasses = (
                    glasses_prob > self.confidence_threshold and
                    glasses_prob > sunglasses_prob
                )
                
                return {
                    'is_glasses': is_glasses,
                    'glasses_confidence': glasses_prob,
                    'sunglasses_confidence': sunglasses_prob
                }
                
        except Exception as e:
            logger.warning(f"Error in classification: {str(e)}")
            return None

def validate_face_size(bbox):
    """
    Validate if face size meets minimum requirements.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        bool: True if face size is valid
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width >= MIN_FACE_SIZE and height >= MIN_FACE_SIZE 