import torch
import logging
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class CLIPFaceGlassesClassifier:
    """
    A CLIP-based classifier for detecting eyeglasses on human faces.
    Uses improved prompts specifically designed for transparent eyeglass detection.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP-based glasses classifier.
        
        Args:
            model_name: Name of the CLIP model to use
        """
        self.logger = logging.getLogger('clip_face_classifier')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Updated prompts for better accuracy
        self.prompts = [
            "a real human face wearing transparent eyeglasses",
            "a real human face wearing sunglasses",
            "a real human face without any glasses"
        ]
        
        # Pre-compute text embeddings
        self.text_inputs = self.processor(
            text=self.prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.model.get_text_features(**self.text_inputs)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        
        self.logger.info(f"Initialized CLIP classifier with model: {model_name}")
        self.logger.info(f"Using device: {self.device}")

    @torch.no_grad()
    def classify(self, face_img: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Classify if a face is wearing glasses using CLIP.
        
        Args:
            face_img: PIL Image of the face
            
        Returns:
            Dictionary with classification results or None if classification fails
        """
        try:
            # Process image
            inputs = self.processor(
                images=face_img,
                text=self.prompts,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get image and text features
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
            
            # Get confidence scores
            eyeglasses_conf = float(probs[0])
            sunglasses_conf = float(probs[1])
            no_glasses_conf = float(probs[2])
            
            # Log all confidence scores
            logger.info(
                f"CLIP confidence scores - Eyeglasses: {eyeglasses_conf:.3f}, "
                f"Sunglasses: {sunglasses_conf:.3f}, "
                f"No glasses: {no_glasses_conf:.3f}"
            )
            
            # Apply threshold-based rules
            is_eyeglasses = (
                eyeglasses_conf > 0.5 and  # High confidence for eyeglasses
                no_glasses_conf < 0.3 and   # Low confidence for no glasses
                eyeglasses_conf > sunglasses_conf  # Higher confidence than sunglasses
            )
            
            is_sunglasses = (
                sunglasses_conf > 0.5 and   # High confidence for sunglasses
                no_glasses_conf < 0.3 and    # Low confidence for no glasses
                sunglasses_conf > eyeglasses_conf  # Higher confidence than eyeglasses
            )
            
            # Determine glasses type
            if is_eyeglasses:
                glasses_type = "eyeglasses"
            elif is_sunglasses:
                glasses_type = "sunglasses"
            else:
                glasses_type = "no_glasses"
            
            return {
                "eyeglasses_conf": eyeglasses_conf,
                "sunglasses_conf": sunglasses_conf,
                "no_glasses_conf": no_glasses_conf,
                "glasses_type": glasses_type,
                "is_eyeglasses": is_eyeglasses,
                "is_sunglasses": is_sunglasses
            }
            
        except Exception as e:
            logger.error(f"Error classifying face: {str(e)}")
            return None

    def get_face_size(self, bbox: Tuple[int, int, int, int]) -> int:
        """
        Calculate face size from bounding box.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Face size in pixels (width * height)
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height

    def is_valid_face_size(self, bbox: Tuple[int, int, int, int], min_size: int = 100) -> bool:
        """
        Check if face size meets minimum requirements.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            min_size: Minimum face size in pixels
            
        Returns:
            True if face size is valid
        """
        face_size = self.get_face_size(bbox)
        return face_size >= min_size 