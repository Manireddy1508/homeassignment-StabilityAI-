import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class FaceFilter:
    """Filter faces using basic heuristics to ensure they are real human faces."""
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize the face filter.
        
        Args:
            min_confidence: Minimum confidence threshold for face detection
        """
        self.min_confidence = min_confidence
        # Minimum face size (width * height) to consider it valid
        self.min_face_size = 100 * 100  # 100x100 pixels
        logger.info(f"Initialized FaceFilter with confidence threshold: {min_confidence}")
    
    def verify_face(self, image: np.ndarray, bbox: List[int]) -> bool:
        """
        Verify if a detected face is likely a real human face using basic heuristics.
        
        Args:
            image: Full image as numpy array
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            bool: True if face passes basic verification, False otherwise
        """
        try:
            # Extract face region
            x1, y1, x2, y2 = bbox
            face_img = image[y1:y2, x1:x2]
            
            # Check face size
            face_width = x2 - x1
            face_height = y2 - y1
            face_size = face_width * face_height
            
            if face_size < self.min_face_size:
                logger.debug(f"Face too small: {face_width}x{face_height}")
                return False
            
            # Check aspect ratio (typical human faces have aspect ratio between 0.5 and 2.0)
            aspect_ratio = face_width / face_height
            if not (0.5 <= aspect_ratio <= 2.0):
                logger.debug(f"Invalid face aspect ratio: {aspect_ratio:.2f}")
                return False
            
            # Convert to grayscale for additional checks
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Check if the face region has reasonable contrast
            contrast = np.std(gray)
            if contrast < 20:  # Minimum contrast threshold
                logger.debug(f"Face region has low contrast: {contrast:.2f}")
                return False
            
            logger.debug(f"Face passed verification: size={face_width}x{face_height}, aspect={aspect_ratio:.2f}, contrast={contrast:.2f}")
            return True
            
        except Exception as e:
            logger.warning(f"Face verification failed: {str(e)}")
            return False
    
    def filter_faces(self, image: np.ndarray, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a list of detected faces to keep only verified human faces.
        
        Args:
            image: Full image as numpy array
            faces: List of detected faces with bounding boxes
            
        Returns:
            List of verified faces
        """
        verified_faces = []
        
        for face in faces:
            if self.verify_face(image, face['bbox']):
                verified_faces.append(face)
                logger.info(f"Verified human face at {face['bbox']}")
            else:
                logger.info(f"Filtered out non-human face detection at {face['bbox']}")
        
        return verified_faces 