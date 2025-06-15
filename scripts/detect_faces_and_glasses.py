import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import logging
import argparse
from typing import List, Dict, Any, Optional
import cv2
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import json
import matplotlib.pyplot as plt
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.glasses_utils import EyeglassesClassifier, validate_face_size
from utils.clip_utils import CLIPFaceGlassesClassifier
from utils.face_filter import FaceFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('glasses_detection.log')
    ]
)
logger = logging.getLogger(__name__)

# Set specific log levels for different components
logging.getLogger('clip_face_classifier').setLevel(logging.INFO)
logging.getLogger('face_filter').setLevel(logging.INFO)
logging.getLogger('ultralytics').setLevel(logging.WARNING)  # Reduce YOLO noise

# Suppress tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default configuration
DEFAULT_CONFIG = {
    'images_dir': Path("data/images"),
    'processed_dir': Path("data/processed"),
    'models_dir': Path("models"),
    'min_face_size': 100,  # Updated to 100x100
    'face_confidence': 0.3,
    'glasses_confidence': 0.5,
    'output_file': 'glasses.parquet',
    'checkpoint_file': 'checkpoint.json',
    'thumbnail_size': (100, 100),  # Size for preview thumbnails
    'glasses_dir': Path("data/glasses_images"),  # Directory for images with glasses
    'batch_size': 32,  # Number of images to process in parallel
    'max_workers': 4,   # Number of worker threads
    'save_formats': {   # Image saving formats and their settings
        'annotated': {
            'dir': Path("data/glasses_images/annotated"),
            'suffix': 'annotated'
        }
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect faces and eyeglasses in images')
    parser.add_argument('--confidence-threshold', type=float, 
                      default=DEFAULT_CONFIG['glasses_confidence'],
                      help='Confidence threshold for eyeglasses detection')
    parser.add_argument('--output-dir', type=str,
                      default=str(DEFAULT_CONFIG['processed_dir']),
                      help='Directory to save results')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from last checkpoint')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--use-clip', action='store_true',
                      help='Use CLIP-based classifier instead of ResNet')
    parser.add_argument('--save-glasses', action='store_true',
                      help='Save images with detected glasses')
    parser.add_argument('--batch-size', type=int,
                      default=DEFAULT_CONFIG['batch_size'],
                      help='Number of images to process in parallel')
    parser.add_argument('--max-workers', type=int,
                      default=DEFAULT_CONFIG['max_workers'],
                      help='Number of worker threads')
    parser.add_argument('--device', type=str,
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run models on (cuda/cpu)')
    return parser.parse_args()

def setup_directories(output_dir: Path, save_glasses: bool = False):
    """Create necessary directories if they don't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Created directory: {output_dir}")
    
    if save_glasses:
        # Create main glasses directory
        glasses_dir = DEFAULT_CONFIG['glasses_dir']
        glasses_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory for glasses images: {glasses_dir}")
        
        # Create directory for annotated images
        annotated_dir = DEFAULT_CONFIG['save_formats']['annotated']['dir']
        annotated_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory for annotated images: {annotated_dir}")

def load_models(use_clip: bool = False, device: str = 'cuda'):
    """Load both face detection and eyeglasses classification models."""
    # Load YOLOv8 face detection model
    try:
        face_model = YOLO(str(DEFAULT_CONFIG['models_dir'] / 'yolov8n-face.pt'))
        face_model.to(device)
    except FileNotFoundError:
        logger.error(f"Face detection model not found at {DEFAULT_CONFIG['models_dir'] / 'yolov8n-face.pt'}")
        raise
    logger.info(f"✅ Loaded YOLOv8 face detection model on {device}")
    
    # Load appropriate glasses classifier
    if use_clip:
        glasses_model = CLIPFaceGlassesClassifier()
        logger.info(f"✅ Loaded CLIP eyeglasses classifier")
    else:
        glasses_model = EyeglassesClassifier(device=device)
        logger.info(f"✅ Loaded ResNet eyeglasses classifier on {device}")
    
    return face_model, glasses_model

def create_thumbnail(image_path: Path, bbox: List[int], size: tuple = (100, 100)) -> str:
    """Create a base64-encoded thumbnail of the face region."""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return ""
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract face region
        x1, y1, x2, y2 = bbox
        face = img[y1:y2, x1:x2]
        
        # Resize to thumbnail size
        face_pil = Image.fromarray(face)
        face_pil.thumbnail(size)
        
        # Convert to base64
        buffered = BytesIO()
        face_pil.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logger.warning(f"Failed to create thumbnail: {str(e)}")
        return ""

def detect_faces(image_path, face_model, min_face_size=100, face_confidence=0.3):
    """
    Detect faces in an image using YOLOv8.
    
    Args:
        image_path: Path to the image
        face_model: YOLOv8 face detection model
        min_face_size: Minimum face size in pixels
        face_confidence: Minimum confidence for face detection
        
    Returns:
        List of face bounding boxes and confidences
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to read image: {image_path}")
            return []
        
        # Validate image dimensions
        height, width = img.shape[:2]
        if width < min_face_size or height < min_face_size:
            logger.warning(f"Image too small: {width}x{height}")
            return []
        
        # Detect faces
        results = face_model(img, conf=face_confidence)[0]
        faces = []
        
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < face_confidence:
                continue
                
            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate face size
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Check if face is large enough
            if face_width >= min_face_size and face_height >= min_face_size:
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'size': (face_width, face_height)
                })
                logger.debug(f"Found face: size={face_width}x{face_height}, conf={confidence:.2f}")
        
        return faces
        
    except Exception as e:
        logger.error(f"Error detecting faces in {image_path}: {str(e)}")
        return []

def save_glasses_image(image_path: Path, faces: List[Dict], output_dir: Path):
    """
    Save images with detected eyeglasses in original and annotated formats.
    Only saves images where transparent eyeglasses are confirmed.
    
    Args:
        image_path: Path to the original image
        faces: List of detected faces with glasses
        output_dir: Base directory to save images
    """
    try:
        # Filter faces to only include those with transparent eyeglasses
        eyeglasses_faces = [
            face for face in faces 
            if face['glasses_type'] == 'eyeglasses' and face['is_eyeglasses']
        ]
        
        if not eyeglasses_faces:
            logger.debug(f"No transparent eyeglasses detected in {image_path}")
            return
            
        # Read the original image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to read image: {image_path}")
            return
            
        # Save original image
        original_dir = DEFAULT_CONFIG['save_formats']['original']['dir']
        original_path = original_dir / f"{image_path.stem}_{DEFAULT_CONFIG['save_formats']['original']['suffix']}{image_path.suffix}"
        cv2.imwrite(str(original_path), img)
        logger.info(f"Saved original image with eyeglasses: {original_path}")
        
        # Create and save annotated image
        annotated_img = img.copy()
        for face in eyeglasses_faces:
            x1, y1, x2, y2 = face['bbox']
            conf = face['eyeglasses_conf']
            
            # Draw rectangle
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence score
            label = f"Eyeglasses ({conf:.2f})"
            cv2.putText(annotated_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save annotated image
        annotated_dir = DEFAULT_CONFIG['save_formats']['annotated']['dir']
        annotated_path = annotated_dir / f"{image_path.stem}_{DEFAULT_CONFIG['save_formats']['annotated']['suffix']}{image_path.suffix}"
        cv2.imwrite(str(annotated_path), annotated_img)
        logger.info(f"Saved annotated image with eyeglasses: {annotated_path}")
        
    except Exception as e:
        logger.error(f"Error saving glasses images for {image_path}: {str(e)}")

def save_glasses_images_batch(images_to_save: List[Dict], config: Dict):
    """
    Save a batch of annotated images with detected eyeglasses.
    
    Args:
        images_to_save: List of dictionaries containing image paths and faces with glasses
        config: Configuration dictionary
    """
    try:
        for image_data in images_to_save:
            image_path = Path(image_data['image_path'])
            faces = image_data['faces_with_glasses']
            
            # Read the original image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                continue
                
            # Create and save annotated image
            annotated_img = img.copy()
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                conf = face['eyeglasses_conf']
                
                # Draw rectangle
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with confidence score
                label = f"Eyeglasses ({conf:.2f})"
                cv2.putText(annotated_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save annotated image
            annotated_dir = config['save_formats']['annotated']['dir']
            annotated_path = annotated_dir / f"{image_path.stem}_{config['save_formats']['annotated']['suffix']}{image_path.suffix}"
            cv2.imwrite(str(annotated_path), annotated_img)
            logger.info(f"Saved annotated image with eyeglasses: {annotated_path}")
            
    except Exception as e:
        logger.error(f"Error saving batch of glasses images: {str(e)}")

def process_image(image_path, face_model, glasses_model, config, use_clip=False, save_glasses=False):
    """
    Process a single image to detect faces and eyeglasses.
    
    Args:
        image_path: Path to the image
        face_model: YOLOv8 face detection model
        glasses_model: Eyeglasses classification model
        config: Configuration dictionary
        use_clip: Whether to use CLIP-based classifier
        save_glasses: Whether to save images with detected glasses
        
    Returns:
        Dictionary with detection results
    """
    try:
        # Read image for face detection
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to read image: {image_path}")
            return None
            
        # Detect faces
        faces = detect_faces(
            image_path, 
            face_model,
            min_face_size=config['min_face_size'],
            face_confidence=config['face_confidence']
        )
        
        if not faces:
            logger.debug(f"No faces detected in {image_path}")
            return None
            
        logger.info(f"Found {len(faces)} faces in {image_path}")
        
        # Initialize face filter
        face_filter = FaceFilter(min_confidence=0.5)
        
        # Filter faces using DeepFace
        verified_faces = face_filter.filter_faces(img, faces)
        
        if not verified_faces:
            logger.info(f"No verified human faces in {image_path}")
            return None
            
        logger.info(f"Verified {len(verified_faces)} human faces in {image_path}")
            
        # Process each verified face
        processed_faces = []
        faces_with_glasses = []
        
        for i, face in enumerate(verified_faces):
            logger.info(f"Processing face {i+1}/{len(verified_faces)} in {image_path}")
            
            # Convert face region to PIL Image
            x1, y1, x2, y2 = face['bbox']
            face_img = Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            
            # Classify eyeglasses
            if use_clip:
                result = glasses_model.classify(face_img)
                if result is None:
                    logger.warning(f"Failed to classify face in {image_path}")
                    continue
                    
                # Create thumbnail
                thumbnail = create_thumbnail(
                    image_path,
                    face['bbox'],
                    config['thumbnail_size']
                )
                    
                # Add face results
                face_data = {
                    'bbox': face['bbox'],
                    'size': face['size'],
                    'face_confidence': face['confidence'],
                    'eyeglasses_conf': result['eyeglasses_conf'],
                    'sunglasses_conf': result['sunglasses_conf'],
                    'no_glasses_conf': result['no_glasses_conf'],
                    'glasses_type': result['glasses_type'],
                    'is_eyeglasses': result['is_eyeglasses'],
                    'thumbnail': thumbnail
                }
                processed_faces.append(face_data)
                
                # Track faces with glasses
                if result['is_eyeglasses']:
                    faces_with_glasses.append(face_data)
                    logger.info(
                        f"Found transparent eyeglasses in {image_path} (face {i+1}): "
                        f"conf={result['eyeglasses_conf']:.3f}, "
                        f"no_glasses_conf={result['no_glasses_conf']:.3f}"
                    )
                else:
                    logger.info(
                        f"No transparent eyeglasses in {image_path} (face {i+1}): "
                        f"eyeglasses_conf={result['eyeglasses_conf']:.3f}, "
                        f"no_glasses_conf={result['no_glasses_conf']:.3f}"
                    )
            else:
                # Preprocess face for ResNet
                face_tensor = glasses_model.preprocess_face(image_path, face['bbox'])
                if face_tensor is None:
                    logger.warning(f"Failed to preprocess face in {image_path}")
                    continue
                    
                # Classify with ResNet
                result = glasses_model.classify_face(face_tensor)
                if result is None:
                    logger.warning(f"Failed to classify face in {image_path}")
                    continue
                    
                # Create thumbnail
                thumbnail = create_thumbnail(
                    image_path,
                    face['bbox'],
                    config['thumbnail_size']
                )
                    
                # Add face results
                face_data = {
                    'bbox': face['bbox'],
                    'size': face['size'],
                    'face_confidence': face['confidence'],
                    'eyeglasses_conf': result['glasses_confidence'],
                    'sunglasses_conf': result['sunglasses_confidence'],
                    'no_glasses_conf': 1.0 - max(result['glasses_confidence'], result['sunglasses_confidence']),
                    'glasses_type': 'eyeglasses' if result['is_glasses'] else 'no_glasses',
                    'is_eyeglasses': result['is_glasses'],
                    'thumbnail': thumbnail
                }
                processed_faces.append(face_data)
                
                # Track faces with glasses
                if result['is_glasses']:
                    faces_with_glasses.append(face_data)
                    logger.info(f"Found glasses in {image_path} (face {i+1}): conf={result['glasses_confidence']:.3f}")
                else:
                    logger.info(f"No glasses in {image_path} (face {i+1}): conf={result['glasses_confidence']:.3f}")
        
        if not processed_faces:
            logger.debug(f"No valid faces processed in {image_path}")
            return None
            
        return {
            'image_path': str(image_path),
            'num_faces': len(processed_faces),
            'faces': processed_faces,
            'has_glasses': len(faces_with_glasses) > 0
        }
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def save_checkpoint(output_dir: Path, processed_images: List[str], results: List[Dict]):
    """Save checkpoint of processed images and their results."""
    checkpoint_file = output_dir / DEFAULT_CONFIG['checkpoint_file']
    checkpoint_data = {
        'processed_images': processed_images,
        'results': results
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    logger.info(f"Saved checkpoint with {len(processed_images)} images")

def load_checkpoint(output_dir: Path) -> tuple[List[str], List[Dict]]:
    """Load checkpoint of processed images and their results."""
    checkpoint_file = output_dir / DEFAULT_CONFIG['checkpoint_file']
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            return data.get('processed_images', []), data.get('results', [])
    return [], []

def save_results(results: List[Dict], output_dir: Path):
    """Save results to parquet file."""
    if results:
        df = pd.DataFrame(results)
        output_file = output_dir / DEFAULT_CONFIG['output_file']
        df.to_parquet(output_file)
        logger.info(f"✅ Saved results to {output_file}")
        
        # Log summary
        total_faces = sum(r['num_faces'] for r in results)
        total_glasses = sum(1 for r in results if r['has_glasses'])
        logger.info(f"Processed {len(results)} images with {total_faces} faces")
        logger.info(f"Found {total_glasses} images with glasses")

def process_image_batch(image_paths: List[Path], face_model, glasses_model, config, use_clip: bool, save_glasses: bool) -> List[Dict]:
    """Process a batch of images in parallel."""
    results = []
    images_to_save = []  # List to collect images that need saving
    
    for image_path in image_paths:
        try:
            result = process_image(image_path, face_model, glasses_model, config, use_clip, save_glasses)
            if result:
                results.append(result)
                # If image has glasses and saving is enabled, add to save list
                if save_glasses and result['has_glasses']:
                    images_to_save.append({
                        'image_path': str(image_path),
                        'faces_with_glasses': [face for face in result['faces'] if face['is_eyeglasses']]
                    })
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
    
    # Save all images with glasses in this batch
    if images_to_save:
        logger.info(f"Saving batch of {len(images_to_save)} images with glasses")
        save_glasses_images_batch(images_to_save, config)
    
    return results

def main():
    """Main function to process images and detect faces with eyeglasses."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Set a more verbose level for the main logger
        logging.getLogger().setLevel(logging.INFO)
    
    # Setup directories
    output_dir = Path(args.output_dir)
    setup_directories(output_dir, args.save_glasses)
    
    # Load models
    face_model, glasses_model = load_models(args.use_clip, args.device)
    
    # Get list of images
    image_files = list(DEFAULT_CONFIG['images_dir'].glob('*.jpg'))
    if not image_files:
        logger.error(f"No images found in {DEFAULT_CONFIG['images_dir']}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Load checkpoint if resuming
    processed_images, results = load_checkpoint(output_dir)
    if args.resume and processed_images:
        logger.info(f"Resuming from checkpoint: {len(processed_images)} images processed")
    
    # Filter out already processed images
    image_files = [f for f in image_files if str(f) not in processed_images]
    logger.info(f"Remaining images to process: {len(image_files)}")
    
    # Process images in batches
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for i in tqdm(range(0, len(image_files), args.batch_size), desc="Processing batches"):
            batch = image_files[i:i + args.batch_size]
            batch_results = process_image_batch(
                batch, face_model, glasses_model,
                DEFAULT_CONFIG, args.use_clip, args.save_glasses
            )
            
            if batch_results:
                results.extend(batch_results)
                processed_images.extend([str(f) for f in batch])
                
                # Save results immediately after each batch
                save_results(results, output_dir)
                
                # Save checkpoint every 10 batches
                if (i // args.batch_size) % 10 == 0:
                    save_checkpoint(output_dir, processed_images, results)
                    logger.info(f"Saved checkpoint: {len(processed_images)} images processed")
    
    # Save final results
    save_results(results, output_dir)
    
    # Log final statistics
    total_faces = sum(r['num_faces'] for r in results)
    total_glasses = sum(1 for r in results if r['has_glasses'])
    logger.info(f"Processing complete:")
    logger.info(f"- Total images processed: {len(results)}")
    logger.info(f"- Total faces detected: {total_faces}")
    logger.info(f"- Images with glasses: {total_glasses}")
    
    # Clean up checkpoint file after successful completion
    checkpoint_file = output_dir / DEFAULT_CONFIG['checkpoint_file']
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Cleaned up checkpoint file")

if __name__ == '__main__':
    main() 