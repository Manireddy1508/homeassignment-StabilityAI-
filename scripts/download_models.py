import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
import logging
from pathlib import Path
import argparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_yolo_model(models_dir):
    """Download YOLOv8n-face model for face detection if not present."""
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    yolo_path = models_dir / "yolov8n-face.pt"
    
    if yolo_path.exists():
        logger.info(f"YOLOv8n-face model already exists at {yolo_path}. Skipping download.")
        return True
        
    logger.info("Downloading YOLOv8n-face model...")
    try:
        # Try to copy from existing models directory if available
        existing_yolo = Path("models/yolov8n-face.pt")
        if existing_yolo.exists():
            import shutil
            shutil.copy2(existing_yolo, yolo_path)
            logger.info(f"✅ Copied YOLOv8n-face model from {existing_yolo} to {yolo_path}")
            return True
            
        # If not found locally, try to download
        model = YOLO('yolov8n-face.pt')
        model.save(yolo_path)
        logger.info(f"✅ Saved YOLOv8n-face model to {yolo_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error downloading YOLOv8n-face model: {str(e)}")
        return False

def download_clip_model(models_dir):
    """Download CLIP model for glasses classification if not present."""
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    clip_path = models_dir / "clip"
    
    # Check if CLIP model files already exist
    required_files = [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",  # Changed from pytorch_model.bin
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    if clip_path.exists() and all((clip_path / f).exists() for f in required_files):
        logger.info(f"CLIP model already exists at {clip_path}. Skipping download.")
        return True
        
    logger.info("Downloading CLIP model...")
    try:
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        # Save with safetensors format
        model.save_pretrained(clip_path, safe_serialization=True)
        processor.save_pretrained(clip_path)
        
        # Verify all required files are present
        for file in required_files:
            if not (clip_path / file).exists():
                logger.error(f"❌ Missing required file: {file}")
                return False
                
        logger.info(f"✅ Saved CLIP model and processor to {clip_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error downloading CLIP model: {str(e)}")
        return False

def verify_downloads(models_dir):
    """Verify that all required models are downloaded."""
    models_dir = Path(models_dir)
    required_files = [
        models_dir / "yolov8n-face.pt",
        models_dir / "clip" / "config.json",
        models_dir / "clip" / "preprocessor_config.json",
        models_dir / "clip" / "model.safetensors",  # Changed from pytorch_model.bin
        models_dir / "clip" / "tokenizer.json",
        models_dir / "clip" / "tokenizer_config.json",
        models_dir / "clip" / "special_tokens_map.json"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        logger.error("❌ Missing required model files:")
        for f in missing_files:
            logger.error(f"  - {f}")
        return False
        
    logger.info("✅ All required model files are present")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download required models for the pipeline.")
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to save models (default: models)')
    args = parser.parse_args()
    models_dir = args.models_dir
    
    logger.info(f"Starting model downloads to {models_dir} ...")
    yolo_success = download_yolo_model(models_dir)
    clip_success = download_clip_model(models_dir)
    verification_success = verify_downloads(models_dir)
    
    if yolo_success and clip_success and verification_success:
        logger.info("✅ All models downloaded and verified successfully!")
    else:
        logger.error("❌ Some models failed to download or verify. Please check the logs above.")

if __name__ == "__main__":
    main() 