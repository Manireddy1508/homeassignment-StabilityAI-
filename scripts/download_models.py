import torch
import torchvision.models as models
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name):
    """Download and save a model."""
    logger.info(f"Downloading {model_name}...")
    try:
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Save the model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"✅ Saved {model_name} to {model_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error downloading {model_name}: {str(e)}")
        return False

def main():
    """Download all required models."""
    logger.info("Starting model downloads...")
    
    # Download ResNet18
    resnet_success = download_model('resnet18')
    
    # Download MobileNetV3
    mobilenet_success = download_model('mobilenet_v3_small')
    
    # Summary
    if resnet_success and mobilenet_success:
        logger.info("✅ All models downloaded successfully!")
    else:
        logger.error("❌ Some models failed to download. Please check the logs above.")

if __name__ == "__main__":
    main() 