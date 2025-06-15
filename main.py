#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Scalable Eyeglasses Image Filtering Pipeline'
    )
    parser.add_argument(
        '--step',
        type=str,
        required=True,
        choices=[
            'download_wit',
            'extract_images',
            'detect_faces',
            'detect_eyeglasses',
            'filter_dataset',
            'upload_to_hf',
            'test_pipeline'
        ],
        help='Pipeline step to execute'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for processing (default: from config)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use (-1 for CPU)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    return parser

def run_pipeline_step(step: str, batch_size: Optional[int] = None, gpu: int = 0) -> None:
    """
    Execute the specified pipeline step.
    
    Args:
        step: Pipeline step to execute
        batch_size: Optional batch size override
        gpu: GPU device ID
    """
    try:
        # Import step module
        module_name = f"scripts.{step}"
        module = __import__(module_name, fromlist=['main'])
        
        # Update config if batch_size provided
        if batch_size is not None:
            from utils.image_utils import load_config
            config = load_config()
            config['dataset']['batch_size'] = batch_size
        
        # Run step
        logger.info(f"Starting pipeline step: {step}")
        module.main(gpu_id=gpu)
        logger.info(f"Completed pipeline step: {step}")
        
    except ImportError as e:
        logger.error(f"Failed to import pipeline step {step}: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in pipeline step {step}: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point."""
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Run pipeline step
    run_pipeline_step(args.step, args.batch_size, args.gpu)

if __name__ == "__main__":
    main() 