import pandas as pd
import requests
from pathlib import Path
import logging
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RAW_DIR = Path('data/raw')
IMAGES_DIR = Path('data/images')
BATCH_SIZE = 100
MAX_RETRIES = 3
TIMEOUT = 10
MAX_WORKERS = 10

def setup_directories():
    """Create necessary directories if they don't exist."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created images directory at {IMAGES_DIR}")

def download_image(url, image_id):
    output_path = IMAGES_DIR / f"{image_id}.jpg"
    if output_path.exists():
        return True
    headers = {
        "User-Agent": "WIT-Eyeglasses-Filtering/1.0 (mani@example.com)"
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=TIMEOUT, headers=headers)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.warning(f"Failed to download {url}: {str(e)}")
                return False
            time.sleep(1)

def process_batch(batch_df):
    """Process a batch of URLs using thread pool."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, row in batch_df.iterrows():
            url = row['image_url']
            image_id = f"image_{idx:08d}"  # Generate unique image ID
            futures.append(executor.submit(download_image, url, image_id))
        
        # Wait for all downloads to complete
        for future in futures:
            future.result()

def download_images():
    """Main function to download images from URLs."""
    setup_directories()
    
    # Read the parquet file
    input_file = RAW_DIR / 'wit_processed.parquet'
    if not input_file.exists():
        logger.error(f"Input file {input_file} not found")
        return
    
    try:
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded {len(df)} rows from {input_file}")
        
        # Process in batches
        total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in tqdm(range(0, len(df), BATCH_SIZE), total=total_batches, desc="Downloading images"):
            batch_df = df.iloc[i:i + BATCH_SIZE]
            process_batch(batch_df)
            
        logger.info("Image download completed")
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    download_images() 