import pandas as pd
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, output_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def process_parquet_file(file_path):
    """Process a single parquet file and extract relevant information."""
    logger.info(f"Processing {file_path}")
    df = pd.read_parquet(file_path)
    
    # Extract relevant columns
    processed_df = df[['image_url', 'caption_attribution_description', 'original_height', 'original_width', 'mime_type']].copy()
    
    # Rename columns for clarity
    processed_df = processed_df.rename(columns={
        'caption_attribution_description': 'caption',
        'original_height': 'height',
        'original_width': 'width'
    })
    
    # Add processing status
    processed_df['processed'] = False
    processed_df['face_detected'] = None
    processed_df['glasses_detected'] = None
    processed_df['face_size'] = None
    
    return processed_df

def main():
    """Download and process WIT dataset."""
    # Create data directory
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # WIT dataset files to download
    base_url = "https://huggingface.co/datasets/wikimedia/wit_base/resolve/main/data"
    files = [
        "train-00000-of-00330.parquet",
        "train-00001-of-00330.parquet"
    ]
    
    # Download and process files
    all_data = []
    for file in files:
        url = f"{base_url}/{file}"
        output_path = raw_dir / file
        
        # Download if not exists
        if not output_path.exists():
            logger.info(f"Downloading {file}")
            download_file(url, output_path)
        
        # Process file
        processed_df = process_parquet_file(output_path)
        all_data.append(processed_df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save processed data
    output_path = data_dir / "raw" / "wit_processed.parquet"
    output_path.parent.mkdir(exist_ok=True)
    combined_df.to_parquet(output_path)
    
    logger.info(f"✅ Processed {len(combined_df)} images")
    logger.info(f"✅ Saved processed data to {output_path}")

if __name__ == "__main__":
    main() 