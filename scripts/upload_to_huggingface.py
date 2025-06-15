import os
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
import argparse
from pathlib import Path

def create_dataset_from_parquet(parquet_path):
    """Create a Hugging Face dataset from the Parquet file."""
    # Read the Parquet file
    df = pd.read_parquet(parquet_path)
    
    # Convert to Hugging Face dataset
    dataset = Dataset.from_pandas(df)
    return dataset

def upload_to_huggingface(dataset, repo_name, private=True):
    """Upload dataset to Hugging Face Hub."""
    # Initialize Hugging Face API
    api = HfApi()
    
    try:
        # Create a private repository
        create_repo(repo_name, private=private, repo_type="dataset")
        print(f"Created repository: {repo_name}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Push the dataset to the hub
    dataset.push_to_hub(repo_name)
    print(f"Successfully uploaded dataset to: {repo_name}")

def main():
    parser = argparse.ArgumentParser(description='Upload processed data to Hugging Face')
    parser.add_argument('--parquet', type=str, default='data/processed/glasses.parquet',
                      help='Path to Parquet file')
    parser.add_argument('--repo-name', type=str, required=True,
                      help='Name of the Hugging Face repository (format: username/dataset-name)')
    parser.add_argument('--public', action='store_true',
                      help='Make the dataset public (default: private)')
    args = parser.parse_args()
    
    # Check if Parquet file exists
    if not os.path.exists(args.parquet):
        print(f"Error: Parquet file not found at {args.parquet}")
        return
    
    # Create dataset from Parquet
    print("Creating dataset from Parquet file...")
    dataset = create_dataset_from_parquet(args.parquet)
    
    # Upload to Hugging Face
    print("Uploading to Hugging Face...")
    upload_to_huggingface(dataset, args.repo_name, not args.public)
    
    print("\nUpload complete!")
    print(f"Dataset available at: https://huggingface.co/datasets/{args.repo_name}")

if __name__ == "__main__":
    main() 