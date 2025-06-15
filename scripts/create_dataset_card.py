from huggingface_hub import HfApi
import json

def create_dataset_card(repo_name):
    """Create a dataset card for the Hugging Face repository."""
    card_content = """---
language:
- en
license:
- unknown
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- image-classification
- object-detection
task_ids:
- face-detection
- multi-class-image-classification
---

# Glasses Detection Dataset

## Dataset Description

This dataset contains processed results from a face and glasses detection pipeline. It includes:
- Face detection results using YOLOv8
- Glasses classification using CLIP
- Bounding box coordinates
- Confidence scores
- Image metadata

### Dataset Structure

The dataset is stored in Parquet format with the following columns:
- `image_path`: Path to the original image
- `faces`: List of detected faces, each containing:
  - `bbox`: Bounding box coordinates [x1, y1, x2, y2]
  - `confidence`: Face detection confidence score
  - `is_eyeglasses`: Boolean indicating if glasses were detected
  - `eyeglasses_conf`: Confidence score for glasses detection
  - `thumbnail`: Base64 encoded face thumbnail

### Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Manireddy1508/glasses-detection-dataset")

# Access the data
for item in dataset['train']:
    print(f"Image: {item['image_path']}")
    for face in item['faces']:
        print(f"  Face confidence: {face['confidence']:.2f}")
        print(f"  Has glasses: {face['is_eyeglasses']}")
        print(f"  Glasses confidence: {face['eyeglasses_conf']:.2f}")
```

### Source Data

The original images were sourced from the Wikipedia-based Image Text (WIT) dataset and processed using:
- YOLOv8 for face detection
- CLIP for glasses classification
- Custom face verification heuristics

### License

This dataset is provided under the same license as the original WIT dataset.

### Citation

If you use this dataset, please cite:
```
@misc{glasses-detection-dataset,
  author = {Manireddy},
  title = {Glasses Detection Dataset},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\\url{https://huggingface.co/datasets/Manireddy1508/glasses-detection-dataset}}
}
```
"""
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Upload the dataset card
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset"
    )
    
    print(f"Dataset card uploaded to {repo_name}")

if __name__ == "__main__":
    create_dataset_card("Manireddy1508/glasses-detection-dataset") 