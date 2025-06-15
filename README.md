# Face and Eyeglasses Detection Pipeline

A robust pipeline for filtering large-scale image datasets to identify human faces wearing transparent eyeglasses. Built for research and dataset curation, this system combines YOLOv8 face detection with CLIP-based eyeglasses classification.

## 🏗️ Project Overview

This pipeline processes large image datasets to:
1. Detect human faces using YOLOv8 (minimum 100x100 pixels)
2. Verify faces using size, aspect ratio, and contrast checks
3. Classify eyeglasses using CLIP's zero-shot learning
4. Save results in Parquet format with detailed metadata
5. Optionally save annotated images with bounding boxes

The goal is to create a filtered dataset of faces wearing transparent eyeglasses, excluding sunglasses and non-glasses faces.

## 🗂️ Project Structure

```
.
├── config/                 # Configuration files
├── data/
│   ├── images/            # Input images
│   ├── processed/         # Processed results and checkpoints
│   ├── glasses_images/    # Images with detected glasses
│   │   └── annotated/     # Annotated images with bounding boxes
│   ├── raw/              # Raw dataset files
│   └── final/            # Final processed datasets
├── models/               # ML model weights
├── scripts/
│   ├── detect_faces_and_glasses.py  # Main processing script (entry point)
│   ├── test_single_image.py        # Test script for single images
│   ├── display_glasses_images.py   # Visualization script
│   ├── download_images.py          # Image download utility
│   ├── download_wit.py            # WIT dataset downloader
│   └── download_models.py         # Model download utility
├── utils/
│   ├── clip_utils.py     # CLIP-based glasses classifier
│   ├── face_filter.py    # Face verification utilities
│   └── glasses_utils.py  # Additional glasses detection utilities
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

### Key Components

#### `scripts/detect_faces_and_glasses.py`
- Main entry point for the pipeline
- Implements batch processing and checkpointing
- Manages model loading and image processing pipeline
- Handles result saving in Parquet format

#### `clip_utils.py`
- CLIP-based glasses classifier implementation
- Custom prompts for transparent eyeglasses detection
- Zero-shot classification with confidence thresholds
- Handles model initialization and inference

#### `face_filter.py`
- Face verification using heuristics
- Size, aspect ratio, and contrast checks
- Filters out non-human face detections
- Ensures minimum face quality standards

#### `glasses_utils.py`
- Additional glasses detection utilities
- Optional ResNet-based classifier
- Face size validation
- Image preprocessing functions

## 🛠️ Installation Instructions

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Manireddy1508/homeassignment-StabilityAI-.git
cd homeassignment-StabilityAI-
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLOv8 face detection model:
```bash
mkdir -p models
wget https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt -O models/yolov8n-face.pt
```

## 🖼️ Dataset Preparation

The pipeline is designed to work with the WIT (Wikimedia Image Text) dataset. To prepare your dataset:

1. Download WIT dataset from Hugging Face:
```bash
pip install datasets
python -c "from datasets import load_dataset; dataset = load_dataset('wikimedia/wit')"
```

2. Extract images to `data/images/`:
```bash
mkdir -p data/images
# Copy images from WIT dataset to data/images/
```

## 🚀 Getting Started

### Step 1: Download Required Models
First, download the necessary ML models:
```bash
python scripts/download_models.py
```
This will download:
- YOLOv8 face detection model
- CLIP model for glasses classification
Models will be saved in the `models/` directory.

### Step 2: Download WIT Dataset
Download the Wikipedia-based Image Text (WIT) dataset:
```bash
python scripts/download_wit.py
```
This will:
- Create `data/raw/` directory
- Download dataset metadata and image URLs
- Prepare the dataset for image downloading

### Step 3: Download Images
Download the actual images from the WIT dataset:
```bash
python scripts/download_images.py
```
This will:
- Read the WIT dataset from `data/raw/`
- Download images to `data/images/` directory
- Handle failed downloads and retries
- Create a clean dataset for processing

### Step 4: Run Face and Glasses Detection
Process the images to detect faces and glasses:
```bash
python scripts/detect_faces_and_glasses.py --use-clip --save-glasses --verbose
```
This will:
- Load the downloaded models
- Process images in batches
- Detect faces using YOLOv8
- Verify faces using heuristics
- Classify glasses using CLIP
- Save annotated images to `data/glasses_images/annotated/`
- Save results to `data/processed/glasses.parquet`

### Step 5: Upload to Hugging Face (Optional)
Upload the processed data to Hugging Face for private storage:
```bash
# First, login to Hugging Face
huggingface-cli login

# Then upload the dataset
python scripts/upload_to_huggingface.py --repo-name "your-username/glasses-detection-dataset"
```
This will:
- Create a private dataset repository on Hugging Face
- Upload the processed data from `glasses.parquet`
- Make the dataset available for private access

To make the dataset public, add the `--public` flag:
```bash
python scripts/upload_to_huggingface.py --repo-name "your-username/glasses-detection-dataset" --public
```

### Step 6: View Results (Optional)
View the detected faces with glasses:
```bash
python scripts/display_glasses_images.py --parquet data/processed/glasses.parquet
```
This will:
- Load the processed results
- Display images with detected glasses
- Show bounding boxes and confidence scores

### Command Line Arguments

#### For `detect_faces_and_glasses.py`:
```bash
python scripts/detect_faces_and_glasses.py [options]

Options:
  --confidence-threshold FLOAT  Minimum confidence for face detection (default: 0.5)
  --output-dir PATH            Output directory for results (default: data/processed)
  --resume                     Resume from last checkpoint
  --verbose                    Enable verbose logging
  --use-clip                   Use CLIP model for glasses detection
  --save-glasses              Save images with detected glasses
  --batch-size INT            Number of images to process in each batch (default: 32)
  --max-workers INT           Maximum number of worker processes (default: 4)
```

#### For `display_glasses_images.py`:
```bash
python scripts/display_glasses_images.py [options]

Options:
  --parquet PATH              Path to Parquet file (default: data/processed/glasses.parquet)
  --min-confidence FLOAT      Minimum confidence for glasses detection (default: 0.7)
  --show-thumbnails          Show face thumbnails instead of full images
```

### Expected Directory Structure After Running
```
.
├── models/                  # Downloaded ML models
├── data/
│   ├── raw/                # WIT dataset metadata
│   ├── images/             # Downloaded images
│   ├── processed/          # Results and checkpoints
│   │   └── glasses.parquet # Processed results
│   └── glasses_images/     # Images with detected glasses
│       └── annotated/      # Annotated images with bounding boxes
```

## 📊 Output

### Results Format

1. **Parquet File** (`data/processed/glasses.parquet`):
   - `image_path`: Path to source image
   - `num_faces`: Number of faces detected
   - `faces`: List of detected faces with:
     - Bounding box coordinates
     - Face size and confidence
     - Eyeglasses classification scores
     - Base64-encoded thumbnail

2. **Checkpoint File** (`data/processed/checkpoint.json`):
   - List of processed images
   - Results for each image
   - Updated every 10 batches

3. **Annotated Images** (`data/glasses_images/annotated/`):
   - Original images with bounding boxes
   - Confidence scores displayed
   - Only saved if `--save-glasses` is enabled

## 🧠 Model Design & Tech Choices

### YOLOv8 Face Detection
- Chosen for high accuracy and speed
- Minimum face size: 100x100 pixels
- Confidence threshold: 0.3
- Efficient batch processing

### CLIP-based Classification
- Zero-shot learning for eyeglasses detection
- Custom prompts for transparent glasses
- Three-class classification:
  - Transparent eyeglasses
  - Sunglasses
  - No glasses
- Confidence threshold: 0.5

### Architecture Decisions
- ThreadPoolExecutor for parallel processing
- Modular design for easy model swapping
- Checkpointing for resumable processing
- Parquet format for efficient storage

## 📈 Scaling & Deployment Notes

### Scaling to Large Datasets
1. **Parallel Processing**:
   - Increase `--batch-size` and `--max-workers`
   - Use multiple GPUs with `torch.nn.DataParallel`

2. **Cloud Deployment**:
   - AWS/GCP instance with GPU
   - S3/GCS for image storage
   - EMR/Dataproc for distributed processing

3. **Dataset Versioning**:
   - Hugging Face for processed datasets
   - Git LFS for large files
   - DVC for data versioning

### Performance Optimization
- GPU: ~100ms per image
- CPU: ~500ms per image
- Memory: ~2GB per batch
- Storage: ~1KB per face metadata

## 🚀 Reproducibility Checklist

1. **Environment**:
   - Python 3.8+
   - CUDA 11.7+ (for GPU)
   - All dependencies in `requirements.txt`

2. **Data**:
   - WIT dataset downloaded
   - YOLOv8 model in `models/`
   - Images in `data/images/`

3. **Configuration**:
   - Default values in `DEFAULT_CONFIG`
   - Checkpointing enabled
   - Logging configured

4. **Running**:
   - Single command execution
   - Resumable from checkpoints
   - Consistent results across runs

## 📄 License and Acknowledgements

### License
[Add your license information here]

### Acknowledgements
- YOLOv8 for face detection
- OpenAI CLIP for zero-shot classification
- Wikimedia for the WIT dataset
- Hugging Face for model hosting

### Hugging Face Upload
Processed datasets should be uploaded to Hugging Face as private datasets:
1. Create a new private dataset
2. Upload the Parquet file
3. Add metadata and documentation
4. Share with reviewers via private link

## 📊 Dataset

The processed results are available as a private dataset on Hugging Face:
[Manireddy1508/glasses-detection-dataset](https://huggingface.co/datasets/Manireddy1508/glasses-detection-dataset)

### Accessing the Dataset

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

For more details about the dataset structure and usage, please visit the [dataset card](https://huggingface.co/datasets/Manireddy1508/glasses-detection-dataset).

## Features

- Face detection using YOLOv8
- Eyeglasses classification using CLIP
- Batch processing of images
- Face verification using heuristics
- Detailed logging and statistics
- Interactive web interface for dataset exploration

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Manireddy1508/homeassignment-StabilityAI-.git
cd homeassignment-StabilityAI-
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
python scripts/download_models.py
```

## Usage

### Command Line Interface

The main script can be run with various options:

```bash
python scripts/detect_faces_and_glasses.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --confidence-threshold 0.5 \
    --batch-size 32 \
    --use-clip \
    --save-glasses
```

### Web Interface

To explore the dataset interactively, run the Streamlit app:

```bash
streamlit run scripts/app.py
```

The web interface provides:
- Dataset statistics and metrics
- Interactive filtering by:
  - Face detection confidence
  - Glasses detection confidence
  - Glasses type (Eyeglasses/Sunglasses/No Glasses)
- Visual display of detected faces with bounding boxes
- Confidence scores for face and glasses detection

## Project Structure

```
.
├── config/                 # Configuration files
├── data/                   # Data directories
│   ├── raw/               # Raw input images
│   ├── processed/         # Processed images
│   ├── glasses_images/    # Extracted glasses images
│   └── final/             # Final output
├── models/                # Model weights and configurations
├── scripts/               # Python scripts
│   ├── detect_faces_and_glasses.py  # Main detection script
│   ├── test_single_image.py        # Single image testing
│   ├── display_glasses_images.py   # Visualization script
│   ├── download_images.py          # Image downloader
│   ├── download_wit.py            # WIT dataset downloader
│   ├── download_models.py         # Model downloader
│   ├── upload_to_huggingface.py   # Dataset upload script
│   ├── create_dataset_card.py     # Dataset documentation
│   ├── test_dataset_access.py     # Dataset testing
│   └── app.py                     # Streamlit web interface
├── utils/                 # Utility modules
│   ├── face_filter.py    # Face verification
│   ├── clip_utils.py     # CLIP model utilities
│   └── glasses_utils.py  # Glasses detection utilities
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```
