# Face and Eyeglasses Detection Pipeline

A robust pipeline for detecting faces and eyeglasses in images using YOLOv8 and CLIP. The system processes images in batches, verifies faces using heuristics, and classifies eyeglasses using zero-shot learning.

## Features

- Face detection using YOLOv8
- Face verification using size, aspect ratio, and contrast checks
- Eyeglasses classification using CLIP with custom prompts
- Batch processing with checkpointing
- Detailed logging and progress tracking
- Annotated image saving with bounding boxes
- Results saved in Parquet format

## System Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM
- 10GB+ free disk space

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd imageclassifier
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

4. Download the YOLOv8 face detection model:
```bash
mkdir -p models
wget https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt -O models/yolov8n-face.pt
```

## Project Structure

```
.
├── data/
│   ├── images/              # Input images
│   ├── processed/           # Processed results
│   └── glasses_images/      # Images with detected glasses
│       └── annotated/       # Annotated images with bounding boxes
├── models/
│   └── yolov8n-face.pt     # YOLOv8 face detection model
├── scripts/
│   └── detect_faces_and_glasses.py  # Main processing script
├── utils/
│   ├── clip_utils.py       # CLIP-based glasses classifier
│   ├── face_filter.py      # Face verification utilities
│   └── glasses_utils.py    # Additional glasses detection utilities
└── requirements.txt        # Project dependencies
```

## Usage

### Basic Usage

Process images with default settings:
```bash
python scripts/detect_faces_and_glasses.py --use-clip --save-glasses
```

### Command Line Arguments

- `--confidence-threshold`: Confidence threshold for eyeglasses detection (default: 0.5)
- `--output-dir`: Directory to save results (default: data/processed)
- `--resume`: Resume from last checkpoint
- `--verbose`: Enable verbose logging
- `--use-clip`: Use CLIP-based classifier instead of ResNet
- `--save-glasses`: Save images with detected glasses
- `--batch-size`: Number of images to process in parallel (default: 32)
- `--max-workers`: Number of worker threads (default: 4)
- `--device`: Device to run models on (cuda/cpu)

### Example Commands

1. Process images with verbose logging:
```bash
python scripts/detect_faces_and_glasses.py --use-clip --save-glasses --verbose
```

2. Resume from checkpoint with custom batch size:
```bash
python scripts/detect_faces_and_glasses.py --use-clip --save-glasses --resume --batch-size 64
```

3. Run on CPU with custom confidence threshold:
```bash
python scripts/detect_faces_and_glasses.py --use-clip --save-glasses --device cpu --confidence-threshold 0.6
```

## Output Files

### Results

- `data/processed/glasses.parquet`: Contains detection results with columns:
  - `image_path`: Path to the image
  - `num_faces`: Number of faces detected
  - `faces`: List of detected faces with details:
    - Bounding box coordinates
    - Face size
    - Confidence scores
    - Base64-encoded thumbnail

### Checkpoints

- `data/processed/checkpoint.json`: Saves progress every 10 batches
  - List of processed images
  - Results for each processed image
  - Used for resuming interrupted processing

### Annotated Images

- `data/glasses_images/annotated/`: Contains images with:
  - Bounding boxes around faces
  - Confidence scores for eyeglasses detection
  - Filename format: `{original_name}_annotated.{extension}`

## Models

### Face Detection
- Model: YOLOv8n-face
- Purpose: Detect faces in images
- Minimum face size: 100x100 pixels
- Confidence threshold: 0.3

### Eyeglasses Classification
- Model: CLIP (openai/clip-vit-base-patch32)
- Purpose: Classify eyeglasses using zero-shot learning
- Classes:
  - Transparent eyeglasses
  - Sunglasses
  - No glasses
- Confidence threshold: 0.5

## Logging

- Log file: `glasses_detection.log`
- Log levels:
  - INFO: General processing information
  - DEBUG: Detailed face detection and verification
  - WARNING: Image reading failures
  - ERROR: Processing errors

## Performance

The pipeline processes images in batches for efficiency:
- Default batch size: 32 images
- Parallel processing with ThreadPoolExecutor
- Checkpointing every 10 batches
- Progress bar shows completion percentage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here] # homeassignment-StabilityAI-
