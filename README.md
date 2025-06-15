# homeassignment-StabilityAI-

A comprehensive pipeline for detecting faces and eyeglasses in images using YOLOv8 and CLIP models.

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

### Model Download
Download required models (YOLOv8n-face and CLIP):
```bash
python scripts/download_models.py
```
Or specify a custom directory:
```bash
python scripts/download_models.py --models-dir test_downloads
```

### Face and Glasses Detection
Process images to detect faces and eyeglasses:
```bash
python scripts/detect_faces_and_glasses.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --confidence-threshold 0.5 \
    --batch-size 32 \
    --use-clip \
    --save-glasses
```

### Dataset Management
Upload processed dataset to Hugging Face:
```bash
python scripts/upload_to_huggingface.py
```

Test dataset access:
```bash
python scripts/test_dataset_access.py
```

### Web Interface
Explore the dataset interactively:
```bash
streamlit run scripts/app.py
```

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
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Models

### YOLOv8n-face
- Used for face detection
- Lightweight and fast
- Included in the repository

### CLIP (ViT-B/32)
- Used for glasses classification
- Downloaded automatically
- Stored in safetensors format

## Logging

All operations are logged to:
- Console output
- `glasses_detection.log` file

## Performance

- Face detection: ~30ms per image (on GPU)
- Glasses classification: ~50ms per face (on GPU)
- Batch processing supported for better throughput

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to open an issue or submit a pull request.

## License

This project is open-source
