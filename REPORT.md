# Glasses Detection Pipeline — Brief Report

## 🔐 Repository Access

### GitHub Repository
- URL: https://github.com/Manireddy1508/homeassignment-StabilityAI-
- Access: Private repository
- Granted read access to:
  - leemengtw
  - polm-stability
  - Cwgzstab
  - fernando-andreotti-sai
  - vlad-stability

### Hugging Face Dataset
- Repository: Manireddy1508/glasses-detection-dataset
- Access: Private repository
- Contains processed dataset with:
  - Filtered images of faces wearing eyeglasses
  - Detection metadata and confidence scores
  - Face bounding boxes and thumbnails

## 🧩 Process Summary

1. **Dataset**: We used the WIT dataset (wikimedia/wit) focusing on `train-00000-of-00330.parquet` and `train-00001-of-00330.parquet`. Image URLs were extracted and downloaded with retry logic and validation.

2. **Face Detection**: We used YOLOv8n-face for accurate and lightweight face detection. Only faces ≥100×100 px were processed. Further filters excluded faces that were cropped, too close to edges, or distorted.

3. **Glasses Classification**: CLIP (ViT-B/32) classified faces as:
   - "a human face with transparent vision correction eyeglasses"
   - "a human face with sunglasses"
   - "a human face with no glasses"

   Only images with at least one face confidently classified as wearing **eyeglasses** (not sunglasses) were retained.

4. **Output Format**: Metadata for each valid image and face is saved in a Parquet file, along with base64-encoded thumbnails and annotated visuals. Only images with eyeglasses are saved.

---

## 📈 Improvements

- **Model Enhancements**: Fine-tune CLIP on real-world eyewear datasets. Use RetinaFace or larger YOLO variants for edge-case accuracy.
- **Filtering**: Improve rejection of false positives using a BLIP-2 image captioning module or rule-based scene validation (e.g., presence of pancakes misidentified as faces).
- **Data Labeling**: Add support for age, pose, and lighting-based filters for higher control over image curation.

---

## 🚀 Scaling Strategy

To handle **billions of images/videos**, we recommend:

1. **Distributed Compute**:
   - Use Apache Spark or Ray for large-scale parallel processing
   - Containerize pipeline using Docker, orchestrated via Kubernetes or Vertex AI Pipelines
   - For videos:
     - Implement frame sampling strategies (e.g., 1 frame per second)
     - Use video-specific processing clusters with GPU acceleration
     - Deploy video decoding workers with hardware acceleration

2. **Efficient Storage**:
   - Store raw images in S3 or GCS
   - Store metadata in compressed columnar formats (Parquet)
   - Use vector DBs (e.g., Milvus) for CLIP embeddings
   - For videos:
     - Store video chunks in distributed object storage
     - Use video streaming formats (e.g., HLS, DASH)
     - Implement tiered storage for hot/cold video data

3. **Optimized Inference**:
   - Convert YOLO and CLIP to ONNX for FP16 inference
   - Use GPU batch inference
   - Cache repeated scene embeddings for multi-face frames
   - For videos:
     - Implement temporal consistency checks
     - Use video-specific face tracking
     - Batch process frames with temporal context

4. **Video-Specific Optimizations**:
   - Frame Selection:
     - Skip similar consecutive frames
     - Focus on frames with significant motion
     - Use scene change detection
   - Processing Pipeline:
     - Parallel video decoding and processing
     - GPU-accelerated video transcoding
     - Distributed video metadata extraction
   - Storage and Retrieval:
     - Video chunking for efficient streaming
     - Frame-level indexing for quick access
     - Temporal metadata for scene navigation

---

This setup allows high-precision filtering of human faces with transparent eyeglasses and can scale from local batches to billions of samples with minimal architectural changes. The system is designed to handle both static images and video content efficiently, with optimizations specific to each media type. 