import streamlit as st
import pandas as pd
from datasets import load_dataset
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os

def load_data():
    """Load the dataset from Hugging Face."""
    dataset = load_dataset("Manireddy1508/glasses-detection-dataset")
    return dataset['train']

def decode_base64_to_image(base64_string):
    """Convert base64 string to PIL Image."""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        st.error(f"Error decoding image: {e}")
        return None

def display_face_with_bbox(image, bbox, confidence, glasses_conf):
    """Display face with bounding box and confidence scores."""
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Draw bounding box
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add confidence scores
    cv2.putText(img_array, f"Face: {confidence:.2f}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img_array, f"Glasses: {glasses_conf:.2f}", (x1, y2+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convert back to RGB for display
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_array)

def main():
    st.title("Glasses Detection Dataset Explorer")
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        dataset = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Confidence threshold
    face_conf_threshold = st.sidebar.slider(
        "Face Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    glasses_conf_threshold = st.sidebar.slider(
        "Glasses Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    # Filter by glasses type
    glasses_type = st.sidebar.selectbox(
        "Glasses Type",
        ["All", "Eyeglasses", "Sunglasses", "No Glasses"]
    )
    
    # Display dataset statistics
    st.header("Dataset Statistics")
    total_images = len(dataset)
    total_faces = sum(item['num_faces'] for item in dataset)
    total_glasses = sum(1 for item in dataset if item['has_glasses'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", total_images)
    col2.metric("Total Faces", total_faces)
    col3.metric("Images with Glasses", total_glasses)
    
    # Display filtered results
    st.header("Filtered Results")
    
    # Create a grid of images
    cols = st.columns(3)
    col_idx = 0
    
    for item in dataset:
        # Apply filters
        if item['num_faces'] == 0:
            continue
            
        for face in item['faces']:
            if (face['face_confidence'] < face_conf_threshold or
                face['eyeglasses_conf'] < glasses_conf_threshold):
                continue
                
            if glasses_type != "All":
                if glasses_type == "Eyeglasses" and not face['is_eyeglasses']:
                    continue
                if glasses_type == "Sunglasses" and face['glasses_type'] != "sunglasses":
                    continue
                if glasses_type == "No Glasses" and face['is_eyeglasses']:
                    continue
            
            # Display face
            with cols[col_idx]:
                face_img = decode_base64_to_image(face['thumbnail'])
                if face_img:
                    annotated_img = display_face_with_bbox(
                        face_img,
                        face['bbox'],
                        face['face_confidence'],
                        face['eyeglasses_conf']
                    )
                    st.image(annotated_img, caption=f"Confidence: {face['face_confidence']:.2f}")
                    st.write(f"Glasses: {face['glasses_type']}")
                    st.write(f"Glasses Conf: {face['eyeglasses_conf']:.2f}")
            
            col_idx = (col_idx + 1) % 3
            if col_idx == 0:
                cols = st.columns(3)

if __name__ == "__main__":
    main() 