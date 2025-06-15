import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import base64
from io import BytesIO
import argparse

def load_parquet_data(parquet_path):
    """Load data from Parquet file."""
    return pd.read_parquet(parquet_path)

def display_face_thumbnail(thumbnail_base64):
    """Display a face thumbnail from base64 string."""
    try:
        # Decode base64 to image
        image_data = base64.b64decode(thumbnail_base64)
        image = Image.open(BytesIO(image_data))
        
        # Display image
        plt.imshow(np.array(image))
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error displaying thumbnail: {e}")

def display_original_image(image_path, bbox):
    """Display original image with face bounding box."""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read image: {image_path}")
            return
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display image
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {e}")

def main():
    parser = argparse.ArgumentParser(description='Display images with glasses')
    parser.add_argument('--parquet', type=str, default='data/processed/glasses.parquet',
                      help='Path to Parquet file')
    parser.add_argument('--min-confidence', type=float, default=0.7,
                      help='Minimum confidence for glasses detection')
    parser.add_argument('--show-thumbnails', action='store_true',
                      help='Show face thumbnails instead of full images')
    args = parser.parse_args()
    
    # Load data
    df = load_parquet_data(args.parquet)
    
    # Filter for faces with glasses
    glasses_faces = []
    for _, row in df.iterrows():
        for face in row['faces']:
            if face['is_eyeglasses'] and face['eyeglasses_conf'] > args.min_confidence:
                glasses_faces.append({
                    'image_path': row['image_path'],
                    'face': face
                })
    
    print(f"Found {len(glasses_faces)} faces with glasses")
    
    # Display faces
    for i, face_data in enumerate(glasses_faces):
        print(f"\nFace {i+1}/{len(glasses_faces)}")
        print(f"Image: {face_data['image_path']}")
        print(f"Confidence: {face_data['face']['eyeglasses_conf']:.2f}")
        
        if args.show_thumbnails:
            display_face_thumbnail(face_data['face']['thumbnail'])
        else:
            display_original_image(face_data['image_path'], face_data['face']['bbox'])
        
        # Wait for user input to continue
        input("Press Enter to continue...")

if __name__ == "__main__":
    main()