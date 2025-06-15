from datasets import load_dataset
import json
from pprint import pprint

def test_dataset_access():
    """Test accessing the dataset and display sample data."""
    print("Loading dataset...")
    dataset = load_dataset("Manireddy1508/glasses-detection-dataset")
    
    print("\nDataset structure:")
    print(dataset)
    
    print("\nSample data:")
    sample = dataset['train'][0]
    
    # Print basic information
    print(f"\nImage path: {sample['image_path']}")
    print(f"Number of faces: {sample['num_faces']}")
    print(f"Has glasses: {sample['has_glasses']}")
    
    # Print face details
    for i, face in enumerate(sample['faces']):
        print(f"\nFace {i+1}:")
        print(f"  Bounding box: {face['bbox']}")
        # Print all available keys in the face dictionary
        print("  Available data:")
        for key, value in face.items():
            if key != 'bbox':  # We already printed bbox
                print(f"    {key}: {value}")

if __name__ == "__main__":
    test_dataset_access() 