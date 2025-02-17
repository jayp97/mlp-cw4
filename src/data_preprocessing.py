"""
data_preprocessing.py

Contains functions to preprocess the HAM10000 images and metadata:
1) Load metadata from CSV
2) Resize images to a fixed dimension (e.g., 224x224)
3) Save preprocessed images to data/processed/images
"""

import os
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm

# Adjust these paths as needed
RAW_IMAGES_PATH = "data/raw/images/"
METADATA_PATH = "data/raw/HAM10000_metadata.csv"
PROCESSED_PATH = "data/processed/images/"
IMG_SIZE = (224, 224)  # Example size


def load_metadata():
    """
    Loads HAM10000 metadata CSV, returning a pandas DataFrame.
    """
    df = pd.read_csv(METADATA_PATH)
    return df


def preprocess_images():
    """
    Resizes all raw images to IMG_SIZE and saves them in data/processed/images/.
    """
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    image_paths = glob(os.path.join(RAW_IMAGES_PATH, "*.jpg"))

    for img_path in tqdm(image_paths, desc="Resizing images"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        img_resized = cv2.resize(img, IMG_SIZE)
        save_path = os.path.join(PROCESSED_PATH, os.path.basename(img_path))
        cv2.imwrite(save_path, img_resized)


def main():
    """
    If this script is run directly, we execute the preprocessing steps.
    """
    df = load_metadata()
    print(f"Loaded metadata: {len(df)} records")
    preprocess_images()
    print("Image preprocessing complete.")


if __name__ == "__main__":
    main()
