"""
data_preprocessing.py

Contains functions to preprocess the HAM10000 images and metadata:
1) Load metadata from CSV
2) Resize images to a fixed dimension (e.g., 224x224) for classification
3) (NEW) Resize images to 512x512 for Stable Diffusion
4) Save the processed images to the appropriate directories.
"""

import os
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm

# Paths for the classification images (224x224)
RAW_IMAGES_PATH = "data/raw/images/"
METADATA_PATH = "data/raw/HAM10000_metadata.csv"
PROCESSED_PATH = "data/processed/images/"
IMG_SIZE = (224, 224)  # For classification

# NEW: Paths for the stable diffusion images (512x512)
SD_PROCESSED_PATH = "data/processed_sd/images/"
SD_IMG_SIZE = (512, 512)  # For stable diffusion fine-tuning


def load_metadata():
    """
    Loads HAM10000 metadata CSV, returning a pandas DataFrame.
    """
    df = pd.read_csv(METADATA_PATH)
    return df


def preprocess_images():
    """
    Resizes all raw images to 224x224 and saves them in data/processed/images/.
    This is used for classification tasks (EfficientNet).
    """
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    image_paths = glob(os.path.join(RAW_IMAGES_PATH, "*.jpg"))

    for img_path in tqdm(image_paths, desc="Resizing images to 224x224"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Resize to 224x224
        img_resized = cv2.resize(img, IMG_SIZE)
        save_path = os.path.join(PROCESSED_PATH, os.path.basename(img_path))
        cv2.imwrite(save_path, img_resized)


def preprocess_images_for_sd():
    """
    Resizes all raw images to 512x512 and saves them in data/processed_sd/images/.
    This is used for stable diffusion fine-tuning.
    """
    os.makedirs(SD_PROCESSED_PATH, exist_ok=True)
    image_paths = glob(os.path.join(RAW_IMAGES_PATH, "*.jpg"))

    for img_path in tqdm(image_paths, desc="Resizing images to 512x512 for SD"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Resize to 512x512
        img_resized = cv2.resize(img, SD_IMG_SIZE)
        save_path = os.path.join(SD_PROCESSED_PATH, os.path.basename(img_path))
        cv2.imwrite(save_path, img_resized)


def main():
    """
    If this script is run directly, we execute both preprocessing steps.
    1) For classification (224x224)
    2) For stable diffusion (512x512)
    """
    df = load_metadata()
    print(f"Loaded metadata: {len(df)} records")

    # 1) Resize for classification
    preprocess_images()
    print("Image preprocessing (224x224) complete.")

    # 2) Resize for stable diffusion
    preprocess_images_for_sd()
    print("Image preprocessing (512x512) for SD complete.")


if __name__ == "__main__":
    main()
