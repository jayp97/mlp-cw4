"""
data_preprocessing.py

This script contains functions to preprocess and resize the original HAM10000
images to a standardized size. It also loads and filters metadata as needed.
"""

import os
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm

# Define paths (adjust as appropriate)
RAW_IMAGES_PATH = "data/raw/images/"
METADATA_PATH = "data/raw/HAM10000_metadata.csv"
PROCESSED_PATH = "data/processed/images/"
IMG_SIZE = (224, 224)  # For example, 224x224


def load_metadata():
    """
    Load the HAM10000 metadata CSV file which contains the diagnosis labels.
    Returns a Pandas DataFrame.
    """
    df = pd.read_csv(METADATA_PATH)
    return df


def preprocess_images():
    """
    Resize all raw images to IMG_SIZE and save them to the processed directory.
    """
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    image_paths = glob(os.path.join(RAW_IMAGES_PATH, "*.jpg"))

    for img_path in tqdm(image_paths, desc="Resizing images"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Resize
        img_resized = cv2.resize(img, IMG_SIZE)

        # Save
        filename = os.path.basename(img_path)
        save_path = os.path.join(PROCESSED_PATH, filename)
        cv2.imwrite(save_path, img_resized)


def main():
    """
    Main entry point for preprocessing if this file is run directly.
    """
    df = load_metadata()
    print(f"Metadata loaded. Number of entries: {len(df)}")
    preprocess_images()
    print("Image preprocessing completed.")


if __name__ == "__main__":
    main()
