import os
import pandas as pd
import cv2
from glob import glob

DATA_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"


def load_metadata():
    """Load metadata CSV file."""
    csv_path = os.path.join(DATA_PATH, "HAM10000_metadata.csv")
    return pd.read_csv(csv_path)


def preprocess_images():
    """Resize images and store them in processed folder."""
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    image_paths = glob(os.path.join(DATA_PATH, "*.jpg"))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (224, 224))  # Resize for EfficientNet
        save_path = os.path.join(PROCESSED_PATH, os.path.basename(img_path))
        cv2.imwrite(save_path, img_resized)


if __name__ == "__main__":
    df = load_metadata()
    preprocess_images()
    print(f"Processed {len(df)} images and saved to {PROCESSED_PATH}")
