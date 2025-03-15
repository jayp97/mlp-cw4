"""
data_preprocessing.py

Contains functions to preprocess the HAM10000 images and metadata:
1) Load metadata from CSV
2) Stratified splitting by class for training validation and test splits
3) Resize images to a fixed dimension (e.g., 224x224) for classification
4) Resize images to 512x512 for Stable Diffusion
5) Save the processed images to the appropriate directories.
"""

import os
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Paths for the classification images (512x512)
RAW_IMAGES_PATH = "../data/raw/images/"
METADATA_PATH = "../data/raw/HAM10000_metadata.csv"
PROCESSED_PATH = "../data/processed/images/"
SYNTHETIC_PATH = "../data/synthetic_by_class/synthetic"
PROCESSED_SYNTHETIC = "../data/processed_synth"

IMG_SIZE = (256,256) # classification
TRAIN_TEST_SPLIT = 0.2

# Paths for the stable diffusion images (512x512)
SD_PROCESSED_PATH = "../data/processed_sd/images/"
SD_IMG_SIZE = (512, 512)  # For stable diffusion fine-tuning


def load_metadata():
    """
    Loads HAM10000 metadata CSV, returning a pandas DataFrame.
    """
    df = pd.read_csv(METADATA_PATH)
    return df


def preprocess_images():
    """
    Resizes all raw images and saves them in data/processed/images/.
    This is used for classification tasks (EfficientNet).
    """
    # Create subfolders for train, val, test
    train_dir = os.path.join(PROCESSED_PATH, "train")
    val_dir   = os.path.join(PROCESSED_PATH, "val")
    test_dir  = os.path.join(PROCESSED_PATH, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    df = pd.read_csv(METADATA_PATH)
    df["filename"] = df["image_id"] + ".jpg"

    # Stratified splitting:
    # First, hold out 20% for test. Then split the remaining 80% again into 80% train, 20% val
    train_val_df, test_df = train_test_split(
        df,
        test_size=TRAIN_TEST_SPLIT, 
        stratify=df["dx"],       # stratify on dx
        random_state=42
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=TRAIN_TEST_SPLIT,
        stratify=train_val_df["dx"],
        random_state=42
    )

    # Convert these into sets of filenames 
    train_fnames = set(train_df["filename"])
    val_fnames   = set(val_df["filename"])
    test_fnames  = set(test_df["filename"])

    print(f"Total images: {len(df)}")
    print(f"Train: {len(train_fnames)} | Val: {len(val_fnames)} | Test: {len(test_fnames)}")

    # Loop over all raw images, resize, and place in the correct subfolder
    image_paths = glob(os.path.join(RAW_IMAGES_PATH, "*.jpg"))
    for img_path in tqdm(image_paths, desc=f"Resizing images to {IMG_SIZE}"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Resize
        img_resized = cv2.resize(img, IMG_SIZE)
        base_name   = os.path.basename(img_path)

        # Decide which subfolder by membership
        if base_name in train_fnames:
            save_folder = train_dir
        elif base_name in val_fnames:
            save_folder = val_dir
        elif base_name in test_fnames:
            save_folder = test_dir

        # Save in the chosen subfolder
        save_path = os.path.join(save_folder, base_name)
        cv2.imwrite(save_path, img_resized)

    print("Done resizing and splitting images stratified by dx.")

    image_paths = glob(os.path.join(SYNTHETIC_PATH, "*.png"))

    for img_path in tqdm(image_paths, desc=f"Resizing images to {IMG_SIZE}"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Convert the image to JPG by saving it as a temporary .jpg file
        temp_jpg_path = img_path.replace(".png", ".jpg")
        cv2.imwrite(temp_jpg_path, img)

        # Read the image back as a jpg
        img = cv2.imread(temp_jpg_path)

        # Resize the image
        img_resized = cv2.resize(img, IMG_SIZE)

        # Get the base name (without extension) and append .jpg
        base_name = os.path.basename(img_path).replace(".png", ".jpg")

        # Save the resized image to the processed folder as .jpg
        save_path = os.path.join(PROCESSED_SYNTHETIC, base_name)
        cv2.imwrite(save_path, img_resized)

        # Optionally, remove the temporary jpg file after saving
        os.remove(temp_jpg_path)

    print("Done resizing and splitting images stratified by dx.")


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
