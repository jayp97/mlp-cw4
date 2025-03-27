import os
import shutil
from glob import glob

# Define paths
TRAIN_DIR = "data/processed/images/train"
VAL_DIR = "data/processed/images/val"
TRAIN_VAL_DIR = "data/processed/images/train_val"  # New directory to store combined images

# Create the train_val directory if it doesn't exist
os.makedirs(TRAIN_VAL_DIR, exist_ok=True)

# Get the paths of all the images in the train and val directories
train_images = glob(os.path.join(TRAIN_DIR, "*.jpg"))
val_images = glob(os.path.join(VAL_DIR, "*.jpg"))

# Copy images from train directory to train_val
for img_path in train_images:
    shutil.copy(img_path, os.path.join(TRAIN_VAL_DIR, os.path.basename(img_path)))

# Copy images from val directory to train_val
for img_path in val_images:
    shutil.copy(img_path, os.path.join(TRAIN_VAL_DIR, os.path.basename(img_path)))

print(f"Copied {len(train_images) + len(val_images)} images to {TRAIN_VAL_DIR}")