#!/usr/bin/env python
"""
Create a train/test split matching the group's classifier code.
This script creates the exact same train/test split as in data_preprocessing.py,
ensuring compatibility between our work and the group's classifier.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_matching_split(
    metadata_path="data/raw/HAM10000_metadata.csv",
    processed_images_path="data/processed_sd/images",
    output_dir="data/split",
    test_size=0.2,
    random_seed=42,
    create_symlinks=False,
):
    """
    Create a train/test split matching the group's classifier code.
    Uses the same stratification approach and random seed for consistency.

    Args:
        metadata_path: Path to HAM10000 metadata CSV
        processed_images_path: Path to processed images
        output_dir: Root directory to save split information
        test_size: Proportion of data for test set
        random_seed: Random seed for reproducibility
        create_symlinks: Whether to create symbolic links to images
    """
    logger.info("Creating train/test split matching the classifier...")

    # Read metadata
    metadata = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata with {len(metadata)} entries")

    # Ensure we have a filename column
    if "filename" not in metadata.columns:
        metadata["filename"] = metadata["image_id"] + ".jpg"

    # Get available files in the processed_images_path
    available_files = {
        f for f in os.listdir(processed_images_path) if f.endswith((".jpg", ".png"))
    }

    # Filter metadata to include only available files
    image_ids = []
    for _, row in metadata.iterrows():
        for ext in [".jpg", ".png"]:
            filename = f"{row['image_id']}{ext}"
            if filename in available_files:
                image_ids.append(row["image_id"])
                break

    filtered_metadata = metadata[metadata["image_id"].isin(image_ids)]
    logger.info(f"Found {len(filtered_metadata)} images in processed directory")

    # Get unique lesion types and their counts
    lesion_types = filtered_metadata["dx"].unique()
    lesion_counts = filtered_metadata["dx"].value_counts()
    logger.info(f"Found {len(lesion_types)} lesion types: {', '.join(lesion_types)}")
    for lesion_type in lesion_types:
        logger.info(f"  {lesion_type}: {lesion_counts[lesion_type]} images")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Perform stratified splitting exactly like in data_preprocessing.py
    # Split into train/test using stratification
    train_df, test_df = train_test_split(
        filtered_metadata,
        test_size=test_size,
        stratify=filtered_metadata["dx"],
        random_state=random_seed,
    )

    # Save split information to CSV files
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # Print split statistics
    logger.info(f"Split statistics:")
    logger.info(f"  Train set: {len(train_df)} images")
    logger.info(f"  Test set: {len(test_df)} images")

    # Per-class split statistics
    logger.info("Class distribution:")
    for lesion_type in lesion_types:
        train_count = sum(train_df["dx"] == lesion_type)
        test_count = sum(test_df["dx"] == lesion_type)
        total_count = train_count + test_count

        logger.info(
            f"  {lesion_type}: {train_count} train, {test_count} test (total: {total_count})"
        )

    # Create symbolic links if requested
    if create_symlinks:
        logger.info("Creating symbolic links...")

        # Create directories for each split
        train_dir = os.path.join(output_dir, "train")
        test_dir = os.path.join(output_dir, "test")

        for directory in [train_dir, test_dir]:
            os.makedirs(directory, exist_ok=True)

            # Create subdirectories for each class
            for lesion_type in lesion_types:
                os.makedirs(os.path.join(directory, lesion_type), exist_ok=True)

        # Create symlinks for train images
        for _, row in tqdm(
            train_df.iterrows(), total=len(train_df), desc="Creating train symlinks"
        ):
            for ext in [".jpg", ".png"]:
                filename = f"{row['image_id']}{ext}"
                src_path = os.path.join(processed_images_path, filename)
                if os.path.exists(src_path):
                    dst_path = os.path.join(train_dir, row["dx"], filename)
                    try:
                        if not os.path.exists(dst_path):
                            os.symlink(os.path.abspath(src_path), dst_path)
                    except OSError:
                        logger.warning(f"Could not create symlink for {filename}")
                    break

        # Create symlinks for test images
        for _, row in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Creating test symlinks"
        ):
            for ext in [".jpg", ".png"]:
                filename = f"{row['image_id']}{ext}"
                src_path = os.path.join(processed_images_path, filename)
                if os.path.exists(src_path):
                    dst_path = os.path.join(test_dir, row["dx"], filename)
                    try:
                        if not os.path.exists(dst_path):
                            os.symlink(os.path.abspath(src_path), dst_path)
                    except OSError:
                        logger.warning(f"Could not create symlink for {filename}")
                    break

    logger.info("Split creation completed successfully.")
    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train/test split matching the group's classifier"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/raw/HAM10000_metadata.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="data/processed_sd/images",
        help="Path to processed images",
    )
    parser.add_argument(
        "--output", type=str, default="data/split", help="Output directory"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Proportion of data for test set"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--create_links", action="store_true", help="Create symbolic links to images"
    )

    args = parser.parse_args()

    create_matching_split(
        metadata_path=args.metadata,
        processed_images_path=args.images,
        output_dir=args.output,
        test_size=args.test_size,
        random_seed=args.seed,
        create_symlinks=args.create_links,
    )
