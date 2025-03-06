#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import random
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_train_test_split(
    metadata_path="data/raw/HAM10000_metadata.csv",
    processed_images_path="data/processed_sd/images",
    output_dir="data/split",
    train_ratio=0.8,
    random_seed=42,
    create_symlinks=False,
):
    """
    Create a permanent train/test split of the HAM10000 dataset.
    Splits are stratified by lesion type.

    Args:
        metadata_path: Path to HAM10000 metadata CSV
        processed_images_path: Path to processed images
        output_dir: Root directory to save split information
        train_ratio: Ratio of images to use for training
        random_seed: Random seed for reproducibility
        create_symlinks: Whether to create symbolic links to images
    """
    logger.info("Creating train/test split...")

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Read metadata
    metadata = pd.read_csv(metadata_path)

    # Get list of available image files
    available_images = set(os.listdir(processed_images_path))

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # Initialize dataframes for train and test splits
    train_df = pd.DataFrame(columns=metadata.columns)
    test_df = pd.DataFrame(columns=metadata.columns)

    # Get unique lesion types
    lesion_types = metadata["dx"].unique()

    # Track counts
    counts = {}

    # Process each lesion type
    for lesion_type in lesion_types:
        logger.info(f"Processing {lesion_type}...")

        # Filter metadata for this lesion type
        lesion_metadata = metadata[metadata["dx"] == lesion_type].copy()

        # Shuffle the metadata
        lesion_metadata = lesion_metadata.sample(frac=1, random_state=random_seed)

        # Get train/test counts
        total_count = len(lesion_metadata)
        train_count = int(total_count * train_ratio)
        test_count = total_count - train_count

        # Store counts
        counts[lesion_type] = {
            "total": total_count,
            "train": train_count,
            "test": test_count,
        }

        # Create train/test splits
        train_metadata = lesion_metadata.iloc[:train_count]
        test_metadata = lesion_metadata.iloc[train_count:]

        # Add to train/test dataframes
        train_df = pd.concat([train_df, train_metadata])
        test_df = pd.concat([test_df, test_metadata])

        # Create symbolic links or actual copies if requested
        if create_symlinks:
            train_dir = os.path.join(output_dir, "train", lesion_type)
            test_dir = os.path.join(output_dir, "test", lesion_type)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            logger.info(f"Creating symbolic links for {lesion_type}...")
            for _, row in tqdm(train_metadata.iterrows(), total=len(train_metadata)):
                image_id = row["image_id"]
                for ext in [".jpg", ".png"]:
                    if f"{image_id}{ext}" in available_images:
                        src_path = os.path.join(
                            processed_images_path, f"{image_id}{ext}"
                        )
                        dst_path = os.path.join(train_dir, f"{image_id}{ext}")
                        try:
                            os.symlink(os.path.abspath(src_path), dst_path)
                        except FileExistsError:
                            # Skip if already exists
                            pass
                        break

            for _, row in tqdm(test_metadata.iterrows(), total=len(test_metadata)):
                image_id = row["image_id"]
                for ext in [".jpg", ".png"]:
                    if f"{image_id}{ext}" in available_images:
                        src_path = os.path.join(
                            processed_images_path, f"{image_id}{ext}"
                        )
                        dst_path = os.path.join(test_dir, f"{image_id}{ext}")
                        try:
                            os.symlink(os.path.abspath(src_path), dst_path)
                        except FileExistsError:
                            # Skip if already exists
                            pass
                        break

    # Save train/test splits to CSV
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # Save counts
    counts_df = pd.DataFrame.from_dict(counts, orient="index")
    counts_df.to_csv(os.path.join(output_dir, "counts.csv"))

    logger.info("Split creation completed.")
    logger.info(f"Train data: {len(train_df)} images")
    logger.info(f"Test data: {len(test_df)} images")
    logger.info(f"Split information saved to {output_dir}")

    # Print summary of counts by class
    logger.info("Class distribution:")
    for lesion_type, count_data in counts.items():
        logger.info(
            f"  {lesion_type}: {count_data['train']} train, {count_data['test']} test"
        )

    return train_df, test_df, counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/test split for HAM10000")
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/raw/HAM10000_metadata.csv",
        help="Path to HAM10000 metadata CSV",
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
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of images to use for training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--create_links", action="store_true", help="Create symbolic links to images"
    )

    args = parser.parse_args()

    create_train_test_split(
        metadata_path=args.metadata,
        processed_images_path=args.images,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        create_symlinks=args.create_links,
    )
