import os
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_data_for_fine_tuning(
    metadata_path="data/raw/HAM10000_metadata.csv",
    processed_images_path="data/processed_sd/images",
    output_dir="data/fine_tuning",
    val_split=0.1,
    prompt_template="{lesion_type} skin lesion, dermatological photo, clinical image",  # Enhanced prompt
    train_split_path=None,  # New parameter
    test_split_path=None,  # New parameter
):
    """
    Prepare HAM10000 dataset for Stable Diffusion fine-tuning.

    Args:
        metadata_path: Path to HAM10000 metadata CSV
        processed_images_path: Path to processed 512x512 images
        output_dir: Output directory for fine-tuning data
        val_split: Fraction of data to use for validation
        prompt_template: Template for generating prompts
        train_split_path: Path to train split CSV (optional)
        test_split_path: Path to test split CSV (optional)
    """
    logger.info("Preparing data for fine-tuning...")

    # Read metadata
    metadata = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata with {len(metadata)} entries")

    # If train/test splits are provided, use them
    if train_split_path and os.path.exists(train_split_path):
        logger.info(f"Using provided train split: {train_split_path}")
        train_df = pd.read_csv(train_split_path)

        # We'll use only training data for fine-tuning
        metadata = train_df

        # No need for val_split as we'll use test data for validation
        val_split = 0.0

        logger.info(f"Using {len(metadata)} images from train split")

    # Get unique lesion types
    lesion_types = metadata["dx"].unique()
    logger.info(f"Found {len(lesion_types)} lesion types: {', '.join(lesion_types)}")

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Create train and validation text files
    train_file = open(os.path.join(output_dir, "train.txt"), "w")
    val_file = open(os.path.join(output_dir, "val.txt"), "w")

    # If we have a test split, use it for validation
    test_image_ids = set()
    if test_split_path and os.path.exists(test_split_path) and val_split == 0.0:
        logger.info(f"Using provided test split for validation: {test_split_path}")
        test_df = pd.read_csv(test_split_path)
        test_image_ids = set(test_df["image_id"].tolist())
        logger.info(f"Found {len(test_image_ids)} images in test split")

    # Process each image
    processed_files = set(os.listdir(processed_images_path))
    count_by_type = {lesion_type: 0 for lesion_type in lesion_types}

    for _, row in tqdm(
        metadata.iterrows(), total=len(metadata), desc="Processing images"
    ):
        image_id = row["image_id"]
        lesion_type = row["dx"]

        # Find corresponding image file
        image_file = None
        for ext in [".jpg", ".png"]:
            if f"{image_id}{ext}" in processed_files:
                image_file = f"{image_id}{ext}"
                break

        if image_file is None:
            continue

        # Create prompt with more specific details
        prompt = prompt_template.format(lesion_type=lesion_type)

        # Decide train or validation
        is_val = False

        # If using test split for validation
        if test_image_ids:
            is_val = image_id in test_image_ids
        else:
            # Use random split
            is_val = random.random() < val_split

        # Copy image to appropriate directory
        src_path = os.path.join(processed_images_path, image_file)

        if is_val:
            dst_path = os.path.join(val_dir, image_file)
            val_file.write(f"{image_file},{prompt}\n")
        else:
            dst_path = os.path.join(train_dir, image_file)
            train_file.write(f"{image_file},{prompt}\n")

        shutil.copy2(src_path, dst_path)
        count_by_type[lesion_type] += 1

    train_file.close()
    val_file.close()

    # Print statistics
    logger.info("\nData preparation completed:")
    logger.info(f"Total images processed: {sum(count_by_type.values())}")
    logger.info("Images per lesion type:")
    for lesion_type, count in count_by_type.items():
        logger.info(f"  - {lesion_type}: {count}")

    return lesion_types


def create_class_specific_data(
    specific_class,
    metadata_path="data/raw/HAM10000_metadata.csv",
    processed_images_path="data/processed_sd/images",
    output_dir="data/fine_tuning_class_specific",
    prompt_template="{lesion_type} skin lesion, dermatological photo, clinical image, centered, high detail, realistic texture",  # Enhanced prompt
    train_split_path=None,  # New parameter
):
    """
    Create a dataset focused on a specific lesion class for targeted fine-tuning.

    Args:
        specific_class: The specific lesion class to focus on
        metadata_path: Path to HAM10000 metadata CSV
        processed_images_path: Path to processed 512x512 images
        output_dir: Output directory for fine-tuning data
        prompt_template: Template for generating prompts
        train_split_path: Path to train split CSV (optional)
    """
    logger.info(f"Preparing class-specific data for: {specific_class}")

    # Read metadata
    metadata = pd.read_csv(metadata_path)

    # Filter for specific class
    class_metadata = metadata[metadata["dx"] == specific_class]

    # If train split is provided, use only training images
    if train_split_path and os.path.exists(train_split_path):
        logger.info(f"Using provided train split: {train_split_path}")
        train_df = pd.read_csv(train_split_path)
        train_df = train_df[train_df["dx"] == specific_class]

        if len(train_df) > 0:
            logger.info(
                f"Using {len(train_df)} images from train split for class '{specific_class}'"
            )
            class_metadata = train_df
        else:
            logger.warning(
                f"Warning: No images of class '{specific_class}' found in train split"
            )
            # Continue with all images of this class if none found in train split

    logger.info(f"Found {len(class_metadata)} images for class '{specific_class}'")

    if len(class_metadata) == 0:
        logger.error(f"Error: No images found for class '{specific_class}'")
        logger.error(f"Available classes: {', '.join(metadata['dx'].unique())}")
        return None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    class_dir = os.path.join(output_dir, specific_class)
    os.makedirs(class_dir, exist_ok=True)

    # Create text file for prompts
    prompt_file = os.path.join(output_dir, f"{specific_class}_prompts.txt")
    with open(prompt_file, "w") as f:
        # Process each image
        processed_files = set(os.listdir(processed_images_path))
        count = 0

        for _, row in tqdm(
            class_metadata.iterrows(),
            total=len(class_metadata),
            desc=f"Processing {specific_class} images",
        ):
            image_id = row["image_id"]

            # Find corresponding image file
            image_file = None
            for ext in [".jpg", ".png"]:
                if f"{image_id}{ext}" in processed_files:
                    image_file = f"{image_id}{ext}"
                    break

            if image_file is None:
                continue

            # Create prompt
            prompt = prompt_template.format(lesion_type=specific_class)

            # Copy image
            src_path = os.path.join(processed_images_path, image_file)
            dst_path = os.path.join(class_dir, image_file)
            shutil.copy2(src_path, dst_path)

            # Write prompt to file
            f.write(f"{image_file},{prompt}\n")
            count += 1

    logger.info(f"\nPrepared {count} images for class '{specific_class}'")
    return prompt_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare HAM10000 dataset for fine-tuning"
    )
    parser.add_argument(
        "--specific_class",
        type=str,
        default=None,
        help="Specific lesion class to prepare",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default=None,
        help="Path to train split CSV",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default=None,
        help="Path to test split CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for data",
    )
    args = parser.parse_args()

    # Set default output directory
    output_dir = args.output_dir or "data/fine_tuning_class_specific"

    if args.specific_class:
        create_class_specific_data(
            args.specific_class,
            train_split_path=args.train_split,
            output_dir=output_dir,
        )
    else:
        # Get available lesion types
        output_dir_general = args.output_dir or "data/fine_tuning"
        lesion_types = prepare_data_for_fine_tuning(
            train_split_path=args.train_split,
            test_split_path=args.test_split,
            output_dir=output_dir_general,
        )

        # Example: Create dataset for a specific class
        if lesion_types is not None and len(lesion_types) > 0:
            # You can change this to any class you want to focus on
            specific_class = lesion_types[0]  # e.g., 'dermatofibroma'
            create_class_specific_data(
                specific_class, train_split_path=args.train_split, output_dir=output_dir
            )
