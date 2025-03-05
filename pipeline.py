import os
import argparse
import logging
from data_preparation import create_class_specific_data
from train_lora import train_lora
from generate_images import generate_images

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def end_to_end_pipeline(
    model_id="runwayml/stable-diffusion-v1-5",
    metadata_path="data/raw/HAM10000_metadata.csv",
    processed_images_path="data/processed_sd/images",
    specific_class="dermatofibroma",
    num_epochs=50,
    num_images=50,
    output_dir="results",
):
    """
    Run the complete pipeline: data preparation, fine-tuning, and image generation.

    Args:
        model_id: HuggingFace model ID for Stable Diffusion
        metadata_path: Path to HAM10000 metadata CSV
        processed_images_path: Path to processed 512x512 images
        specific_class: Lesion class to train on and generate
        num_epochs: Number of training epochs
        num_images: Number of images to generate
        output_dir: Directory to save results
    """
    logger.info("Starting end-to-end pipeline")

    # Create output directories
    data_dir = os.path.join(output_dir, "data")
    model_dir = os.path.join(output_dir, "model")
    synthetic_dir = os.path.join(output_dir, "synthetic")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(synthetic_dir, exist_ok=True)

    # Step 1: Data preparation
    logger.info(f"Step 1: Preparing data for class '{specific_class}'")
    prompt_file = create_class_specific_data(
        specific_class=specific_class,
        metadata_path=metadata_path,
        processed_images_path=processed_images_path,
        output_dir=data_dir,
    )

    if prompt_file is None:
        logger.error(f"Failed to prepare data for class '{specific_class}'")
        return

    # Step 2: Fine-tune the model
    logger.info(f"Step 2: Fine-tuning the model for class '{specific_class}'")
    model_path = train_lora(
        model_id=model_id,
        train_data_dir=data_dir,
        class_name=specific_class,
        output_dir=model_dir,
        num_epochs=num_epochs,
    )

    # Step 3: Generate synthetic images
    logger.info(
        f"Step 3: Generating {num_images} synthetic images for class '{specific_class}'"
    )
    output_path = generate_images(
        model_id=model_id,
        lora_model_path=model_path,
        class_name=specific_class,
        num_images=num_images,
        output_dir=synthetic_dir,
    )

    logger.info(f"Pipeline completed! Generated images saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for skin lesion image generation"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/raw/HAM10000_metadata.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--processed_images_path",
        type=str,
        default="data/processed_sd/images",
        help="Path to processed images",
    )
    parser.add_argument(
        "--specific_class",
        type=str,
        required=True,
        help="Specific lesion class to train on",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_images", type=int, default=50, help="Number of images to generate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )

    args = parser.parse_args()
    end_to_end_pipeline(**vars(args))
