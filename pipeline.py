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

# Add this dictionary near the top of pipeline.py
CLASS_MAPPING = {
    # Full names to codes
    "dermatofibroma": "df",
    "melanocytic_nevi": "nv",
    "melanoma": "mel",
    "benign_keratosis": "bkl",
    "basal_cell_carcinoma": "bcc",
    "actinic_keratoses": "akiec",
    "vascular_lesions": "vasc",
    # Codes remain the same
    "df": "df",
    "nv": "nv",
    "mel": "mel",
    "bkl": "bkl",
    "bcc": "bcc",
    "akiec": "akiec",
    "vasc": "vasc",
}


# Then in the end_to_end_pipeline function, update the specific_class parameter:
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
    """
    logger.info("Starting end-to-end pipeline")

    # Map class name to dataset code if needed
    if specific_class in CLASS_MAPPING:
        dataset_class_code = CLASS_MAPPING[specific_class]
    else:
        dataset_class_code = specific_class

    # Use full name for display/output but code for data preparation
    display_class_name = specific_class

    # Create output directories
    data_dir = os.path.join(output_dir, "data")
    model_dir = os.path.join(output_dir, "model")
    synthetic_dir = os.path.join(output_dir, "synthetic")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(synthetic_dir, exist_ok=True)

    # Step 1: Data preparation using the dataset code
    logger.info(f"Step 1: Preparing data for class '{display_class_name}'")
    prompt_file = create_class_specific_data(
        specific_class=dataset_class_code,
        metadata_path=metadata_path,
        processed_images_path=processed_images_path,
        output_dir=data_dir,
        prompt_template="{lesion_type} skin lesion, dermatology image",  # Use code here
    )

    if prompt_file is None:
        logger.error(f"Failed to prepare data for class '{display_class_name}'")
        return

    # Remaining function stays the same, but we pass dataset_class_code to the training function
    # and display_class_name to the generation function with mapping...


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
