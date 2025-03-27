#!/usr/bin/env python
"""
Fine-tune Stable Diffusion on specific skin lesion classes.
This script allows fine-tuning for any class, with maximum quality
settings to produce the best possible synthetic images.

Usage:
  python fine_tune_class.py --class_name df  # Dermatofibroma
  python fine_tune_class.py --class_name vasc  # Vascular lesions
  python fine_tune_class.py --class_name bcc  # Basal cell carcinoma
  python fine_tune_class.py --class_name akiec  # Actinic keratosis
"""

import os
import argparse
import logging
import subprocess
import time
from datetime import datetime
import json
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default configuration for fine-tuning (max quality settings)
DEFAULT_CONFIG = {
    # Training parameters (max quality)
    "num_epochs": 75,  # Higher number of epochs for better learning
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "model_id": "runwayml/stable-diffusion-v1-5",
    "seed": 42,
    "lora_r": 16,  # Higher LoRA rank for better quality
    "mixed_precision": "bf16",  # bf16 for better quality
    "target_modules_preset": "comprehensive",  # More targeted modules for better quality
    "use_8bit_adam": True,
    "train_text_encoder": True,  # Train text encoder for better prompting
    # Generation parameters (max quality)
    "num_images": 50,  # Number of synthetic images to generate
    "guidance_scale": 7.5,  # Controls how closely to follow the prompt
    "inference_steps": 75,  # Higher number of steps for better quality
    "prompt_template": "{class_name} skin lesion, dermatology image, high quality, close-up, medical photograph, detailed skin texture, sharp focus, medical imaging, clinical documentation, clear detail",
    "negative_prompt": "low quality, blurry, distorted, deformed, unrealistic, cartoon, drawing, painting, watermark, text, signature, border, frame, poorly rendered, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting",
}

# Class-specific configuration customizations
CLASS_CONFIGS = {
    "df": {
        "prompt_template": "dermatofibroma skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, sharp focus, centered, medical imaging, clinical documentation, clear detail, firm nodule"
    },
    "vasc": {
        "prompt_template": "vascular skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, sharp focus, centered, medical imaging, clinical documentation, clear detail, blood vessel anomaly"
    },
    "bcc": {
        "prompt_template": "basal cell carcinoma skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, sharp focus, centered, medical imaging, clinical documentation, clear detail, pearly border"
    },
    "akiec": {
        "prompt_template": "actinic keratosis skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, sharp focus, centered, medical imaging, clinical documentation, clear detail, rough scaly patch"
    },
    "bkl": {
        "prompt_template": "benign keratosis skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, sharp focus, centered, medical imaging, clinical documentation, clear detail, waxy surface"
    },
    "mel": {
        "prompt_template": "melanoma skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, sharp focus, centered, medical imaging, clinical documentation, clear detail, irregular border"
    },
    "nv": {
        "prompt_template": "melanocytic nevus skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, sharp focus, centered, medical imaging, clinical documentation, clear detail, well-circumscribed"
    },
}


def run_command(command, desc=None):
    """Run a shell command and log the output."""
    if desc:
        logger.info(f"Running: {desc}")
    logger.info(f"Command: {' '.join(command)}")

    start_time = time.time()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Print output in real-time
    output = []
    for line in iter(process.stdout.readline, ""):
        line = line.strip()
        output.append(line)
        print(line)

    # Wait for process to complete
    process.wait()
    end_time = time.time()
    elapsed = end_time - start_time

    if process.returncode != 0:
        logger.error(f"Command failed with code {process.returncode}")
        return False, output, elapsed

    logger.info(f"Command completed successfully in {elapsed:.2f} seconds")
    return True, output, elapsed


def fine_tune_class(class_name, output_dir=None, custom_config=None):
    """
    Fine-tune on any skin lesion class with maximum quality settings.
    Uses the train/test split matching the group's classifier.

    Args:
        class_name: Skin lesion class name (df, vasc, bcc, akiec, etc.)
        output_dir: Output directory (defaults to 'results_{class_name}')
        custom_config: Dictionary of custom configuration parameters to override defaults
    """
    # Start timing
    start_time = time.time()

    # Get configuration for this class
    config = DEFAULT_CONFIG.copy()

    # Apply class-specific overrides if available
    if class_name in CLASS_CONFIGS:
        config.update(CLASS_CONFIGS[class_name])

    # Apply custom overrides if provided
    if custom_config:
        config.update(custom_config)

    # Set default output directory if not specified
    if output_dir is None:
        output_dir = f"results_{class_name}_high_quality"

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    data_dir = os.path.join(run_dir, "data")
    model_dir = os.path.join(run_dir, "models")
    synthetic_dir = os.path.join(run_dir, "synthetic")
    eval_dir = os.path.join(run_dir, "evaluation")

    # Create directories
    for directory in [data_dir, model_dir, synthetic_dir, eval_dir]:
        os.makedirs(directory, exist_ok=True)

    # Save configuration
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Track timing for each step
    timings = {}

    # Step 1: Ensure we have the matching split
    logger.info("Step 1: Creating matching train/test split")
    split_dir = os.path.join(data_dir, "split")

    split_cmd = [
        "python",
        "create_matching_split.py",
        "--output",
        split_dir,
        "--seed",
        str(config["seed"]),
    ]

    success, output, elapsed = run_command(split_cmd, "Creating matching split")
    timings["create_split"] = elapsed

    if not success:
        logger.error("Failed to create matching split")
        return False

    train_split_path = os.path.join(split_dir, "train.csv")

    # Step 2: Prepare class-specific data
    logger.info(f"Step 2: Preparing data for {class_name}")
    class_specific_dir = os.path.join(data_dir, "fine_tuning_class_specific")

    data_prep_cmd = [
        "python",
        "data_preparation.py",
        "--specific_class",
        class_name,
        "--output_dir",
        class_specific_dir,
        "--train_split",
        train_split_path,
    ]

    success, output, elapsed = run_command(
        data_prep_cmd, f"Preparing data for {class_name}"
    )
    timings["data_preparation"] = elapsed

    if not success:
        logger.error(f"Failed to prepare data for {class_name}")
        return False

    # Step 3: Train LoRA model with max quality parameters
    logger.info(f"Step 3: Training LoRA model for {class_name}")
    lora_output_dir = os.path.join(model_dir, "lora")

    train_cmd = [
        "python",
        "train_lora.py",
        "--model_id",
        config["model_id"],
        "--train_data_dir",
        class_specific_dir,
        "--output_dir",
        lora_output_dir,
        "--num_epochs",
        str(config["num_epochs"]),
        "--batch_size",
        str(config["batch_size"]),
        "--gradient_accumulation_steps",
        str(config["gradient_accumulation_steps"]),
        "--learning_rate",
        str(config["learning_rate"]),
        "--mixed_precision",
        config["mixed_precision"],
        "--lora_r",
        str(config["lora_r"]),
        "--class_name",
        class_name,
        "--train_split",
        train_split_path,
        "--target_modules_preset",
        config["target_modules_preset"],
    ]

    if config["use_8bit_adam"]:
        train_cmd.append("--use_8bit_adam")

    if config["train_text_encoder"]:
        train_cmd.append("--train_text_encoder")

    success, output, elapsed = run_command(train_cmd, f"Training LoRA for {class_name}")
    timings["train_lora"] = elapsed

    if not success:
        logger.error(f"Failed to train LoRA model for {class_name}")
        return False

    # Step 4: Generate synthetic images
    logger.info(f"Step 4: Generating synthetic {class_name} images")
    lora_model_path = os.path.join(lora_output_dir, "final")

    generate_cmd = [
        "python",
        "generate_images.py",
        "--model_id",
        config["model_id"],
        "--lora_model_path",
        lora_model_path,
        "--num_images",
        str(config["num_images"]),
        "--output_dir",
        synthetic_dir,
        "--guidance_scale",
        str(config["guidance_scale"]),
        "--num_inference_steps",
        str(config["inference_steps"]),
        "--scheduler",
        "ddim",
        "--seed",
        str(config["seed"]),
        "--class_name",
        class_name,
        "--prompt_template",
        config["prompt_template"],
        "--negative_prompt",
        config["negative_prompt"],
    ]

    success, output, elapsed = run_command(
        generate_cmd, f"Generating {class_name} images"
    )
    timings["generate_images"] = elapsed

    if not success:
        logger.error(f"Failed to generate {class_name} images")
        return False

    # Extract synthetic images directory
    synthetic_images_dir = os.path.join(synthetic_dir, f"images_{class_name}")

    # Step 5: Evaluate synthetic images
    logger.info(f"Step 5: Evaluating {class_name} synthetic images")

    # First, filter the train.csv file to only include images of this class
    train_df = pd.read_csv(train_split_path)
    class_df = train_df[train_df["dx"] == class_name]
    class_ids = class_df["image_id"].tolist()

    # Now get the paths to the real images
    real_images_dir = "data/processed_sd/images"

    evaluate_cmd = [
        "python",
        "evaluate_images.py",
        "--real_dir",
        real_images_dir,
        "--synthetic_dir",
        synthetic_images_dir,
        "--output_dir",
        eval_dir,
        "--metadata_path",
        "data/raw/HAM10000_metadata.csv",
        "--class_name",
        class_name,
    ]

    success, output, elapsed = run_command(
        evaluate_cmd, f"Evaluating {class_name} images"
    )
    timings["evaluate_images"] = elapsed

    if not success:
        logger.warning(f"Evaluation of {class_name} images failed, but continuing")

    # Save timings
    with open(os.path.join(run_dir, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)

    # Calculate total time
    total_time = sum(timings.values())
    logger.info(f"Fine-tuning of {class_name} completed in {total_time:.2f} seconds")
    logger.info(f"Results saved to: {run_dir}")

    # Print timing summary
    logger.info("Time spent on each step:")
    for step, time_taken in timings.items():
        percentage = (time_taken / total_time) * 100
        logger.info(f"  {step}: {time_taken:.2f} seconds ({percentage:.1f}%)")

    # Return results info
    return {
        "success": True,
        "run_dir": run_dir,
        "synthetic_dir": synthetic_images_dir,
        "model_path": lora_model_path,
        "evaluation_dir": eval_dir,
        "total_time": total_time,
        "timings": timings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion on any skin lesion class with max quality settings"
    )
    parser.add_argument(
        "--class_name",
        type=str,
        required=True,
        help="Class to fine-tune on (df, vasc, bcc, akiec, etc.)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to results_{class_name}_high_quality)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Number of synthetic images to generate",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    parser.add_argument(
        "--no_train_text_encoder",
        action="store_true",
        help="Don't train the text encoder (faster but lower quality)",
    )
    parser.add_argument(
        "--target_modules_preset",
        type=str,
        choices=["minimal", "efficient", "comprehensive"],
        default=None,
        help="Which set of modules to target with LoRA",
    )

    args = parser.parse_args()

    # Create custom config from command line arguments
    custom_config = {}
    if args.num_epochs is not None:
        custom_config["num_epochs"] = args.num_epochs
    if args.num_images is not None:
        custom_config["num_images"] = args.num_images
    if args.learning_rate is not None:
        custom_config["learning_rate"] = args.learning_rate
    if args.lora_r is not None:
        custom_config["lora_r"] = args.lora_r
    if args.no_train_text_encoder:
        custom_config["train_text_encoder"] = False
    if args.target_modules_preset is not None:
        custom_config["target_modules_preset"] = args.target_modules_preset

    fine_tune_class(
        class_name=args.class_name,
        output_dir=args.output_dir,
        custom_config=custom_config,
    )
