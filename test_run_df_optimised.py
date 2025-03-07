#!/usr/bin/env python
"""
Optimized test run for dermatofibroma (df) class with faster training.
This script runs the optimized pipeline with settings for speed while preserving quality.
"""

import os
import subprocess
import sys
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Optimized Configuration
CONFIG = {
    "specific_class": "df",
    "num_epochs": 40,  # Reduced from 75 to 40
    "num_images": 50,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,  # Increased from 4 to 8
    "learning_rate": 5e-5,
    "output_dir": "results_df_optimized",
    "model_id": "runwayml/stable-diffusion-v1-5",
    "seed": 42,
    "lora_r": 8,  # Reduced from 16 to 8
    "mixed_precision": "bf16",
    "guidance_scale": 7.5,
    "inference_steps": 50,  # Reduced from 75 to 50
    "target_modules_preset": "efficient",
    "use_8bit_adam": True,
    "train_text_encoder": False,
}


def main():
    """Run the test with the optimized pipeline for dermatofibroma class."""
    start_time = time.time()
    logger.info(f"Starting optimized test run for {CONFIG['specific_class']} class")

    # First make sure we have the train/test split
    split_dir = "data/split"
    if not os.path.exists(os.path.join(split_dir, "train.csv")):
        logger.info("Creating train/test split")
        split_cmd = [
            sys.executable,
            "create_train_test_split.py",
            "--output",
            split_dir,
            "--seed",
            str(CONFIG["seed"]),
        ]
        subprocess.run(split_cmd, check=True)

    # Prepare data
    logger.info("Preparing data for fine-tuning")
    class_specific_dir = os.path.join(
        CONFIG["output_dir"], "data", "fine_tuning_class_specific"
    )
    os.makedirs(class_specific_dir, exist_ok=True)

    data_prep_cmd = [
        sys.executable,
        "data_preparation.py",
        "--specific_class",
        CONFIG["specific_class"],
        "--train_split",
        os.path.join(split_dir, "train.csv"),
        "--output_dir",
        class_specific_dir,
    ]

    subprocess.run(data_prep_cmd, check=True)

    # Train with optimized LoRA
    logger.info("Training LoRA model with optimized settings")
    lora_output_dir = os.path.join(CONFIG["output_dir"], "models", "lora")
    os.makedirs(lora_output_dir, exist_ok=True)

    train_cmd = [
        sys.executable,
        "train_lora.py",
        "--model_id",
        CONFIG["model_id"],
        "--train_data_dir",
        class_specific_dir,
        "--output_dir",
        lora_output_dir,
        "--num_epochs",
        str(CONFIG["num_epochs"]),
        "--batch_size",
        str(CONFIG["batch_size"]),
        "--gradient_accumulation_steps",
        str(CONFIG["gradient_accumulation_steps"]),
        "--learning_rate",
        str(CONFIG["learning_rate"]),
        "--mixed_precision",
        CONFIG["mixed_precision"],
        "--lora_r",
        str(CONFIG["lora_r"]),
        "--class_name",
        CONFIG["specific_class"],
        "--train_split",
        os.path.join(split_dir, "train.csv"),
        "--target_modules_preset",
        CONFIG["target_modules_preset"],
    ]

    if CONFIG["use_8bit_adam"]:
        train_cmd.append("--use_8bit_adam")

    if CONFIG["train_text_encoder"]:
        train_cmd.append("--train_text_encoder")

    subprocess.run(train_cmd, check=True)

    # Generate images
    logger.info("Generating synthetic images")
    lora_model_path = os.path.join(lora_output_dir, "final")  # Use final model
    synthetic_dir = os.path.join(CONFIG["output_dir"], "synthetic")
    os.makedirs(synthetic_dir, exist_ok=True)

    generate_cmd = [
        sys.executable,
        "generate_images.py",
        "--model_id",
        CONFIG["model_id"],
        "--lora_model_path",
        lora_model_path,
        "--num_images",
        str(CONFIG["num_images"]),
        "--output_dir",
        synthetic_dir,
        "--guidance_scale",
        str(CONFIG["guidance_scale"]),
        "--num_inference_steps",
        str(CONFIG["inference_steps"]),
        "--scheduler",
        "ddim",
        "--seed",
        str(CONFIG["seed"]),
        "--class_name",
        CONFIG["specific_class"],
        "--test_prompts",
    ]

    subprocess.run(generate_cmd, check=True)

    # Run evaluation
    logger.info("Evaluating synthetic images")
    synthetic_images_dir = os.path.join(
        synthetic_dir, f"images_{CONFIG['specific_class'].lower()}"
    )
    eval_dir = os.path.join(CONFIG["output_dir"], "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    evaluate_cmd = [
        sys.executable,
        "evaluate_images.py",
        "--real_dir",
        "data/processed_sd/images",
        "--synthetic_dir",
        synthetic_images_dir,
        "--output_dir",
        eval_dir,
        "--metadata_path",
        "data/raw/HAM10000_metadata.csv",
        "--class_name",
        CONFIG["specific_class"],
    ]

    subprocess.run(evaluate_cmd, check=True)

    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Optimized test run completed in {elapsed:.2f} seconds")
    logger.info(f"Results saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
