#!/usr/bin/env python
"""
Test run for dermatofibroma (df) class with improved pipeline.
This script runs the improved pipeline with optimal settings.
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

# Configuration
CONFIG = {
    "specific_class": "df",
    "num_epochs": 75,
    "num_images": 50,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "output_dir": "results_df_improved",
    "model_id": "runwayml/stable-diffusion-v1-5",
    "seed": 42,
    "lora_r": 16,
    "mixed_precision": "bf16",
    "guidance_scale": 7.5,
    "inference_steps": 75,
}


def main():
    """Run the test with the improved pipeline for dermatofibroma class."""
    start_time = time.time()
    logger.info(f"Starting test run for {CONFIG['specific_class']} class")

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

    # Run the pipeline
    cmd = [
        sys.executable,
        "pipeline.py",
        "--specific_class",
        CONFIG["specific_class"],
        "--num_epochs",
        str(CONFIG["num_epochs"]),
        "--num_images",
        str(CONFIG["num_images"]),
        "--batch_size",
        str(CONFIG["batch_size"]),
        "--gradient_accumulation_steps",
        str(CONFIG["gradient_accumulation_steps"]),
        "--learning_rate",
        str(CONFIG["learning_rate"]),
        "--output_dir",
        CONFIG["output_dir"],
        "--model_id",
        CONFIG["model_id"],
        "--seed",
        str(CONFIG["seed"]),
        "--lora_r",
        str(CONFIG["lora_r"]),
        "--mixed_precision",
        CONFIG["mixed_precision"],
        "--guidance_scale",
        str(CONFIG["guidance_scale"]),
        "--inference_steps",
        str(CONFIG["inference_steps"]),
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Test run completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
