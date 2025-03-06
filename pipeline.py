#!/usr/bin/env python
import os
import argparse
import logging
import sys
import subprocess
import time
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def run_pipeline(
    specific_class=None,
    num_epochs=50,
    num_images=50,
    batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    output_dir="results",
    model_id="runwayml/stable-diffusion-v1-5",
    seed=42,
    use_train_split=True,
    evaluate=True,
    lora_r=16,
    mixed_precision="bf16",
    guidance_scale=7.5,
    inference_steps=75,
    use_test_prompts=True,
):
    """
    Run the full pipeline for fine-tuning and image generation.

    Args:
        specific_class: The specific lesion class to focus on (e.g., 'df')
        num_epochs: Number of training epochs
        num_images: Number of synthetic images to generate
        batch_size: Batch size for training
        gradient_accumulation_steps: Steps for gradient accumulation
        learning_rate: Learning rate for training
        output_dir: Base output directory
        model_id: HuggingFace model ID for Stable Diffusion
        seed: Random seed for reproducibility
        use_train_split: Whether to create and use train/test split
        evaluate: Whether to run evaluation after generation
        lora_r: LoRA rank for fine-tuning
        mixed_precision: Mixed precision mode for training
        guidance_scale: Classifier-free guidance scale for generation
        inference_steps: Number of inference steps for generation
        use_test_prompts: Whether to use different test prompts for generation
    """
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    class_suffix = f"_{specific_class}" if specific_class else ""
    run_dir = os.path.join(output_dir, f"run_{timestamp}{class_suffix}")

    # Create subdirectories
    data_dir = os.path.join(run_dir, "data")
    model_dir = os.path.join(run_dir, "models")
    synthetic_dir = os.path.join(run_dir, "synthetic")
    eval_dir = os.path.join(run_dir, "evaluation")

    for directory in [data_dir, model_dir, synthetic_dir, eval_dir]:
        os.makedirs(directory, exist_ok=True)

    # Save run configuration
    config = {
        "specific_class": specific_class,
        "num_epochs": num_epochs,
        "num_images": num_images,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "output_dir": output_dir,
        "model_id": model_id,
        "seed": seed,
        "use_train_split": use_train_split,
        "evaluate": evaluate,
        "lora_r": lora_r,
        "mixed_precision": mixed_precision,
        "guidance_scale": guidance_scale,
        "inference_steps": inference_steps,
        "use_test_prompts": use_test_prompts,
        "timestamp": timestamp,
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Track time for each step
    timings = {}

    # Step 1: Create train/test split if requested
    train_split_path = None
    test_split_path = None
    if use_train_split:
        logger.info("Step 1: Creating train/test split")
        split_output_dir = os.path.join(data_dir, "split")

        split_cmd = [
            sys.executable,
            "create_train_test_split.py",
            "--output",
            split_output_dir,
            "--seed",
            str(seed),
        ]

        success, output, elapsed = run_command(split_cmd, "Creating train/test split")
        timings["create_split"] = elapsed

        if not success:
            logger.error("Failed to create train/test split")
            return False

        train_split_path = os.path.join(split_output_dir, "train.csv")
        test_split_path = os.path.join(split_output_dir, "test.csv")

    # Step 2: Prepare data for fine-tuning
    logger.info("Step 2: Preparing data for fine-tuning")
    class_specific_dir = os.path.join(data_dir, "fine_tuning_class_specific")

    data_prep_cmd = [
        sys.executable,
        "data_preparation.py",
    ]

    if specific_class:
        data_prep_cmd.extend(["--specific_class", specific_class])

    if train_split_path:
        data_prep_cmd.extend(["--train_split", train_split_path])

    success, output, elapsed = run_command(
        data_prep_cmd, "Preparing data for fine-tuning"
    )
    timings["data_preparation"] = elapsed

    if not success:
        logger.error("Failed to prepare data for fine-tuning")
        return False

    # Step 3: Train LoRA model
    logger.info("Step 3: Training LoRA model")
    lora_output_dir = os.path.join(model_dir, "lora")

    train_cmd = [
        sys.executable,
        "train_lora.py",
        "--model_id",
        model_id,
        "--train_data_dir",
        class_specific_dir,
        "--output_dir",
        lora_output_dir,
        "--num_epochs",
        str(num_epochs),
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--learning_rate",
        str(learning_rate),
        "--mixed_precision",
        mixed_precision,
        "--lora_r",
        str(lora_r),
    ]

    if specific_class:
        train_cmd.extend(["--class_name", specific_class])

    if train_split_path:
        train_cmd.extend(["--train_split", train_split_path])

    success, output, elapsed = run_command(train_cmd, "Training LoRA model")
    timings["train_lora"] = elapsed

    if not success:
        logger.error("Failed to train LoRA model")
        return False

    # Step 4: Generate synthetic images
    logger.info("Step 4: Generating synthetic images")
    lora_model_path = os.path.join(lora_output_dir, "final")  # Use final model

    generate_cmd = [
        sys.executable,
        "generate_images.py",
        "--model_id",
        model_id,
        "--lora_model_path",
        lora_model_path,
        "--num_images",
        str(num_images),
        "--output_dir",
        synthetic_dir,
        "--guidance_scale",
        str(guidance_scale),
        "--num_inference_steps",
        str(inference_steps),
        "--scheduler",
        "ddim",  # Use DDIM scheduler
        "--seed",
        str(seed) if seed is not None else "42",  # Use a seed for reproducibility
    ]

    if specific_class:
        generate_cmd.extend(["--class_name", specific_class])

    if use_test_prompts:
        generate_cmd.append("--test_prompts")

    success, output, elapsed = run_command(generate_cmd, "Generating synthetic images")
    timings["generate_images"] = elapsed

    if not success:
        logger.error("Failed to generate synthetic images")
        return False

    # Extract synthetic images directory from output
    synthetic_images_dir = None
    for line in output:
        if "Generated" in line and "images for class" in line and "at:" in line:
            parts = line.split("at:")
            if len(parts) > 1:
                synthetic_images_dir = parts[1].strip()
                break

    if not synthetic_images_dir:
        synthetic_images_dir = (
            f"{synthetic_dir}/images_{specific_class.lower()}"
            if specific_class
            else f"{synthetic_dir}/images"
        )

    # Step 5: Evaluate synthetic images (if requested)
    if evaluate:
        logger.info("Step 5: Evaluating synthetic images")
        real_images_dir = "data/processed_sd/images"  # Default location

        evaluate_cmd = [
            sys.executable,
            "evaluate_images.py",
            "--real_dir",
            real_images_dir,
            "--synthetic_dir",
            synthetic_images_dir,
            "--output_dir",
            eval_dir,
            "--metadata_path",
            "data/raw/HAM10000_metadata.csv",
        ]

        if specific_class:
            evaluate_cmd.extend(["--class_name", specific_class])

        success, output, elapsed = run_command(
            evaluate_cmd, "Evaluating synthetic images"
        )
        timings["evaluate_images"] = elapsed

        if not success:
            logger.warning("Evaluation of synthetic images failed, but continuing")

    # Step 6: Visualize training loss
    logger.info("Step 6: Visualizing training loss")
    loss_files = []
    for root, dirs, files in os.walk(lora_output_dir):
        for file in files:
            if file.startswith("loss_log_") and file.endswith(".csv"):
                loss_files.append(os.path.join(root, file))

    if loss_files:
        loss_file = loss_files[0]  # Use the first loss log file

        visualize_cmd = [
            sys.executable,
            "visualise_loss.py",
            loss_file,
            "--output_dir",
            os.path.join(model_dir, "loss_plots"),
        ]

        if specific_class:
            visualize_cmd.extend(["--class_name", specific_class])

        success, output, elapsed = run_command(
            visualize_cmd, "Visualizing training loss"
        )
        timings["visualize_loss"] = elapsed

        if not success:
            logger.warning("Visualization of training loss failed, but continuing")

    # Save timings
    with open(os.path.join(run_dir, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)

    # Calculate total time
    total_time = sum(timings.values())
    logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
    logger.info(f"Results saved to: {run_dir}")

    # Print a summary of timings
    logger.info("Time spent on each step:")
    for step, time_taken in timings.items():
        percentage = (time_taken / total_time) * 100
        logger.info(f"  {step}: {time_taken:.2f} seconds ({percentage:.1f}%)")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full pipeline for fine-tuning and image generation"
    )
    parser.add_argument(
        "--specific_class",
        type=str,
        default=None,
        help="Specific lesion class to focus on (e.g., 'df')",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
        help="Number of synthetic images to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Steps for gradient accumulation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base output directory",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model ID for Stable Diffusion",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no_train_split",
        action="store_false",
        dest="use_train_split",
        help="Do not create and use train/test split",
    )
    parser.add_argument(
        "--no_evaluate",
        action="store_false",
        dest="evaluate",
        help="Do not run evaluation after generation",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank for fine-tuning",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode for training",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale for generation",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=75,
        help="Number of inference steps for generation",
    )
    parser.add_argument(
        "--no_test_prompts",
        action="store_false",
        dest="use_test_prompts",
        help="Do not use different test prompts for generation",
    )

    args = parser.parse_args()

    # Set default values
    parser.set_defaults(use_train_split=True, evaluate=True, use_test_prompts=True)

    run_pipeline(
        specific_class=args.specific_class,
        num_epochs=args.num_epochs,
        num_images=args.num_images,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        model_id=args.model_id,
        seed=args.seed,
        use_train_split=args.use_train_split,
        evaluate=args.evaluate,
        lora_r=args.lora_r,
        mixed_precision=args.mixed_precision,
        guidance_scale=args.guidance_scale,
        inference_steps=args.inference_steps,
        use_test_prompts=args.use_test_prompts,
    )
