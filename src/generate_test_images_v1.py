#!/usr/bin/env python
"""
Generate a few test images and measure generation time.
This script uses your trained model to generate a small number of images
and reports timing information.
"""

import os
import sys
import time
import argparse
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import PeftModel
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_test_images(
    model_path,
    class_name="df",
    num_images=5,
    output_dir="test_generations",
    prompt=None,
    negative_prompt="low quality, blurry, distorted, deformed, unrealistic, cartoon, drawing, painting",
    seed=42,
):
    """Generate test images and time the process"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine model paths
    base_model_id = "runwayml/stable-diffusion-v1-5"
    lora_model_path = os.path.join(model_path, "final")

    if not os.path.exists(lora_model_path):
        lora_model_path = model_path
        logger.info(f"Using model path directly: {lora_model_path}")
    else:
        logger.info(f"Using final model from: {lora_model_path}")

    # Generate default prompt if none provided
    if prompt is None:
        prompt = f"{class_name} skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture"

    # Log information
    logger.info(f"Generating {num_images} test images")
    logger.info(f"Class: {class_name}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Output directory: {output_dir}")

    # Load model
    logger.info("Loading model...")
    start_time = time.time()

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # Use DDIM scheduler for better quality
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load LoRA weights
    logger.info(f"Loading LoRA weights from: {lora_model_path}")
    unet_lora_path = os.path.join(lora_model_path, "unet_lora")
    if os.path.exists(unet_lora_path):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora_path)
    else:
        logger.error(f"LoRA weights not found at {unet_lora_path}")
        return

    # Load text encoder LoRA weights if available
    text_encoder_lora_path = os.path.join(lora_model_path, "text_encoder_lora")
    if os.path.exists(text_encoder_lora_path):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_lora_path
        )
        logger.info("Loaded text encoder LoRA weights")

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")

    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Generate images and time each generation
    generation_times = []

    for i in range(num_images):
        logger.info(f"Generating image {i+1}/{num_images}...")

        # Set unique seed for this image
        image_seed = seed + i if seed is not None else None
        if image_seed is not None:
            generator = torch.Generator(device=device).manual_seed(image_seed)
        else:
            generator = None

        # Time generation
        start_time = time.time()

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            num_inference_steps=50,
            height=512,
            width=512,
            generator=generator,
        ).images[0]

        generation_time = time.time() - start_time
        generation_times.append(generation_time)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(output_dir, f"{class_name}_{i+1}_{timestamp}.png")
        image.save(image_path)

        logger.info(f"Image {i+1} generated in {generation_time:.2f} seconds")

    # Report timing statistics
    avg_time = sum(generation_times) / len(generation_times)
    min_time = min(generation_times)
    max_time = max(generation_times)

    logger.info("\nGeneration Time Statistics:")
    logger.info(f"Average time per image: {avg_time:.2f} seconds")
    logger.info(f"Minimum time: {min_time:.2f} seconds")
    logger.info(f"Maximum time: {max_time:.2f} seconds")
    logger.info(
        f"Total time for {num_images} images: {sum(generation_times):.2f} seconds"
    )

    # Write statistics to file
    stats_path = os.path.join(output_dir, "generation_stats.txt")
    with open(stats_path, "w") as f:
        f.write("Generation Time Statistics:\n")
        f.write(f"Class: {class_name}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Negative prompt: {negative_prompt}\n")
        f.write(f"Number of images: {num_images}\n")
        f.write(f"Average time per image: {avg_time:.2f} seconds\n")
        f.write(f"Minimum time: {min_time:.2f} seconds\n")
        f.write(f"Maximum time: {max_time:.2f} seconds\n")
        f.write(
            f"Total time for {num_images} images: {sum(generation_times):.2f} seconds\n"
        )
        f.write("\nDetailed timings:\n")
        for i, t in enumerate(generation_times):
            f.write(f"Image {i+1}: {t:.2f} seconds\n")

    logger.info(f"Statistics saved to {stats_path}")
    logger.info(f"Images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and time test images")
    parser.add_argument(
        "--model_path",
        type=str,
        default="results_df_optimized/models/lora",
        help="Path to model directory",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="df",
        help="Class name",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_generations",
        help="Output directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    generate_test_images(
        model_path=args.model_path,
        class_name=args.class_name,
        num_images=args.num_images,
        output_dir=args.output_dir,
        prompt=args.prompt,
        seed=args.seed,
    )
