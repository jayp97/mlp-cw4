#!/usr/bin/env python
import argparse
import os
import subprocess

# Predefined high-quality prompts for each class
PROMPTS = {
    "df": "dermatofibroma skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, sharp focus, centered, medical imaging, clinical documentation, clear detail, firm nodule",
    "vasc": "vascular skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, sharp focus, centered, medical imaging, clinical documentation, clear detail, blood vessel anomaly",
    "akiec": "actinic keratosis skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, sharp focus, centered, medical imaging, clinical documentation, clear detail, rough scaly patch",
    "bcc": "basal cell carcinoma skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, sharp focus, centered, medical imaging, clinical documentation, clear detail, pearly border",
}

NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, unrealistic, cartoon, drawing, painting, watermark, text, signature, border, frame, poorly rendered, bad anatomy"


def main():
    parser = argparse.ArgumentParser(
        description="Generate specific number of images for a class"
    )
    parser.add_argument(
        "--class_name",
        type=str,
        required=True,
        choices=["df", "vasc", "akiec", "bcc"],
        help="Class to generate images for",
    )
    parser.add_argument(
        "--num_images", type=int, required=True, help="Number of images to generate"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: generated_{class_name})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set default output directory if none specified
    if args.output_dir is None:
        args.output_dir = f"generated_{args.class_name}"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get prompt for the class
    prompt = PROMPTS.get(
        args.class_name, f"{args.class_name} skin lesion, high quality, medical image"
    )

    # Build command
    cmd = [
        "python",
        "generate_images.py",
        "--model_id",
        "runwayml/stable-diffusion-v1-5",
        "--lora_model_path",
        args.model_path,
        "--num_images",
        str(args.num_images),
        "--output_dir",
        args.output_dir,
        "--class_name",
        args.class_name,
        "--scheduler",
        "ddim",
        "--num_inference_steps",
        "75",
        "--guidance_scale",
        "7.5",
        "--seed",
        str(args.seed),
        "--prompt_template",
        prompt,
        "--negative_prompt",
        NEGATIVE_PROMPT,
    ]

    # Run command
    print(f"Generating {args.num_images} images for class {args.class_name}...")
    print(f"Using prompt: {prompt}")
    subprocess.run(cmd)

    print(f"Done! Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
