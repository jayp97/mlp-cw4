#!/usr/bin/env python
# coding=utf-8
"""
generate_synthetic_images.py

Generates synthetic skin lesion images for a given target label using:
  1) The base Stable Diffusion model (e.g. "runwayml/stable-diffusion-v1-5"), plus
  2) The LoRA checkpoint we fine-tuned ("lora_weights.safetensors").

At generation time you specify either:
  --target_label="Dermatofibroma"
or
  --lesion_code="df"   (which maps to "Dermatofibroma" via LABEL_MAP)

This script filters the metadata CSV to find all images matching that label,
then uses each image as an init_image in an img2img pipeline, producing
synthetic output images.

Arguments:
  --pretrained_model_name_or_path : Base Stable Diffusion model (e.g. "runwayml/stable-diffusion-v1-5").
  --metadata_file                 : Path to a CSV (e.g. HAM10000_metadata.csv) with columns [image_id, dx].
  --target_label                  : Full target label (e.g., "Dermatofibroma").
  --lesion_code                   : Alternatively, a short code (e.g., "df"); if target_label is not given, we map the code to a full label.
  --train_data_dir                : Folder containing all processed images (512x512; no subfolders).
  --output_dir                    : Folder where the generated synthetic images will be saved.
  --lora_weights                  : Path to the LoRA-only file (e.g., "lora_weights.safetensors").
  --guidance_scale                : Guidance scale for generation (default=7.5).
  --num_inference_steps           : Number of denoising steps (default=50).
  --strength                      : Strength for the img2img transformation (default=0.8).
  --num_images_per_prompt         : Number of synthetic images to generate per input image (default=5).
  --device                        : "cuda" or "cpu" (default="cuda").

Example Usage:
python generate_synthetic_images.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --metadata_file="data/raw/HAM10000_metadata.csv" \
  --lesion_code="df" \
  --train_data_dir="data/processed_sd/images" \
  --output_dir="data/synthetic/images_dermatofibroma" \
  --lora_weights="models/stable_diffusion_lora/lora_weights.safetensors" \
  --guidance_scale=7.5 \
  --num_inference_steps=50 \
  --strength=0.8 \
  --num_images_per_prompt=10 \
  --device="cuda"
"""

import argparse
import os
import torch
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd

# We'll use the recommended API for loading LoRA into a pipeline
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers import StableDiffusionImg2ImgPipeline

LABEL_MAP = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "nv": "Melanocytic Nevus",
    "mel": "Melanoma",
    "vasc": "Vascular Lesion",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--metadata_file", type=str, required=True, help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default=None,
        help="Full target label for generation (e.g., 'Dermatofibroma')",
    )
    parser.add_argument(
        "--lesion_code",
        type=str,
        default=None,
        help="Short lesion code (e.g., 'df'); if provided and target_label is not, it will be mapped to the full label.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory containing all processed images",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Path to the LoRA-only .safetensors file (e.g., lora_weights.safetensors)",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--num_images_per_prompt", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # If target_label is not provided, derive it from lesion_code.
    if not args.target_label:
        if not args.lesion_code:
            raise ValueError("You must supply either --target_label or --lesion_code.")
        args.target_label = LABEL_MAP.get(args.lesion_code.lower(), args.lesion_code)
        print(f"Derived target_label from lesion_code: {args.target_label}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Read metadata CSV and filter for rows with full label matching the target.
    metadata = pd.read_csv(args.metadata_file)
    metadata["full_label"] = (
        metadata["dx"].str.lower().map(lambda x: LABEL_MAP.get(x, x))
    )
    filtered = metadata[metadata["full_label"].str.lower() == args.target_label.lower()]
    image_ids = filtered["image_id"].tolist()
    image_files = []
    for img_id in image_ids:
        path_jpg = os.path.join(args.train_data_dir, f"{img_id}.jpg")
        if os.path.exists(path_jpg):
            image_files.append(path_jpg)

    if not image_files:
        print(f"No images found for target label '{args.target_label}'. Exiting.")
        return

    # 1) Load the base Stable Diffusion img2img pipeline
    #    We'll add LoRA on top in a moment.
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None,  # user request
    ).to(args.device)

    # 2) If LoRA is provided, load it
    if args.lora_weights and os.path.exists(args.lora_weights):
        try:
            # Simple approach: ask the pipeline to load the LoRA
            pipe.load_lora_weights(args.lora_weights)
            print(f"Loaded LoRA weights from: {args.lora_weights}")
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
    else:
        print(
            "Warning: --lora_weights not provided or file does not exist; using base UNet only."
        )

    pipe.unet.to(args.device, dtype=torch.float16)

    # 3) Generate synthetic images
    prompt = f"(skin lesion, {args.target_label})"
    print(f"Generating images with prompt: {prompt}")

    for img_path in tqdm(image_files, desc="Generating synthetic images"):
        try:
            init_img = (
                Image.open(img_path).convert("RGB").resize((512, 512), Image.BICUBIC)
            )
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        for i in range(args.num_images_per_prompt):
            result = pipe(
                prompt=prompt,
                image=init_img,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
            )
            gen_image = result.images[0]
            out_fname = f"{base_name}_synthetic_{i}.png"
            gen_image.save(os.path.join(args.output_dir, out_fname))

    print(f"Done. Synthetic images saved in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
