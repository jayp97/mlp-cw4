#!/usr/bin/env python
# coding=utf-8
"""
generate_synthetic_images.py

Generates synthetic skin lesion images for a given target label using a fine-tuned
Stable Diffusion model (with LoRA). At generation time, you specify either:
  - --target_label="Dermatofibroma"
  - or --lesion_code="df"  (which maps to "Dermatofibroma" via LABEL_MAP)

The script filters the metadata CSV to find all images matching that label, then uses
each image as an init_image in an img2img pipeline, producing synthetic output images.

Arguments:
  --pretrained_model_name_or_path : Base Stable Diffusion model (e.g. "runwayml/stable-diffusion-v1-5").
  --metadata_file                 : Path to a CSV (e.g. HAM10000_metadata.csv).
  --target_label                  : e.g. "Dermatofibroma".
  --lesion_code                   : e.g. "df". If target_label is not given, we map code -> label.
  --train_data_dir                : Folder with 512x512 images (no subfolders).
  --output_dir                    : Where to save generated synthetic images.
  --lora_weights                  : Path to LoRA weights file (if any), e.g. "adapter_model.safetensors".
  --guidance_scale                : Guidance scale for generation (default=7.5).
  --num_inference_steps           : Steps for diffusion (default=50).
  --strength                      : Strength for img2img (default=0.8).
  --num_images_per_prompt         : How many synthetic images per real image (default=5).
  --device                        : "cuda" or "cpu" (default="cuda").

Example Usage:
  python generate_synthetic_images.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --metadata_file="data/raw/HAM10000_metadata.csv" \
    --lesion_code="df" \
    --train_data_dir="data/processed_sd/images" \
    --output_dir="data/synthetic/images_dermatofibroma" \
    --lora_weights="models/stable_diffusion_lora/adapter_model.safetensors" \
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

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.loaders import LoraLoaderMixin
import pandas as pd

# Short lesion codes to full label:
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
        "--metadata_file",
        type=str,
        required=True,
        help="CSV with at least columns [image_id, dx].",
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default=None,
        help="Full label for generation (e.g., 'Dermatofibroma').",
    )
    parser.add_argument(
        "--lesion_code",
        type=str,
        default=None,
        help="Short code (e.g., 'df'). If target_label not provided, we map code->label.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Folder containing 512x512 images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder to save the generated images.",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Path to LoRA weights (e.g., 'adapter_model.safetensors').",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--num_images_per_prompt", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # Derive target_label from lesion_code if needed
    if not args.target_label:
        if not args.lesion_code:
            raise ValueError("You must provide either --target_label or --lesion_code.")
        # Map short code to label
        args.target_label = LABEL_MAP.get(args.lesion_code.lower(), args.lesion_code)
        print(f"Derived target_label from lesion_code: {args.target_label}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Read the metadata and filter the images for the chosen label
    df = pd.read_csv(args.metadata_file)
    df["full_label"] = df["dx"].str.lower().map(lambda x: LABEL_MAP.get(x, x))
    subset = df[df["full_label"].str.lower() == args.target_label.lower()]
    image_ids = subset["image_id"].tolist()

    # Build a list of existing image file paths
    image_files = [
        os.path.join(args.train_data_dir, f"{img_id}.jpg")
        for img_id in image_ids
        if os.path.exists(os.path.join(args.train_data_dir, f"{img_id}.jpg"))
    ]
    if not image_files:
        print(f"No images found for target label '{args.target_label}'. Exiting.")
        return

    # 2) Load the pipeline from the base model
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None,  # user-chosen
    ).to(args.device)

    # 3) If LoRA weights are provided, load them using named arguments
    # The function signature typically is:
    #   load_lora_into_unet(unet, lora_state_dict_or_path, ...)
    # So we do:
    if args.lora_weights and os.path.exists(args.lora_weights):
        try:
            LoraLoaderMixin.load_lora_into_unet(
                unet=pipe.unet, state_dict_or_path=args.lora_weights
            )
            print(f"Loaded LoRA from {args.lora_weights}")
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
    else:
        print("Warning: no valid --lora_weights provided. Using base UNet only.")

    # 4) Synthesize images with img2img
    prompt = f"(skin lesion, {args.target_label})"
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
