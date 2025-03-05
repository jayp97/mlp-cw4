#!/usr/bin/env python
# coding=utf-8
"""
generate_synthetic_images.py

Generates synthetic skin lesion images for a given target label using a fine-tuned
Stable Diffusion model (with LoRA). At generation time, you specify the label or code
(e.g., "Dermatofibroma" or "df"), and the script filters the metadata CSV to get images
of that type, then uses them as init images in an img2img pipeline.

Arguments:
  --pretrained_model_name_or_path : Path or identifier of the base Stable Diffusion model.
  --metadata_file                 : Path to the metadata CSV file (HAM10000_metadata.csv, etc).
  --target_label                  : Full target label for generation (e.g., "Dermatofibroma").
  --lesion_code                   : Alternatively, a short lesion code (e.g., "df"). If provided and target_label is not,
                                    it will be mapped to the full label via LABEL_MAP.
  --train_data_dir                : Directory containing all processed images (512x512).
  --output_dir                    : Where to save synthetic images.
  --lora_weights                  : (Optional) Path to the LoRA weights file (e.g. "adapter_model.safetensors").
  --guidance_scale                : Guidance scale for generation.
  --num_inference_steps           : Number of denoising steps.
  --strength                      : Strength of the img2img transformation.
  --num_images_per_prompt         : Number of synthetic images to generate per input image.
  --device                        : Device to use (default "cuda").

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

# Mapping from short lesion codes to full labels:
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
        help="Path to your metadata CSV file.",
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default=None,
        help="Full target label (e.g., 'Dermatofibroma').",
    )
    parser.add_argument(
        "--lesion_code",
        type=str,
        default=None,
        help="Short code (e.g., 'df'). If supplied and target_label is missing, we map code -> label.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory with all processed images (512x512).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the synthetic images.",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Path to LoRA weights file (e.g. 'adapter_model.safetensors').",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--num_images_per_prompt", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # If target_label is None, try to derive from lesion_code.
    if args.target_label is None:
        if args.lesion_code is None:
            raise ValueError("Must supply either --target_label or --lesion_code.")
        else:
            # Map code -> label
            derived_label = LABEL_MAP.get(args.lesion_code.lower(), args.lesion_code)
            args.target_label = derived_label
            print(f"Derived target_label from lesion_code: {args.target_label}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Read metadata CSV and filter rows matching the target label
    metadata = pd.read_csv(args.metadata_file)
    metadata["full_label"] = (
        metadata["dx"].str.lower().map(lambda x: LABEL_MAP.get(x, x))
    )
    filtered = metadata[metadata["full_label"].str.lower() == args.target_label.lower()]
    image_ids = filtered["image_id"].tolist()
    image_files = [
        os.path.join(args.train_data_dir, f"{img_id}.jpg")
        for img_id in image_ids
        if os.path.exists(os.path.join(args.train_data_dir, f"{img_id}.jpg"))
    ]
    if not image_files:
        print(f"No images found for target label: {args.target_label}")
        return

    # 1) Load the Stable Diffusion img2img pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None,  # user chooses to disable it
    )
    pipe = pipe.to(args.device)

    # 2) Load fine-tuned LoRA weights if provided
    if args.lora_weights is not None and os.path.exists(args.lora_weights):
        # Attempt to load a saved base model state first (if present)
        lora_model_path = os.path.join(args.output_dir, "pytorch_model.bin")
        if os.path.exists(lora_model_path):
            pipe.unet.load_state_dict(
                torch.load(lora_model_path, map_location="cpu"), strict=False
            )

        # Important: The method signature is: load_lora_into_unet(unet, weights_path, ...)
        # But your local version might be reversed or differ. We'll match your error message
        # which says the first param is unet, second param is the weights path:
        LoraLoaderMixin.load_lora_into_unet(pipe.unet, args.lora_weights)
    else:
        print(
            "Warning: either --lora_weights not provided, or the file doesn't exist. Using base UNet..."
        )

    pipe.unet.to(args.device, dtype=torch.float16)

    # 3) Generate images using img2img
    prompt = f"(skin lesion, {args.target_label})"
    for img_path in tqdm(image_files, desc="Generating synthetic images"):
        try:
            init_image = (
                Image.open(img_path).convert("RGB").resize((512, 512), Image.BICUBIC)
            )
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        for i in range(args.num_images_per_prompt):
            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
            )
            gen_image = result.images[0]
            out_fname = f"{base_name}_synthetic_{i}.png"
            gen_image.save(os.path.join(args.output_dir, out_fname))

    print(f"Generation complete. Synthetic images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
