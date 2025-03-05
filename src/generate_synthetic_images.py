#!/usr/bin/env python
# coding=utf-8

"""
generate_synthetic_images.py

Loads a base Stable Diffusion model along with a LoRA checkpoint (trained on the entire HAM10000 dataset).
You can specify which lesion type to generate by passing --lesion_code (e.g. "df" for Dermatofibroma).
The script constructs a prompt like "A photo of a Dermatofibroma lesion" and generates synthetic images.

Example usage:
---------------
python generate_synthetic_images.py \
  --pretrained_model "runwayml/stable-diffusion-v1-5" \
  --lora_weights "models/stable_diffusion_lora/pytorch_lora_weights.safetensors" \
  --lesion_code "df" \
  --num_images 20 \
  --guidance_scale 7.5 \
  --num_inference_steps 50 \
  --seed 42 \
  --output_dir "data/synthetic/images_dermatofibroma"
"""

import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Dictionary mapping lesion codes to full labels (same as in train_lora.py)
LABEL_MAP = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "nv": "Melanocytic Nevus",
    "mel": "Melanoma",
    "vasc": "Vascular lesion",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Base stable diffusion model, e.g. 'runwayml/stable-diffusion-v1-5'.",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA checkpoint (pytorch_lora_weights.safetensors) from train_lora.py.",
    )
    parser.add_argument(
        "--lesion_code",
        type=str,
        default="df",
        help="Lesion code to generate (e.g. 'df' for Dermatofibroma).",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps during generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="synthetic_output",
        help="Directory to save generated images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert lesion code to its full textual label
    if args.lesion_code not in LABEL_MAP:
        raise ValueError(
            f"Unknown lesion code '{args.lesion_code}'. Valid codes: {list(LABEL_MAP.keys())}"
        )
    label_text = LABEL_MAP[args.lesion_code]
    prompt = f"A photo of a {label_text} lesion"

    # Load the Stable Diffusion pipeline in FP16
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model, torch_dtype=torch.float16
    ).to("cuda")

    # Load LoRA weights into the pipeline (this method loads them into both UNet and text encoder)
    pipe.load_lora_weights(args.lora_weights)

    # Set a generator for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Generate and save images
    for i in range(args.num_images):
        result = pipe(
            prompt=prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        image = result.images[0]
        out_name = f"{args.lesion_code}_synthetic_{i:03d}.png"
        out_path = os.path.join(args.output_dir, out_name)
        image.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
