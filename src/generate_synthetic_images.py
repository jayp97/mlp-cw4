#!/usr/bin/env python
# coding=utf-8

"""
generate_synthetic_images.py

Loads a base Stable Diffusion model + LoRA checkpoint (trained on the entire HAM10000 dataset).
Specify a lesion type (e.g. --lesion_code="df"), and it will produce a prompt like
"A photo of a Dermatofibroma lesion" to generate images.

Example usage:
---------------
python src/generate_synthetic_images.py \
  --pretrained_model="runwayml/stable-diffusion-v1-5" \
  --lora_weights="models/stable_diffusion_lora/pytorch_lora_weights.safetensors" \
  --lesion_code="df" \
  --num_images=20 \
  --guidance_scale=7.5 \
  --num_inference_steps=50 \
  --seed=42 \
  --output_dir="data/synthetic/images_dermatofibroma"
"""

import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

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
        help="Stable Diffusion base model, e.g. 'runwayml/stable-diffusion-v1-5'.",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA checkpoint from train_lora.py.",
    )
    parser.add_argument(
        "--lesion_code",
        type=str,
        default="df",
        help="Which lesion code to generate. E.g. 'df' for Dermatofibroma.",
    )
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="synthetic_output")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.lesion_code not in LABEL_MAP:
        raise ValueError(
            f"Unknown lesion_code '{args.lesion_code}'. "
            f"Must be one of {list(LABEL_MAP.keys())}."
        )

    label_text = LABEL_MAP[args.lesion_code]
    prompt = f"A photo of a {label_text} lesion"

    # 1) Load pipeline in half-precision
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model, torch_dtype=torch.float16
    ).to("cuda")

    # 2) Load LoRA
    pipe.load_lora_weights(args.lora_weights)

    # 3) Generate images
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    for i in range(args.num_images):
        out = pipe(
            prompt=prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        image = out.images[0]
        filename = f"{args.lesion_code}_synthetic_{i:03d}.png"
        image.save(os.path.join(args.output_dir, filename))
        print(f"Saved {filename}")


if __name__ == "__main__":
    main()
