#!/usr/bin/env python
# coding=utf-8
"""
generate_synthetic_images.py

Generates synthetic skin lesion images for a given target label using a fine-tuned
Stable Diffusion model augmented by LoRA.
"""

import argparse
import os
import torch
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
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
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--target_label", type=str, default=None)
    parser.add_argument("--lesion_code", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--num_images_per_prompt", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.target_label:
        args.target_label = LABEL_MAP.get(args.lesion_code.lower(), args.lesion_code)
        print(f"Derived target_label: {args.target_label}")

    os.makedirs(args.output_dir, exist_ok=True)
    metadata = pd.read_csv(args.metadata_file)
    metadata["full_label"] = (
        metadata["dx"].str.lower().map(lambda x: LABEL_MAP.get(x, x))
    )
    filtered = metadata[metadata["full_label"].str.lower() == args.target_label.lower()]
    image_files = [
        os.path.join(args.train_data_dir, f"{img_id}.jpg")
        for img_id in filtered["image_id"]
        if os.path.exists(os.path.join(args.train_data_dir, f"{img_id}.jpg"))
    ]

    if not image_files:
        print(f"No images found for '{args.target_label}'")
        return

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(args.device)

    if args.lora_weights and os.path.exists(args.lora_weights):
        try:
            # Load with 'unet' adapter name
            pipe.load_lora_weights(args.lora_weights, adapter_name="unet")
            print(f"Loaded LoRA weights from {args.lora_weights}")
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            return
    else:
        print("Warning: No LoRA weights loaded")

    prompt = f"(skin lesion, {args.target_label})"
    print(f"Generating images with prompt: {prompt}")

    for img_path in tqdm(image_files, desc="Generating"):
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
