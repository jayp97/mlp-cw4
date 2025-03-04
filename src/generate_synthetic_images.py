#!/usr/bin/env python
# coding=utf-8
"""
generate_synthetic_images.py

Generates synthetic skin lesion images for a given target label using the fine-tuned
Stable Diffusion model (with LoRA) and the learned textual inversion (or fine-tuned) model.
At generation time you specify the target label (e.g., "Dermatofibroma") and the script
filters the metadata CSV to select initial images of that label.

Example Usage:
python generate_synthetic_images.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --metadata_file="mlp-cw4/data/raw/HAM10000_metadata.csv" \
  --target_label="Dermatofibroma" \
  --train_data_dir="mlp-cw4/data/processed_sd/images" \
  --output_dir="mlp-cw4/data/synthetic/images_dermatofibroma" \
  --guidance_scale=7.5 \
  --num_inference_steps=50 \
  --strength=0.8 \
  --num_images_per_prompt=5 \
  --device="cuda"
"""
import argparse
import os
import torch
from PIL import Image
from tqdm.auto import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.loaders import LoraLoaderMixin
from safetensors.torch import safe_open
import pandas as pd

# Same LABEL_MAP used during training.
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
        required=True,
        help="The full target label for generation (e.g., 'Dermatofibroma')",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory containing all processed images",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--num_images_per_prompt", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata and filter for rows whose full label matches the target label.
    metadata = pd.read_csv(args.metadata_file)
    # For each row, compute full label using LABEL_MAP.
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
        print(f"No images found for target label {args.target_label}")
        return

    # 1) Load pipeline.
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to(args.device)

    # 2) Load fine-tuned LoRA model.
    # (We assume the fine-tuned model from finetune_stable_diffusion.py was saved with .save_pretrained)
    pipe.unet.load_state_dict(
        torch.load(
            os.path.join(args.output_dir, "pytorch_model.bin"), map_location="cpu"
        ),
        strict=False,
    )
    # Alternatively, if using PEFT/LoRA weight file:
    LoraLoaderMixin.load_lora_into_unet(
        pipe.unet,
        os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"),
        alpha=1.0,
    )
    pipe.unet.to(args.device, dtype=torch.float16)

    # 3) Generate synthetic images.
    # Use a prompt based on the target label.
    prompt = f"(skin lesion, {args.target_label})"
    for img_path in tqdm(image_files, desc="Generating"):
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
    print(f"All done! Synthetic images are in {args.output_dir}")


if __name__ == "__main__":
    main()
