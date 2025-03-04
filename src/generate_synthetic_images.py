#!/usr/bin/env python
# coding=utf-8
"""
generate_synthetic_images.py

Generates synthetic skin lesion images for a given lesion label using the fine-tuned
Stable Diffusion model (with LoRA) and the learned textual inversion embeddings.
Images are selected by filtering a metadata CSV file.

Example Usage:
python generate_synthetic_images.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --embed_path="mlp-cw4/models/txt_inversion_dermatofibroma/learned_embeds.safetensors" \
  --lora_weights="mlp-cw4/models/stable_diffusion_lora/pytorch_lora_weights.safetensors" \
  --metadata_file="mlp-cw4/data/raw/HAM10000_metadata.csv" \
  --lesion_code="df" \
  --train_data_dir="mlp-cw4/data/processed_sd/images" \
  --output_dir="mlp-cw4/data/synthetic/images_dermatofibroma" \
  --token="<derm_token>" \
  --guidance_scale=7.5 \
  --num_inference_steps=50 \
  --strength=0.8 \
  --num_images_per_prompt=5
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--embed_path", type=str, required=True)
    parser.add_argument("--lora_weights", type=str, required=True)
    parser.add_argument(
        "--metadata_file", type=str, required=True, help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--lesion_code", type=str, required=True, help="Lesion code (e.g., 'df')"
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory containing all processed images",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="The placeholder token (e.g., <derm_token>)",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--num_images_per_prompt", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata and filter for the given lesion code
    metadata = pd.read_csv(args.metadata_file)
    metadata = metadata[metadata["dx"].str.lower() == args.lesion_code.lower()]
    image_ids = metadata["image_id"].tolist()
    image_files = [
        os.path.join(args.train_data_dir, f"{img_id}.jpg")
        for img_id in image_ids
        if os.path.exists(os.path.join(args.train_data_dir, f"{img_id}.jpg"))
    ]

    # 1) Load pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to(args.device)

    # 2) Load learned embeddings and add token
    learned_embeds_dict = {}
    with safe_open(args.embed_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            learned_embeds_dict[key] = f.get_tensor(key)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    for token_name, token_embed in learned_embeds_dict.items():
        num_added = tokenizer.add_tokens(token_name)
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[token_id] = token_embed

    # 3) Load LoRA weights into UNet
    LoraLoaderMixin.load_lora_into_unet(pipe.unet, args.lora_weights, alpha=1.0)
    pipe.unet.to(args.device, dtype=torch.float16)

    # 4) Generate synthetic images using img2img
    for img_path in tqdm(image_files, desc="Generating"):
        try:
            init_image = (
                Image.open(img_path).convert("RGB").resize((512, 512), Image.BICUBIC)
            )
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        prompt = f"An image of {args.token}"
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
