#!/usr/bin/env python
# coding=utf-8
"""
generate_synthetic_images.py

Use your newly learned token and LoRA weights to generate synthetic images
for the lesion. This script uses image-to-image (img2img) by default, since
providing an existing real lesion image often yields better results.

Example Usage:
python generate_synthetic_images.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --embed_path="mlp-cw4/models/txt_inversion_dermatofibroma/learned_embeds.safetensors" \
  --lora_weights="mlp-cw4/models/stable_diffusion_lora/pytorch_lora_weights.safetensors" \
  --input_image_dir="mlp-cw4/data/processed_sd/images/dermatofibroma" \
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--embed_path", type=str, required=True)
    parser.add_argument("--lora_weights", type=str, required=True)
    parser.add_argument(
        "--input_image_dir",
        type=str,
        required=True,
        help="Directory with real 512×512 images to use as init images.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="The placeholder token you learned, e.g. <derm_token>.",
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

    # 1) Load pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to(args.device)

    # 2) Load textual inversion embeddings
    learned_embeds_dict = {}
    with safe_open(args.embed_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            learned_embeds_dict[key] = f.get_tensor(key)

    # Add token to pipeline
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    for token_name, token_embed in learned_embeds_dict.items():
        num_added = tokenizer.add_tokens(token_name)
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[token_id] = token_embed

    # 3) Load LoRA weights
    LoraLoaderMixin.load_lora_into_unet(pipe.unet, args.lora_weights, alpha=1.0)
    pipe.unet.to(args.device, dtype=torch.float16)

    # 4) For each image in input_image_dir, run img2img with your token
    init_files = [
        f
        for f in os.listdir(args.input_image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not init_files:
        print("No images found in input_image_dir.")
        return

    for fname in tqdm(init_files, desc="Generating"):
        init_path = os.path.join(args.input_image_dir, fname)
        init_image = (
            Image.open(init_path).convert("RGB").resize((512, 512), Image.BICUBIC)
        )

        base_name = os.path.splitext(fname)[0]
        # Example prompt
        prompt = f"An image of {args.token}"
        for i in range(args.num_images_per_prompt):
            out = pipe(
                prompt=prompt,
                image=init_image,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
            )
            gen_image = out.images[0]
            out_name = f"{base_name}_synthetic_{i}.png"
            gen_image.save(os.path.join(args.output_dir, out_name))

    print(f"All done! Synthetic images are in {args.output_dir}")


if __name__ == "__main__":
    main()
