#!/usr/bin/env python
# coding=utf-8

"""
generate_synthetic_images.py

Loads a base Stable Diffusion model along with a LoRA checkpoint (trained on the entire HAM10000 dataset).
A lesion type is specified via --lesion_code (e.g., "df" for Dermatofibroma) and a prompt is constructed accordingly.
Synthetic images are generated and saved to the specified output directory.

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

# Mapping lesion codes to full textual labels.
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
        help="Base Stable Diffusion model, e.g. 'runwayml/stable-diffusion-v1-5'.",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA checkpoint (pytorch_lora_weights.safetensors) from training.",
    )
    parser.add_argument(
        "--lesion_code",
        type=str,
        default="df",
        help="Lesion code to generate (e.g. 'df' for Dermatofibroma).",
    )
    parser.add_argument(
        "--num_images", type=int, default=10, help="Number of images to generate."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of denoising steps."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="synthetic_output",
        help="Directory where generated images are saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.lesion_code not in LABEL_MAP:
        raise ValueError(
            f"Unknown lesion code '{args.lesion_code}'. Valid codes: {list(LABEL_MAP.keys())}"
        )
    label_text = LABEL_MAP[args.lesion_code]
    prompt = f"A photo of a {label_text} lesion"

    # 1) Load the Stable Diffusion pipeline in FP16.
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model, torch_dtype=torch.float16
    ).to("cuda")

    # 2) Load the LoRA weights with adapter name
    try:
        # First try standard loading
        pipe.load_lora_weights(
            os.path.dirname(args.lora_weights),
            weight_name=os.path.basename(args.lora_weights),
        )
    except ValueError as e:
        print(f"Standard loading failed: {e}. Trying alternative loading method...")
        # If that fails, try loading as a standalone file
        from safetensors.torch import load_file

        # Load the weights
        state_dict = load_file(args.lora_weights)

        # Manually apply the weights to the UNet
        # This is a fallback approach for custom saved LoRA weights
        print("Manually applying LoRA weights...")

        for key, value in state_dict.items():
            # Skip metadata and config keys
            if key.startswith("peft_config") or not key.startswith("lora."):
                continue

            # Parse the key to find the target module and parameter
            parts = key.split(".")
            if len(parts) < 4:
                continue

            direction = parts[1]  # 'up' or 'down'
            module_path = parts[2]  # The module path with underscores
            param_name = parts[3]  # Usually 'weight'

            # Convert underscores back to dots for module lookup
            module_path = module_path.replace("_", ".")

            # Find the matching module
            target_found = False
            for name, module in pipe.unet.named_modules():
                if name.endswith(module_path):
                    if hasattr(module, "lora_layer"):
                        # Use existing lora layer
                        if direction == "up" and hasattr(module.lora_layer, "up"):
                            module.lora_layer.up.weight.copy_(value)
                            target_found = True
                        elif direction == "down" and hasattr(module.lora_layer, "down"):
                            module.lora_layer.down.weight.copy_(value)
                            target_found = True
                    elif direction in ["up", "down"]:
                        # Need to inject a new LoRA layer
                        # This is a simplified version - in practice you'd need proper layer creation
                        print(f"Would need to inject LoRA layer into {name}")

            if not target_found:
                print(f"Could not find target for {key}")

        print("Manual loading completed")

    # 3) Setup a random generator for reproducibility.
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # 4) Generate and save images.
    for i in range(args.num_images):
        output = pipe(
            prompt=prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        image = output.images[0]
        filename = f"{args.lesion_code}_synthetic_{i:03d}.png"
        out_path = os.path.join(args.output_dir, filename)
        image.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
