#!/usr/bin/env python
# coding=utf-8

"""
generate_synthetic_images.py

Loads a base Stable Diffusion model along with LoRA weights trained on the HAM10000 dataset.
A lesion type is specified via --lesion_code (e.g., "df" for Dermatofibroma) and a prompt 
is constructed accordingly. Synthetic images are generated and saved to the specified output directory.

Example usage:
---------------
python src/generate_synthetic_images.py \
  --pretrained_model="runwayml/stable-diffusion-v1-5" \
  --lora_weights="models/stable_diffusion_lora/pytorch_lora_weights.bin" \
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
import json
import re
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm
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
        help="Path to LoRA weights (pytorch_lora_weights.bin) from training.",
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
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="Scaling factor for LoRA weights",
    )
    return parser.parse_args()


class LoRAModuleInjector:
    """Injects LoRA layers into existing linear modules."""

    def __init__(self, linear_module, lora_up, lora_down, scale=1.0):
        self.linear_module = linear_module
        self.lora_up = lora_up
        self.lora_down = lora_down
        self.scale = scale

        # Save the original forward method
        self.original_forward = linear_module.forward

        # Replace the forward method with our custom one
        def lora_forward(x):
            # Original output
            original_output = self.original_forward(x)

            # LoRA path: x -> down -> up
            # Preserve original shape
            orig_shape = x.shape
            # Reshape for matrix multiplication if needed
            if len(x.shape) > 2:
                x_2d = x.reshape(-1, x.shape[-1])
            else:
                x_2d = x
            # Down projection
            down_output = torch.mm(x_2d, self.lora_down.T.to(x.device))
            # Up projection
            up_output = torch.mm(down_output, self.lora_up.T.to(x.device))
            # Reshape back
            if len(orig_shape) > 2:
                lora_output = up_output.reshape(
                    orig_shape[:-1] + (self.lora_up.shape[0],)
                )
            else:
                lora_output = up_output

            # Scale and add to original output
            scaled_output = lora_output * self.scale

            # Check if shapes match
            if original_output.shape == scaled_output.shape:
                return original_output + scaled_output
            else:
                print(
                    f"Shape mismatch: original {original_output.shape}, lora {scaled_output.shape}"
                )
                return original_output

        # Replace the forward method
        linear_module.forward = lora_forward


def apply_direct_lora(model, lora_state_dict, scale=1.0):
    """Apply LoRA weights directly to linear layers in the model."""
    print(f"Applying LoRA weights with scale {scale}...")

    # Track which modules we've modified
    modified_modules = set()

    # Find all Linear layers in the model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_up_key = None
            module_down_key = None

            # Try to match this module with LoRA weights by name pattern
            for key in lora_state_dict.keys():
                if "up" in key and name.replace(".", "_") in key:
                    module_up_key = key
                if "down" in key and name.replace(".", "_") in key:
                    module_down_key = key

            # If we found matching weights, apply LoRA
            if module_up_key and module_down_key:
                up_weight = lora_state_dict[module_up_key]
                down_weight = lora_state_dict[module_down_key]

                # Check if dimensions are compatible
                if (
                    up_weight.shape[0] == module.out_features
                    and down_weight.shape[1] == module.in_features
                ):
                    # Inject LoRA
                    LoRAModuleInjector(module, up_weight, down_weight, scale)
                    modified_modules.add(name)

    print(f"Successfully modified {len(modified_modules)} modules with direct LoRA")
    return len(modified_modules) > 0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.lesion_code not in LABEL_MAP:
        raise ValueError(
            f"Unknown lesion code '{args.lesion_code}'. Valid codes: {list(LABEL_MAP.keys())}"
        )
    label_text = LABEL_MAP[args.lesion_code]
    prompt = f"A photo of a {label_text} lesion"
    print(f"Using prompt: '{prompt}'")

    # 1) Load the Stable Diffusion pipeline
    print(f"Loading base model: {args.pretrained_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None,  # Disable safety checker for faster loading
    ).to("cuda")

    # 2) Load and apply LoRA weights
    if os.path.exists(args.lora_weights):
        print(f"Loading LoRA weights from {args.lora_weights}...")
        try:
            # Load with weights_only=True to avoid pickle warning
            lora_state_dict = torch.load(
                args.lora_weights, map_location="cpu", weights_only=True
            )
            print(f"Loaded {len(lora_state_dict)} LoRA parameter entries")

            # Look for adapter_config.json in the same directory
            config_path = os.path.join(
                os.path.dirname(args.lora_weights), "adapter_config.json"
            )
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    adapter_config = json.load(f)
                    print(f"Loaded adapter config: {adapter_config}")

            # Apply LoRA directly to linear layers
            success = apply_direct_lora(
                pipe.unet, lora_state_dict, scale=args.lora_scale
            )
            if not success:
                print("WARNING: Failed to apply LoRA weights correctly")
                print("Continuing with base model only")
        except Exception as e:
            print(f"Error applying LoRA weights: {e}")
            import traceback

            traceback.print_exc()
            print("Continuing with base model only")
    else:
        print(f"WARNING: LoRA weights not found at {args.lora_weights}")
        print("Proceeding with base model only. Results will not have the fine-tuning!")

    # 3) Setup a random generator for reproducibility.
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # 4) Generate and save images one by one.
    print(f"Generating {args.num_images} images...")
    for i in range(args.num_images):
        print(f"Generating image {i+1}/{args.num_images}...")

        # Generate image
        output = pipe(
            prompt=prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )

        # Save image
        image = output.images[0]
        filename = f"{args.lesion_code}_synthetic_{i:03d}.png"
        out_path = os.path.join(args.output_dir, filename)
        image.save(out_path)
        print(f"Saved {out_path}")

    print(f"All images generated successfully in {args.output_dir}")


if __name__ == "__main__":
    main()
