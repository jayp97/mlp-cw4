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
from diffusers import StableDiffusionPipeline, DDIMScheduler
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


def inject_lora_into_unet(unet, lora_weights, scale=1.0):
    """
    Directly inject LoRA weights into UNet model by modifying forward methods.
    This approach bypasses the need for specific attention processor modules.
    """
    print(f"Applying LoRA weights with scale {scale}...")

    # Parse module name format from the LoRA weights
    modules_to_modify = {}
    for key in lora_weights:
        parts = key.split("|")
        if len(parts) != 2:
            continue

        module_name = parts[0]
        param_name = parts[1]

        if module_name not in modules_to_modify:
            modules_to_modify[module_name] = {}

        modules_to_modify[module_name][param_name] = lora_weights[key]

    print(f"Found {len(modules_to_modify)} modules to modify")

    # Modify each module with LoRA weights
    modules_modified = 0
    for module_path, module_weights in modules_to_modify.items():
        # Get the module from the UNet
        module = unet
        for name in module_path.split("."):
            if not hasattr(module, name):
                print(f"Warning: Could not find module at path {module_path}")
                break
            module = getattr(module, name)
        else:
            # Successfully accessed the module, now modify its forward method
            if "up" in module_weights and "down" in module_weights:
                # Get original forward method
                orig_forward = module.forward

                # Apply LoRA weights through a new forward method
                def make_lora_forward(orig_func, up_weight, down_weight, scale):
                    def lora_forward(x, *args, **kwargs):
                        # Call original forward
                        result = orig_func(x, *args, **kwargs)

                        # Apply LoRA: (x @ down_weight.T) @ up_weight.T
                        # First reshape x for matrix multiplication
                        x_reshaped = x.reshape(-1, x.shape[-1])

                        # Apply down projection
                        down_projection = torch.matmul(
                            x_reshaped, down_weight.t().to(x.device)
                        )

                        # Apply up projection
                        up_projection = torch.matmul(
                            down_projection, up_weight.t().to(x.device)
                        )

                        # Reshape back to original shape and scale
                        lora_output = up_projection.reshape(result.shape) * scale

                        # Add LoRA output to the original output
                        return result + lora_output

                    return lora_forward

                # Replace the forward method
                up_weight = module_weights["up.weight"]
                down_weight = module_weights["down.weight"]
                module.forward = make_lora_forward(
                    orig_forward, up_weight, down_weight, scale
                )
                modules_modified += 1

    print(f"Successfully modified {modules_modified} modules")
    return modules_modified > 0


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

    # 1) Load the Stable Diffusion pipeline in FP16.
    print(f"Loading base model: {args.pretrained_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None,  # Disable safety checker for faster loading
    ).to("cuda")

    # Use DDIM scheduler for better quality
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 2) Load and apply LoRA weights
    if os.path.exists(args.lora_weights):
        print(f"Loading LoRA weights from {args.lora_weights}...")
        try:
            lora_state_dict = torch.load(args.lora_weights, map_location="cpu")
            print(f"Loaded {len(lora_state_dict)} LoRA parameter entries")

            # Look for adapter_config.json in the same directory
            config_path = os.path.join(
                os.path.dirname(args.lora_weights), "adapter_config.json"
            )
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    adapter_config = json.load(f)
                    print(f"Loaded adapter config: {adapter_config}")

            # Apply LoRA weights by directly modifying modules
            success = inject_lora_into_unet(
                pipe.unet, lora_state_dict, scale=args.lora_scale
            )
            if not success:
                print("WARNING: Failed to apply LoRA weights correctly")
                print("Continuing with base model only")
        except Exception as e:
            print(f"Error applying LoRA weights: {e}")
            print("Continuing with base model only")
    else:
        print(f"WARNING: LoRA weights not found at {args.lora_weights}")
        print("Proceeding with base model only. Results will not have the fine-tuning!")

    # 3) Setup a random generator for reproducibility.
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # 4) Generate and save images.
    print(f"Generating {args.num_images} images...")
    for i in tqdm(range(args.num_images)):
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

    print(f"All images generated successfully in {args.output_dir}")


if __name__ == "__main__":
    main()
