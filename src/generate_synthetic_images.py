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
from diffusers.models.attention_processor import AttnProcessor
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


class CustomLoRAAttnProcessor(AttnProcessor):
    """Custom LoRA attention processor that directly applies pre-loaded weights."""

    def __init__(
        self,
        hidden_size,
        cross_attention_dim=None,
        rank=4,
        module_name="",
        lora_weights=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else hidden_size
        )
        self.rank = rank
        self.module_name = module_name
        self.lora_weights = lora_weights
        self.scale = 1.0

        # Get the relevant parameters for this module from the LoRA weights
        if lora_weights is not None:
            # Look for keys that match this module
            for key in lora_weights:
                if self.module_name in key and "|up." in key:
                    self.up_weight = lora_weights[key]
                elif self.module_name in key and "|down." in key:
                    self.down_weight = lora_weights[key]

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Apply LoRA attention if we have the weights
        if hasattr(self, "up_weight") and hasattr(self, "down_weight"):
            # Reshape for down projection
            down_hidden_states = hidden_states.reshape(
                hidden_states.shape[0], hidden_states.shape[1], -1
            )
            # Apply down projection
            down_proj = torch.matmul(
                down_hidden_states, self.down_weight.t().to(hidden_states.device)
            )
            # Apply up projection
            up_proj = torch.matmul(
                down_proj, self.up_weight.t().to(hidden_states.device)
            )
            # Reshape to match original shape
            lora_output = up_proj.reshape(hidden_states.shape) * self.scale
            # Add LoRA output to query
            query = query + lora_output

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0
        )
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def apply_lora(unet, lora_path, scale=1.0):
    """Apply LoRA weights to the UNet model."""
    print(f"Loading LoRA weights from {lora_path}...")

    if lora_path.endswith(".bin"):
        # Load PyTorch weights directly
        lora_state_dict = torch.load(lora_path, map_location="cpu")
    elif lora_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file

            lora_state_dict = load_file(lora_path)
        except:
            print("Failed to load safetensors, falling back to PyTorch format")
            lora_state_dict = torch.load(
                lora_path.replace(".safetensors", ".bin"), map_location="cpu"
            )
    else:
        raise ValueError(f"Unsupported weight file format: {lora_path}")

    print(f"Loaded {len(lora_state_dict)} LoRA parameter entries")

    # Look for adapter_config.json in the same directory
    config_path = os.path.join(os.path.dirname(lora_path), "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            adapter_config = json.load(f)
            rank = adapter_config.get("lora_rank", 4)
            print(f"Using rank {rank} from adapter config")
    else:
        # Default rank
        rank = 4

    # Apply LoRA weights to each attention module
    lora_attn_procs = {}

    for name, module in unet.named_modules():
        if "attn1" in name or "attn2" in name:
            if name.endswith("processor"):
                continue

            # Configure correct hidden sizes
            if "attn1" in name:  # self-attention
                hidden_size = module.to_q.in_features
                cross_attention_dim = None
            else:  # cross-attention
                hidden_size = module.to_q.in_features
                cross_attention_dim = module.to_k.in_features

            # Create custom processor with LoRA weights
            lora_processor = CustomLoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
                module_name=name,
                lora_weights=lora_state_dict,
            )
            lora_processor.scale = scale

            # Store for later application
            lora_attn_procs[name] = lora_processor

    # Apply all processors at once
    unet.set_attn_processor(lora_attn_procs)
    print(f"Applied {len(lora_attn_procs)} LoRA attention processors")

    return unet


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

    # 2) Apply LoRA weights
    if os.path.exists(args.lora_weights):
        apply_lora(pipe.unet, args.lora_weights, scale=args.lora_scale)
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
