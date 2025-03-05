#!/usr/bin/env python
# coding=utf-8

"""
train_lora.py

Trains LoRA on a Stable Diffusion model for the entire HAM10000 dataset.
Each image (from a single folder) is paired with a prompt derived from its lesion label
in HAM10000_metadata.csv. Only the LoRA parameters injected into the UNet's cross-attention
layers are updated while the rest of the model remains frozen.

Example usage:
---------------
accelerate launch --mixed_precision=fp16 src/train_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --metadata_file="data/raw/HAM10000_metadata.csv" \
  --train_data_dir="data/processed_sd/images" \
  --output_dir="models/stable_diffusion_lora" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=1000 \
  --learning_rate=1e-4 \
  --seed=42 \
  --rank=4
"""

import argparse
import os
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.models.lora import LoRALinearLayer
from transformers import CLIPTokenizer, CLIPTextModel

logger = logging.getLogger(__name__)

# Mapping from HAM10000 dx codes to full lesion labels.
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
    parser = argparse.ArgumentParser(
        description="Train LoRA for the entire HAM10000 dataset."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Stable Diffusion model name or path (e.g., runwayml/stable-diffusion-v1-5).",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to HAM10000_metadata.csv (with columns [image_id, dx, ...]).",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory containing processed SD images (512x512), named <image_id>.jpg.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-output",
        help="Directory where LoRA weights will be saved (pytorch_lora_weights.safetensors).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training image resolution, e.g. 512.",
    )
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="LoRA rank dimension for each cross-attention projection.",
    )
    return parser.parse_args()


class HAMDataset(Dataset):
    """
    Loads images from train_data_dir using metadata from HAM10000_metadata.csv.
    For each image, constructs a prompt based on the dx code.
    For example, if dx == "df" then the prompt is:
       "A photo of a Dermatofibroma lesion"
    """

    def __init__(self, metadata_file, data_root, resolution=512):
        super().__init__()
        self.metadata = pd.read_csv(metadata_file)
        self.data_root = data_root
        self.resolution = resolution
        self.samples = []

        for _, row in self.metadata.iterrows():
            image_id = str(row["image_id"]).strip()
            dx_code = str(row["dx"]).lower().strip()
            if dx_code not in LABEL_MAP:
                continue
            label_text = LABEL_MAP[dx_code]
            prompt = f"A photo of a {label_text} lesion"
            image_path = os.path.join(data_root, f"{image_id}.jpg")
            if os.path.isfile(image_path):
                self.samples.append((image_path, prompt))
            else:
                logger.warning(f"Missing file: {image_path}; skipping.")
        self._length = len(self.samples)
        logger.info(f"Dataset loaded {self._length} valid images from {data_root}.")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        image_path, prompt = self.samples[idx]
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((self.resolution, self.resolution), resample=Image.BICUBIC)
        arr = np.array(img, dtype=np.float32)
        # Scale pixel values from [0,255] to [-1,1]
        arr = (arr / 127.5) - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return {"pixel_values": tensor, "prompt": prompt}


def inject_lora(linear_layer, rank):
    """
    Injects a LoRA module into a given linear layer by:
      1. Attaching a new LoRALinearLayer as an attribute.
      2. Overriding the forward method so that it returns:
             original_forward(x) + lora_layer(x)
    """
    if not hasattr(linear_layer, "lora_layer"):
        linear_layer.lora_layer = LoRALinearLayer(
            linear_layer.in_features, linear_layer.out_features, rank=rank
        )
    # Save original forward if not already done.
    if not hasattr(linear_layer, "orig_forward"):
        linear_layer.orig_forward = linear_layer.forward

        def new_forward(x):
            return linear_layer.orig_forward(x) + linear_layer.lora_layer(x)

        linear_layer.forward = new_forward


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(args)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="no",  # Changed from auto (default) to "no" to avoid type mismatches
    )
    set_seed(args.seed)

    # 1) Load base Stable Diffusion components.
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # 2) Freeze all parameters.
    for p in unet.parameters():
        p.requires_grad = False
    for p in vae.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    # 3) Create dataset and dataloader.
    dataset = HAMDataset(
        metadata_file=args.metadata_file,
        data_root=args.train_data_dir,
        resolution=args.resolution,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )

    # 4) Inject LoRA into UNet cross-attention linear layers.
    for name, module in unet.named_modules():
        # Look for modules that have attributes "to_q", "to_k", and "to_v".
        if (
            hasattr(module, "to_q")
            and hasattr(module, "to_k")
            and hasattr(module, "to_v")
        ):
            inject_lora(module.to_q, args.rank)
            inject_lora(module.to_k, args.rank)
            inject_lora(module.to_v, args.rank)
            if hasattr(module, "to_out") and isinstance(
                module.to_out, torch.nn.ModuleList
            ):
                if len(module.to_out) > 0 and hasattr(module.to_out[0], "in_features"):
                    inject_lora(module.to_out[0], args.rank)

    # 5) Gather LoRA parameters.
    lora_params = []
    for _, submodule in unet.named_modules():
        if hasattr(submodule, "lora_layer"):
            for param in submodule.lora_layer.parameters():
                param.requires_grad = True
                lora_params.append(param)

    # 6) Create optimizer for LoRA parameters.
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)

    # 7) Create a constant learning rate scheduler.
    max_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps,
    )

    # 8) Prepare UNet, optimizer, and dataloader with Accelerator.
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    unet.train()

    # 9) Move VAE and text_encoder to device.
    device = accelerator.device
    # Don't force float16 for these models
    vae.to(device)
    text_encoder.to(device)

    global_step = 0
    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    for _ in range(max_steps):
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Make sure pixel_values are in full precision (float32)
        pixel_values = batch["pixel_values"].to(device)
        prompts = batch["prompt"]

        with accelerator.accumulate(unet):
            # i) Convert images to latents using VAE.
            with torch.no_grad():
                # Keep in full precision
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            # ii) Sample noise and timesteps.
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # iii) Tokenize prompts.
            token_out = tokenizer(
                list(prompts),
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            token_out = {k: v.to(device) for k, v in token_out.items()}

            # iv) Forward text encoder.
            with torch.no_grad():
                text_out = text_encoder(**token_out)
                encoder_hidden_states = text_out[0]

            # v) Forward pass through UNet.
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # vi) Compute MSE loss.
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})
        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)

        # Create a dict for actual tensor weights
        lora_state_dict = {}

        # Add the LoRA weights
        for name, module in unwrapped_unet.named_modules():
            if hasattr(module, "lora_layer"):
                # We need the correct format for cross attention layers
                if "to_q" in name:
                    layer_type = "q_proj"
                elif "to_k" in name:
                    layer_type = "k_proj"
                elif "to_v" in name:
                    layer_type = "v_proj"
                elif "to_out.0" in name:
                    layer_type = "out_proj"
                else:
                    continue

                # Get the LoRA up and down weights
                up_weight = module.lora_layer.up.weight.data.cpu()
                down_weight = module.lora_layer.down.weight.data.cpu()

                # Use naming convention diffusers expects
                clean_name = name.split(".")[-1]  # Get last part of module name
                adapter_name = "default"
                rank = args.rank

                # Format keys as diffusers expects
                up_key = f"{adapter_name}.{clean_name}.{layer_type}.lora_up.weight"
                down_key = f"{adapter_name}.{clean_name}.{layer_type}.lora_down.weight"
                alpha_key = f"{adapter_name}.{clean_name}.{layer_type}.alpha"

                # Store the weights
                lora_state_dict[up_key] = up_weight
                lora_state_dict[down_key] = down_weight
                # Store alpha as a tensor
                lora_state_dict[alpha_key] = torch.tensor(float(rank))

        from safetensors.torch import save_file
        import json

        # Create metadata as string that safetensors can handle
        metadata = {
            "format": "pt",
            "alpha": str(args.rank),
            "rank": str(args.rank),
        }

        # Create README with usage instructions
        readme_content = """# LoRA weights for Stable Diffusion

These weights were trained on HAM10000 dermatology dataset.

## Usage
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
).to("cuda")

# For newer diffusers versions
pipe.load_lora_weights("./", weight_name="pytorch_lora_weights.safetensors")

# Or load directly
pipe.unet.load_attn_procs("./pytorch_lora_weights.safetensors")
```
"""
        with open(os.path.join(args.output_dir, "README.md"), "w") as f:
            f.write(readme_content)

        # Save the weights with metadata
        out_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        save_file(lora_state_dict, out_path, metadata=metadata)
        logger.info(f"LoRA training complete! Weights saved to {out_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
