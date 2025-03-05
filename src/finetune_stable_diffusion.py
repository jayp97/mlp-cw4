#!/usr/bin/env python
# coding=utf-8
"""
finetune_stable_diffusion.py

LoRA fine-tuning on the entire dataset (all labels). We do not rely on AttnProcessor
having parameters. Instead, we manually insert LoRA into the UNet cross-attn sub-layers
(e.g. to_q, to_k, to_v, to_out).

No manual lesion code or placeholder token is needed. The dataset uses the metadata CSV
to build prompts: (skin lesion, {full_label}).

Requires:
  - diffusers >= 0.14
  - accelerate
  - safetensors
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer


# ---------- LO RA Implementation -----------
# We define a simple LoRA injection approach for cross-attention sub-layers.
def lora_layer_hook(module, input, output):
    # If we inserted a lora_down and lora_up in module, we add them here.
    if hasattr(module, "lora_down") and hasattr(module, "lora_up"):
        # lora_scale can be an attribute if we want to scale the LoRA output
        lora_scale = getattr(module, "lora_scale", 1.0)
        # The output is basically: output + (lora_up(lora_down(input)) * alpha)
        return output + (module.lora_up(module.lora_down(input[0]))) * lora_scale
    return output


def create_lora_weights_for_attn_module(module, lora_rank=4, lora_alpha=1.0):
    """
    Insert LoRA down/up submodules into the linear layers of a cross-attn block:
      to_q, to_k, to_v, to_out[0], etc.
    This is a simplified approach.
    """

    # We'll define a small helper:
    def add_lora_to_linear(linear_layer):
        # Create lora_down and lora_up
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        # lora_down: rank x in_features
        lora_down = torch.nn.Linear(in_features, lora_rank, bias=False)
        # lora_up: out_features x rank
        lora_up = torch.nn.Linear(lora_rank, out_features, bias=False)
        # Initialize
        torch.nn.init.zeros_(lora_down.weight)
        torch.nn.init.zeros_(lora_up.weight)

        # Set them as attributes
        linear_layer.lora_down = lora_down
        linear_layer.lora_up = lora_up
        linear_layer.lora_scale = lora_alpha
        # Overwrite forward hook
        # Instead of rewriting forward, we can do a forward_pre_hook or forward_hook
        # We'll do a forward_pre_hook so we can see input
        if not hasattr(linear_layer, "_lora_hooked"):
            linear_layer.register_forward_hook(lora_layer_hook)
            linear_layer._lora_hooked = True

    # Now let's see if the module is an attention sub-layer with to_q, to_k, to_v, etc.
    if hasattr(module, "to_q") and isinstance(module.to_q, torch.nn.Linear):
        add_lora_to_linear(module.to_q)
    if hasattr(module, "to_k") and isinstance(module.to_k, torch.nn.Linear):
        add_lora_to_linear(module.to_k)
    if hasattr(module, "to_v") and isinstance(module.to_v, torch.nn.Linear):
        add_lora_to_linear(module.to_v)
    if hasattr(module, "to_out") and isinstance(module.to_out[0], torch.nn.Linear):
        add_lora_to_linear(module.to_out[0])


# Mapping from short lesion codes to full names
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
        "--metadata_file",
        type=str,
        required=True,
        help="CSV file containing metadata (e.g. HAM10000_metadata.csv).",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Folder with 512×512 images (all labels).",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="./output/stable_diffusion_lora"
    )
    return parser.parse_args()


class AllLesionDataset(Dataset):
    """
    For each row in the metadata, we load image and build prompt: (skin lesion, {full_label})
    """

    def __init__(self, data_root, metadata_file, resolution, tokenizer):
        self.data_root = data_root
        self.metadata = pd.read_csv(metadata_file)
        self.resolution = resolution
        self.tokenizer = tokenizer
        self._length = len(self.metadata)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row["image_id"]
        dx = row["dx"].lower()
        full_label = LABEL_MAP.get(dx, dx)  # default to dx if not in map
        prompt = f"(skin lesion, {full_label})"

        # Tokenize
        tokenized_text = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        # Load image
        image_path = os.path.join(self.data_root, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.BICUBIC)
        image_np = np.array(image).astype(np.uint8)
        image_np = (image_np / 127.5 - 1.0).astype(np.float32)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        return {"pixel_values": image_tensor, "input_ids": tokenized_text}


def main():
    args = parse_args()
    accelerator = Accelerator()
    set_seed(args.seed)

    # 1) Load CLIP tokenizer & text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    # Freeze text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False

    # 2) Load VAE & UNet
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    vae.to(accelerator.device, dtype=torch.float16)
    unet.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    # 3) Insert LoRA submodules into cross-attention layers
    # We'll iterate over all UNet submodules, find cross-attn blocks, and insert LoRA
    for name, submodule in unet.named_modules():
        # The cross-attn blocks typically have "Transformer2DModel" in their path or "Attention".
        # But the best approach is to detect that submodule has to_q, to_k, to_v attributes:
        if (
            hasattr(submodule, "to_q")
            and hasattr(submodule, "to_k")
            and hasattr(submodule, "to_v")
        ):
            # We'll add LoRA to each of these linear layers
            # Example function create_lora_weights_for_attn_module could be used
            # But let's inline it for clarity:
            def add_lora_to_linear(linear_layer, rank=4, alpha=1.0):
                # Create two linear layers of rank
                in_features = linear_layer.in_features
                out_features = linear_layer.out_features
                # lora_down
                lora_down = torch.nn.Linear(in_features, rank, bias=False)
                torch.nn.init.zeros_(lora_down.weight)
                # lora_up
                lora_up = torch.nn.Linear(rank, out_features, bias=False)
                torch.nn.init.zeros_(lora_up.weight)

                linear_layer.lora_down = lora_down
                linear_layer.lora_up = lora_up
                linear_layer.lora_alpha = alpha

                # We'll attach a forward_pre_hook
                def lora_forward_pre_hook(module, input):
                    # input is a tuple of (x,)
                    x = input[0]
                    # normal forward
                    out = module.original_forward(x)
                    # lora forward
                    out_lora = module.lora_up(module.lora_down(x)) * module.lora_alpha
                    return (out + out_lora,)

                if not hasattr(linear_layer, "original_forward"):
                    linear_layer.original_forward = linear_layer.forward

                    def new_forward(x):
                        # We'll rely on the forward_pre_hook above
                        return x

                    linear_layer.forward = new_forward

                # Insert the forward_pre_hook if not inserted
                found_hook = getattr(linear_layer, "_lora_hook_registered", False)
                if not found_hook:
                    linear_layer.register_forward_pre_hook(lora_forward_pre_hook)
                    linear_layer._lora_hook_registered = True

            # Now actually patch submodule's layers
            # submodule.to_q etc. are typically nn.Linear
            add_lora_to_linear(submodule.to_q, rank=args.lora_rank, alpha=1.0)
            add_lora_to_linear(submodule.to_k, rank=args.lora_rank, alpha=1.0)
            add_lora_to_linear(submodule.to_v, rank=args.lora_rank, alpha=1.0)
            if hasattr(submodule, "to_out") and isinstance(
                submodule.to_out, torch.nn.ModuleList
            ):
                # Typically submodule.to_out is [linear, dropout].
                # We'll patch submodule.to_out[0] if it is linear.
                if len(submodule.to_out) > 0 and isinstance(
                    submodule.to_out[0], torch.nn.Linear
                ):
                    add_lora_to_linear(
                        submodule.to_out[0], rank=args.lora_rank, alpha=1.0
                    )
            elif hasattr(submodule, "to_out") and isinstance(
                submodule.to_out, torch.nn.Linear
            ):
                add_lora_to_linear(submodule.to_out, rank=args.lora_rank, alpha=1.0)

    # 4) Create dataset
    dataset = AllLesionDataset(
        data_root=args.train_data_dir,
        metadata_file=args.metadata_file,
        resolution=args.resolution,
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # 5) Create noise scheduler & gather LoRA parameters
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # gather LoRA parameters
    lora_params = []
    for _, subm in unet.named_modules():
        # if subm has lora_down/lora_up, gather them
        if hasattr(subm, "lora_down") and hasattr(subm, "lora_up"):
            lora_params.append(subm.lora_down.weight)
            lora_params.append(subm.lora_up.weight)
    if len(lora_params) == 0:
        raise ValueError(
            "No LoRA parameters found. Possibly the code didn't find cross-attn layers or your diffusers version is too new/old."
        )
    print(f"Found {len(lora_params)} LoRA param tensors in total.")

    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)
    max_train_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    unet.train()
    vae.eval()
    text_encoder.eval()

    global_step = 0
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )

    for step in range(max_train_steps):
        batch = next(iter(dataloader))
        with accelerator.accumulate(unet):
            pixel_values = batch["pixel_values"].to(
                accelerator.device, dtype=torch.float32
            )
            input_ids = batch["input_ids"].to(accelerator.device)

            # Encode images to latents
            latents = vae.encode(pixel_values.half()).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Random timesteps
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                enc_out = text_encoder(input_ids=input_ids)
                encoder_hidden_states = enc_out[0].half()

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})
        global_step += 1
        if global_step >= max_train_steps:
            break

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # We save the entire UNet with patched LoRA sub-layers
        unet.save_pretrained(args.output_dir)
        print(f"LoRA fine-tuning complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
