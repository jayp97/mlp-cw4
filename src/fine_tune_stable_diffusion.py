#!/usr/bin/env python
# coding=utf-8
"""
finetune_stable_diffusion.py

This script fine-tunes a Stable Diffusion model using LoRA on the entire dataset.
It loads all processed images (all labels) from a single folder and uses the metadata CSV
to automatically construct a prompt for each image in the format:
    (skin lesion, {full_label})
where full_label is computed from the dx field via a built-in mapping (LABEL_MAP).

No manual input of a lesion code or placeholder token is needed.

The fine-tuned model (with LoRA adapters) is saved in --output_dir.

Example Usage:
python finetune_stable_diffusion.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --metadata_file="mlp-cw4/data/raw/HAM10000_metadata.csv" \
  --train_data_dir="mlp-cw4/data/processed_sd/images" \
  --resolution=512 \
  --learning_rate=5e-5 \
  --max_train_steps=1000 \
  --batch_size=1 \
  --lora_rank=4 \
  --output_dir="mlp-cw4/models/stable_diffusion_lora"
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
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import save_file

# Mapping from short lesion codes to full names.
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
        help="CSV file containing metadata (e.g., HAM10000_metadata.csv).",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory containing all processed 512x512 images (no subfolders).",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank dimension.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="./output/stable_diffusion_lora"
    )
    return parser.parse_args()


class AllLesionDataset(Dataset):
    """
    Loads all images from data_root using the metadata CSV.
    For each image, the dx field is mapped (using LABEL_MAP) to a full label,
    and the prompt is constructed as: (skin lesion, {full_label})
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
        full_label = LABEL_MAP.get(dx, dx)
        prompt = f"(skin lesion, {full_label})"
        # Tokenize prompt for use in conditioning.
        tokenized_text = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        image_path = os.path.join(self.data_root, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), resample=Image.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        return {"pixel_values": image_tensor, "input_ids": tokenized_text}


def main():
    args = parse_args()
    accelerator = Accelerator()
    set_seed(args.seed)

    # Load tokenizer and text encoder.
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    # (In this full-dataset fine-tuning we are not adding a new token.)
    # Freeze the text encoder.
    for param in text_encoder.parameters():
        param.requires_grad = False

    # Load VAE and UNet.
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

    # Here we assume LoRA fine-tuning.
    # We replace target modules in UNet with LoRA versions.
    # (Assuming you have a LoRA configuration available in your diffusers installation.)
    # For simplicity, here we assume that you call get_peft_model (if using PEFT) or manually replace attention processors.
    # For our example, we simply replace each UNet attn processor with a LoRA-attention processor.
    from diffusers.models.attention_processor import LoRAAttnProcessor

    for name, _ in unet.attn_processors.items():
        unet.attn_processors[name] = LoRAAttnProcessor()

    # Create the dataset that uses all images and auto-generates prompts.
    dataset = AllLesionDataset(
        data_root=args.train_data_dir,
        metadata_file=args.metadata_file,
        resolution=args.resolution,
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    # Fine-tuning: we update the LoRA parameters in the UNet only.
    # Gather LoRA parameters.
    lora_params = []
    for _, module in unet.attn_processors.items():
        for pname, param in module.named_parameters():
            if "lora_" in pname:
                param.requires_grad = True
                lora_params.append(param)
    print(f"Found {len(lora_params)} LoRA parameters to train.")

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
            latents = vae.encode(pixel_values.half()).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            enc_out = text_encoder(input_ids=input_ids)
            encoder_hidden_states = enc_out[0].half()
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
        # Save the fine-tuned UNet with LoRA adapters.
        unet.save_pretrained(args.output_dir)
        print(f"Saved fine-tuned LoRA weights to {args.output_dir}")


if __name__ == "__main__":
    main()
