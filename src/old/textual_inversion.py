#!/usr/bin/env python
# coding=utf-8
"""
textual_inversion.py

Learns a new textual token (e.g. <derm_token>) corresponding to a given skin lesion label.
Images are selected from a single folder by filtering a metadata CSV file (e.g., HAM10000_metadata.csv)
for rows with dx equal to the specified lesion code (e.g., "df" for Dermatofibroma).

Output:
  - learned_embeds.safetensors is saved in --output_dir

Example Usage:
python textual_inversion.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --tokenizer_name="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="mlp-cw4/data/processed_sd/images" \
  --metadata_file="mlp-cw4/data/raw/HAM10000_metadata.csv" \
  --lesion_code="df" \
  --placeholder_token="<derm_token>" \
  --initializer_token="skin" \
  --resolution=512 \
  --learning_rate=5e-4 \
  --max_train_steps=1000 \
  --output_dir="mlp-cw4/models/txt_inversion_dermatofibroma"
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

# A mapping from short lesion codes to full names (if needed)
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
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory containing 512x512 processed images (all together).",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="CSV file containing metadata (e.g., HAM10000_metadata.csv).",
    )
    parser.add_argument(
        "--lesion_code",
        type=str,
        required=True,
        help="Lesion code to filter images (e.g., 'df').",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        required=True,
        help="The new token to learn (e.g., <derm_token>).",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        required=True,
        help="An existing token used to initialize the new token (e.g., 'skin').",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output/txt_inversion")
    return parser.parse_args()


class LesionDataset(Dataset):
    """
    Loads images from a single directory by filtering the metadata CSV.
    Only images with a dx value matching the given lesion_code are used.
    """

    def __init__(
        self,
        data_root,
        metadata_file,
        lesion_code,
        tokenizer,
        placeholder_token,
        resolution=512,
    ):
        self.data_root = data_root
        self.metadata = pd.read_csv(metadata_file)
        # Filter for rows where the lesion code matches (case-insensitive)
        self.metadata = self.metadata[
            self.metadata["dx"].str.lower() == lesion_code.lower()
        ]
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.resolution = resolution
        self._length = len(self.metadata)
        # Create a uniform prompt using the placeholder token.
        self.prompt_text = f"An image of {self.placeholder_token}"
        self.tokenized_text = self.tokenizer(
            self.prompt_text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row["image_id"]
        image_path = os.path.join(self.data_root, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), resample=Image.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        return {"pixel_values": tensor, "input_ids": self.tokenized_text}


def main():
    args = parse_args()
    accelerator = Accelerator()
    set_seed(args.seed)

    # 1) Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )

    # 2) Add the new placeholder token and initialize its embedding
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The token {args.placeholder_token} already exists in the tokenizer."
        )
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("initializer_token must be a single token.")
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
    text_encoder.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[placeholder_token_id] = (
            text_encoder.get_input_embeddings().weight[initializer_token_id].clone()
        )

    # Freeze all parameters except the new token embedding
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.get_input_embeddings().weight.requires_grad = True

    # 3) Load VAE and UNet (using half precision)
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

    # 4) Create dataset and dataloader (using metadata filtering)
    dataset = LesionDataset(
        data_root=args.train_data_dir,
        metadata_file=args.metadata_file,
        lesion_code=args.lesion_code,
        tokenizer=tokenizer,
        placeholder_token=args.placeholder_token,
        resolution=args.resolution,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    params_to_optimize = [text_encoder.get_input_embeddings().weight]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)

    text_encoder, optimizer, dataloader = accelerator.prepare(
        text_encoder, optimizer, dataloader
    )

    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    text_encoder.train()
    for step in range(args.max_train_steps):
        batch = next(iter(dataloader))
        with accelerator.accumulate(text_encoder):
            pixel_values = batch["pixel_values"].to(
                accelerator.device, dtype=torch.float32
            )
            input_ids = batch["input_ids"].to(accelerator.device)

            latents = vae.encode(pixel_values.half()).latent_dist.sample().detach()
            latents = latents * 0.18215

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            enc_out = text_encoder(input_ids=input_ids)
            encoder_hidden_states = enc_out[0].half()

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})
        global_step += 1
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        learned_embeds = (
            text_encoder.get_input_embeddings()
            .weight[placeholder_token_id]
            .detach()
            .cpu()
            .unsqueeze(0)
        )
        learned_embeds_dict = {args.placeholder_token: learned_embeds}
        embed_path = os.path.join(args.output_dir, "learned_embeds.safetensors")
        save_file(learned_embeds_dict, embed_path)
        print(f"Saved learned embeddings to {embed_path}")


if __name__ == "__main__":
    main()
