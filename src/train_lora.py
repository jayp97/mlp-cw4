#!/usr/bin/env python
# coding=utf-8

"""
train_lora.py

Trains LoRA on top of a Stable Diffusion model for the entire HAM10000 dataset at once.
We do NOT subdivide by lesion type in separate folders. Instead, each image is
accompanied by a textual prompt derived from its label in HAM10000_metadata.csv.

Example usage:

  accelerate launch src/train_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --metadata_file="data/raw/HAM10000_metadata.csv" \
    --train_data_dir="data/processed_sd/images" \
    --output_dir="models/stable_diffusion_lora" \
    --resolution=512 \
    --train_batch_size=1 \
    --max_train_steps=1000 \
    --learning_rate=1e-4 \
    --seed=42

After training completes, you'll get:
  - pytorch_lora_weights.safetensors  (LoRA checkpoint)
You can then use generate_synthetic_images.py to produce images of any lesion type.
"""

import argparse
import os
import math
import logging
import random
import itertools

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

# -------------------------------------------------------------------
# 1) A mapping from short lesion codes -> textual label for the prompt
#    per the standard HAM10000 dataset codes:
#
#    akiec = Actinic Keratoses,
#    bcc   = Basal Cell Carcinoma,
#    bkl   = Benign Keratosis,
#    df    = Dermatofibroma,
#    nv    = Melanocytic Nevus,
#    mel   = Melanoma,
#    vasc  = Vascular lesion
# -------------------------------------------------------------------
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
        help="Hugging Face Stable Diffusion model name or path, e.g. 'runwayml/stable-diffusion-v1-5'.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to 'HAM10000_metadata.csv' which has columns [image_id, dx, ...].",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory containing all processed SD images (512x512), named e.g. image_id.jpg.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-output",
        help="Where to save LoRA weights (pytorch_lora_weights.safetensors).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for training images, typically 512 for SD v1.",
    )
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rank", type=int, default=4, help="LoRA rank dimension (small integer)."
    )
    return parser.parse_args()


class HAMDataset(Dataset):
    """
    Loads all images from 'train_data_dir', with metadata in 'HAM10000_metadata.csv'.
    Creates a textual prompt for each image using its label. For instance, if dx='df'
    then the prompt is 'A photo of a Dermatofibroma lesion'.
    """

    def __init__(self, metadata_file, data_root, resolution=512):
        super().__init__()
        self.metadata = pd.read_csv(metadata_file)
        self.data_root = data_root
        self.resolution = resolution

        # We'll store all rows into a list, each row has image_id, dx
        # The actual image files in data_root are named <image_id>.jpg
        # We'll build a complete list of image paths + text prompts
        self.samples = []
        for idx, row in self.metadata.iterrows():
            image_id = row["image_id"]
            dx_code = str(row["dx"]).lower().strip()
            # if not in LABEL_MAP, we'll skip
            if dx_code not in LABEL_MAP:
                continue

            label_text = LABEL_MAP[dx_code]
            # Example prompt: "A photo of a Dermatofibroma lesion"
            prompt = f"A photo of a {label_text} lesion"
            image_path = os.path.join(self.data_root, f"{image_id}.jpg")

            # We'll check if file actually exists
            if not os.path.isfile(image_path):
                # If the file doesn't exist, skip or log a warning
                # (You might prefer an exception if you want it to always exist)
                logger.warning(f"Missing file: {image_path}, skipping.")
                continue

            self.samples.append((image_path, prompt))

        self._length = len(self.samples)
        logger.info(f"Dataset loaded {self._length} valid images from {data_root}.")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        image_path, prompt = self.samples[idx]
        # Load and preprocess
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # Resize to the model's training resolution (e.g. 512)
        image = image.resize((self.resolution, self.resolution), resample=Image.BICUBIC)
        arr = np.array(image, dtype=np.uint8)

        # Scale from [0..255] to [-1..1]
        arr = (arr / 127.5) - 1.0
        arr = arr.astype(np.float32)

        # Convert to tensor: shape [C, H, W]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {"pixel_values": tensor, "prompt": prompt}


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    set_seed(args.seed)

    # 1) Load the tokenizer, text encoder, VAE, UNet from your chosen SD base model
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

    # Freeze them all except LoRA weights we'll insert
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 2) Create the dataset & dataloader
    dataset = HAMDataset(
        metadata_file=args.metadata_file,
        data_root=args.train_data_dir,
        resolution=args.resolution,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )

    # 3) Insert LoRA modules into the UNet cross-attention layers
    for name, module in unet.named_modules():
        # Identify cross-attn layers by checking if it has to_q, to_k, to_v, etc.
        if (
            hasattr(module, "to_q")
            and hasattr(module, "to_k")
            and hasattr(module, "to_v")
        ):
            # Attach LoRALinearLayer to each
            module.to_q.lora_layer = LoRALinearLayer(
                module.to_q.in_features, module.to_q.out_features, rank=args.rank
            )
            module.to_k.lora_layer = LoRALinearLayer(
                module.to_k.in_features, module.to_k.out_features, rank=args.rank
            )
            module.to_v.lora_layer = LoRALinearLayer(
                module.to_v.in_features, module.to_v.out_features, rank=args.rank
            )

            # Also check if there's a to_out[0] linear
            if hasattr(module, "to_out") and isinstance(
                module.to_out, torch.nn.ModuleList
            ):
                if len(module.to_out) > 0 and hasattr(module.to_out[0], "in_features"):
                    module.to_out[0].lora_layer = LoRALinearLayer(
                        module.to_out[0].in_features,
                        module.to_out[0].out_features,
                        rank=args.rank,
                    )

    # 4) Collect LoRA params
    lora_params = []
    for _, submodule in unet.named_modules():
        if hasattr(submodule, "lora_layer"):
            lora_params.extend(submodule.lora_layer.parameters())

    # Optionally, you could also add LoRA to the text_encoder's cross-attn
    # if you want to adapt text embeddings. But let's keep it simpler here:
    # text_encoder is entirely frozen.

    # 5) Create optimizer
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)

    # 6) Create a scheduler
    max_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps,
    )

    # 7) Prepare with accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    unet.train()

    # We'll put vae & text_encoder on device, but keep them in eval mode & half precision
    device = accelerator.device
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    global_step = 0
    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    for step in range(max_steps):
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            # Re-start the dataloader
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        with accelerator.accumulate(unet):
            # 1) Convert images to latents
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(pixel_values.half()).latent_dist.sample() * 0.18215

            # 2) Sample random noise
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

            # 3) Get text embedding
            prompts = batch["prompt"]
            tokenized = tokenizer(
                list(prompts),
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                text_out = text_encoder(**tokenized)
                encoder_hidden_states = text_out[0].half()

            # 4) UNet forward
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # 5) Loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})
        global_step += 1
        if global_step >= max_steps:
            break

    # Done training
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        # Extract LoRA weights
        unet_lora_state_dict = {}
        for name, module in accelerator.unwrap_model(unet).named_modules():
            if hasattr(module, "lora_layer"):
                prefix_name = f"unet.{name}"
                for param_name, param in module.lora_layer.state_dict().items():
                    unet_lora_state_dict[f"{prefix_name}.{param_name}"] = param.cpu()

        # Save as safetensors
        from safetensors.torch import save_file

        lora_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        save_file(unet_lora_state_dict, lora_path)

        logger.info(f"LoRA training complete! LoRA weights saved to: {lora_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
