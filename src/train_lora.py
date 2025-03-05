#!/usr/bin/env python
# coding=utf-8

"""
train_lora.py

Trains LoRA on a Stable Diffusion model for the entire HAM10000 dataset.
Each image (from a single folder) is paired with a prompt derived from its lesion label
in HAM10000_metadata.csv. Only the LoRA parameters inserted into the UNet cross-attention
layers are updated while the base model weights remain frozen.

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
        help="Path to HAM10000_metadata.csv with columns [image_id, dx, ...].",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory of processed SD images (512x512), each named <image_id>.jpg.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-output",
        help="Directory where LoRA weights are saved (pytorch_lora_weights.safetensors).",
    )
    parser.add_argument(
        "--resolution", type=int, default=512, help="Training resolution, e.g., 512."
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
        help="LoRA rank dimension for each cross-attn projection.",
    )
    return parser.parse_args()


class HAMDataset(Dataset):
    """
    Loads images from train_data_dir using metadata from HAM10000_metadata.csv.
    Each image is paired with a prompt constructed from its dx code.
    E.g., if dx == "df", then prompt is "A photo of a Dermatofibroma lesion".
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

    # 1) Load base components.
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

    # 4) Insert LoRA modules into UNet cross-attention layers.
    for _, module in unet.named_modules():
        if (
            hasattr(module, "to_q")
            and hasattr(module, "to_k")
            and hasattr(module, "to_v")
        ):
            module.to_q.lora_layer = LoRALinearLayer(
                module.to_q.in_features, module.to_q.out_features, rank=args.rank
            )
            module.to_k.lora_layer = LoRALinearLayer(
                module.to_k.in_features, module.to_k.out_features, rank=args.rank
            )
            module.to_v.lora_layer = LoRALinearLayer(
                module.to_v.in_features, module.to_v.out_features, rank=args.rank
            )
            if hasattr(module, "to_out") and isinstance(
                module.to_out, torch.nn.ModuleList
            ):
                if len(module.to_out) > 0 and hasattr(module.to_out[0], "in_features"):
                    module.to_out[0].lora_layer = LoRALinearLayer(
                        module.to_out[0].in_features,
                        module.to_out[0].out_features,
                        rank=args.rank,
                    )

    # 5) Gather LoRA parameters.
    lora_params = []
    for _, submodule in unet.named_modules():
        if hasattr(submodule, "lora_layer"):
            for param in submodule.lora_layer.parameters():
                param.requires_grad = True
                lora_params.append(param)

    # 6) Create optimizer for LoRA parameters.
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)

    # 7) Create a constant LR scheduler.
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

    # 9) Move VAE and text encoder to device in fp16.
    device = accelerator.device
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)
    # Note: unet was prepared and will use mixed precision as set via accelerate.

    global_step = 0
    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    for _ in range(max_steps):
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
        prompts = batch["prompt"]

        with accelerator.accumulate(unet):
            # i) Convert images to latents.
            # IMPORTANT: Cast pixel_values to half to match VAE's weights.
            with torch.no_grad():
                latents = vae.encode(pixel_values.half()).latent_dist.sample() * 0.18215

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
        lora_state = {}
        for name, module in unwrapped_unet.named_modules():
            if hasattr(module, "lora_layer"):
                prefix = f"unet.{name}"
                for p_name, p_val in module.lora_layer.state_dict().items():
                    lora_state[f"{prefix}.{p_name}"] = p_val.cpu()
        from safetensors.torch import save_file

        out_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        save_file(lora_state, out_path)
        logger.info(f"LoRA training complete! Weights saved to {out_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
