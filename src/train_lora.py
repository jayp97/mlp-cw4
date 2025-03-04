#!/usr/bin/env python
# coding=utf-8
"""
train_lora.py

Second step: LoRA fine-tuning of Stable Diffusion for a single lesion concept.
We load the textual embeddings from textual_inversion.py and then let LoRA
layers in the U-Net "learn" to better depict this token in actual images.

Output:
  - "pytorch_lora_weights.safetensors" in --output_dir

Example Usage:
python train_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --embed_path="mlp-cw4/models/txt_inversion_dermatofibroma/learned_embeds.safetensors" \
  --train_data_dir="mlp-cw4/data/processed_sd/images/dermatofibroma" \
  --token="<derm_token>" \
  --max_train_steps=1000 \
  --learning_rate=1e-4 \
  --rank=4 \
  --output_dir="mlp-cw4/models/stable_diffusion_lora"
"""
import argparse
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    LoraLoaderMixin,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import safe_open, save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--embed_path",
        type=str,
        required=True,
        help="learned_embeds.safetensors from textual_inversion.py",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Folder with 512×512 images for the same lesion concept.",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="The placeholder token, e.g. <derm_token>",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank dimension.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output/lora_out")
    return parser.parse_args()


class SingleConceptDataset(Dataset):
    """
    We pair each image with a text prompt that includes <lesion_token>.
    Example prompt: "An image of <derm_token>"
    """

    def __init__(self, data_root, tokenizer, token="<derm_token>", resolution=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.token = token
        self.resolution = resolution

        self.image_paths = []
        for fname in os.listdir(data_root):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_paths.append(os.path.join(data_root, fname))

        self._length = len(self.image_paths)
        self.prompt_text = f"An image of {self.token}"
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
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        return {"pixel_values": tensor, "input_ids": self.tokenized_text}


def main():
    args = parse_args()
    accelerator = Accelerator()
    set_seed(args.seed)

    # 1) Load textual embeddings
    learned_embeds = {}
    with safe_open(args.embed_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            learned_embeds[k] = f.get_tensor(k)

    # 2) Load base SD
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

    # Insert your new token embedding
    for token_name, embed in learned_embeds.items():
        # add token to the tokenizer if not present
        num_added = tokenizer.add_tokens(token_name)
        text_encoder.resize_token_embeddings(len(tokenizer))
        tid = tokenizer.convert_tokens_to_ids(token_name)
        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[tid] = embed

    # Freeze large modules
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float16)
    unet.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    # 3) Setup LoRA in unet
    for name, attn_processor in unet.attn_processors.items():
        unet.attn_processors[name] = LoRAAttnProcessor()

    # 4) Dataset
    dataset = SingleConceptDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        token=args.token,
        resolution=args.resolution,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )

    # 5) Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Trainable params
    lora_params = []
    for _, module in unet.attn_processors.items():
        for pname, p in module.named_parameters():
            if "lora_" in pname:
                p.requires_grad = True
                lora_params.append(p)

    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)
    num_update_steps_per_epoch = len(dataloader)
    max_train_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    # Accelerator
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

            # Convert images to latents
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

            # Text embeddings
            enc_out = text_encoder(input_ids=input_ids)
            encoder_hidden_states = enc_out[0].half()

            # Predict
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

    # Save LoRA weights
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # We'll gather unet's lora params
        lora_state_dict = {}
        for name, attn_processor in unet.attn_processors.items():
            for pname, param in attn_processor.named_parameters():
                if "lora_" in pname:
                    lora_state_dict[f"{name}.{pname}"] = param.cpu()

        lora_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        save_file(lora_state_dict, lora_path)
        print(f"LoRA training done. Weights saved at {lora_path}")


if __name__ == "__main__":
    main()
