#!/usr/bin/env python
# coding: utf-8

"""
train_lora.py

Fine-tunes a Stable Diffusion model using LoRA + Textual Inversion to learn a new
concept token representing your target lesion class (e.g. Dermatofibroma).

Usage (example):
---------------
accelerate launch train_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="data/dermatofibroma_images" \
  --output_dir="models/dermatofibroma_lora" \
  --placeholder_token="<LESION>" \
  --initializer_token="lesion" \
  --train_batch_size=1 \
  --max_train_steps=1000

After training, you will see a LoRA checkpoint (pytorch_lora_weights.safetensors)
and a learned_embeds.safetensors file in the output directory.
You can then load them together to generate new synthetic images representing
the concept <LESION>.
"""

import argparse
import os
import math
import logging
import random
import itertools
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.lora import LoRALinearLayer
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image

import numpy as np


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA on a lesion concept.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="Directory with images of the target lesion.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-output",
        help="Where to save the LoRA weights + embeddings.",
    )
    parser.add_argument("--placeholder_token", type=str, default="<LESION>")
    parser.add_argument(
        "--initializer_token",
        type=str,
        default="lesion",
        help="A similar word from the existing SD vocab to initialize from.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank dimension.")
    parser.add_argument(
        "--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]
    )
    return parser.parse_args()


class SimpleDataset(torch.utils.data.Dataset):
    """
    A dataset that loads all images from `instance_data_dir` and pairs them
    with prompts that reference a placeholder token, e.g. "<LESION> lesion".
    """

    def __init__(
        self, data_root, placeholder_token="<LESION>", resolution=512, center_crop=False
    ):
        self.data_root = data_root
        self.resolution = resolution
        self.placeholder_token = placeholder_token

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        self.images = []
        for fname in os.listdir(data_root):
            if any(fname.lower().endswith(e) for e in exts):
                self.images.append(os.path.join(data_root, fname))
        self._length = len(self.images)

        # Simple transforms
        self.center_crop = center_crop
        self.placeholder_prompt = f"{placeholder_token} lesion"

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        path = self.images[idx % self._length]
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # Resize to resolution
        image = image.resize((self.resolution, self.resolution), resample=Image.BICUBIC)
        arr = np.array(image).astype(np.uint8)
        arr = (arr / 127.5 - 1.0).astype(np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {
            "pixel_values": tensor,
            "prompt": self.placeholder_prompt,
        }


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
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    # 1) Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    # 2) Load text_encoder, vae, unet
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

    # Freeze them except LoRA/embedding
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 3) Add a new placeholder token to the tokenizer
    num_added_tokens = tokenizer.add_tokens([args.placeholder_token])
    if num_added_tokens == 0:
        raise ValueError(
            f"The token {args.placeholder_token} already exists in the tokenizer."
        )

    # 4) Resize token embeddings + initialize from initializer_token
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    init_token_id = tokenizer.convert_tokens_to_ids(args.initializer_token)
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
    # If initializer_token is not a single token, you'll need more advanced logic
    token_embeds[placeholder_token_id] = token_embeds[init_token_id].clone()

    # 5) Insert LoRA modules
    # In diffusers >=0.14, you can do unet.set_attn_processor(...)
    # but let's do it more manually to show how.
    # The unet has many attention blocks. We'll add low-rank adapters to them.
    for name, module in unet.named_modules():
        if (
            hasattr(module, "to_q")
            and hasattr(module, "to_k")
            and hasattr(module, "to_v")
        ):
            # Insert LoRA for each of these linear layers
            module.to_q.lora_layer = LoRALinearLayer(
                module.to_q.in_features, module.to_q.out_features, rank=args.rank
            )
            module.to_k.lora_layer = LoRALinearLayer(
                module.to_k.in_features, module.to_k.out_features, rank=args.rank
            )
            module.to_v.lora_layer = LoRALinearLayer(
                module.to_v.in_features, module.to_v.out_features, rank=args.rank
            )
            # We also add LoRA for to_out[0] if it exists
            if hasattr(module, "to_out") and isinstance(
                module.to_out, torch.nn.ModuleList
            ):
                if len(module.to_out) > 0 and hasattr(module.to_out[0], "in_features"):
                    module.to_out[0].lora_layer = LoRALinearLayer(
                        module.to_out[0].in_features,
                        module.to_out[0].out_features,
                        rank=args.rank,
                    )

    # We won't train text_encoder as a normal DreamBooth approach would. But we do want
    # to train the newly added placeholder token's embedding, so let's see:
    text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

    # 6) Create dataset + dataloader
    dataset = SimpleDataset(
        args.instance_data_dir,
        placeholder_token=args.placeholder_token,
        resolution=args.resolution,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )

    # 7) Collect trainable parameters
    trainable_params = []
    # The new token embedding
    embed_params = text_encoder.get_input_embeddings().weight[placeholder_token_id]
    embed_params.requires_grad_(True)
    # But the embedding weight is a single slice of a bigger tensor, so let's do a hack:
    # We'll pass the entire embedding as a parameter, but freeze all except placeholder token.
    text_encoder_params = [text_encoder.get_input_embeddings().weight]

    # The LoRA parameters in the UNet
    lora_params = []
    for _, module in unet.named_modules():
        if hasattr(module, "lora_layer"):
            lora_params.extend(module.lora_layer.parameters())
        # We also might have "to_out[0].lora_layer" if we find them

    # We'll unify them
    trainable_params = list(text_encoder_params) + list(lora_params)

    # Create the optimizer
    optimizer = torch.optim.AdamW(
        itertools.chain(*trainable_params), lr=args.learning_rate
    )

    # 8) Create LR scheduler
    max_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps,
    )

    # Prepare with accelerator
    unet, text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, dataloader, lr_scheduler
    )

    # Convert some modules to fp16/bf16 if needed
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    unet.train()
    text_encoder.train()

    global_step = 0
    total_steps = max_steps
    progress_bar = tqdm(
        range(total_steps), disable=not accelerator.is_local_main_process
    )

    for step in range(total_steps):
        batch = next(iter(dataloader))
        with accelerator.accumulate(unet):
            # 1) Convert images to latents
            pixel_values = batch["pixel_values"].to(
                accelerator.device, dtype=weight_dtype
            )
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # 2) Sample random noise
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

            # 3) Get text embedding
            prompts = batch["prompt"]
            token_output = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = token_output.input_ids.to(accelerator.device)
            with torch.no_grad():
                enc_out = text_encoder(input_ids=input_ids)
                encoder_hidden_states = enc_out[0].to(dtype=weight_dtype)

            # 4) Predict noise
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # 5) Compute loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)

        if global_step >= total_steps:
            break

        progress_bar.set_postfix({"step_loss": loss.item()})

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Save LoRA weights
        os.makedirs(args.output_dir, exist_ok=True)

        # We'll extract LoRA layers from unet
        unet_lora_state_dict = {}
        for name, module in accelerator.unwrap_model(unet).named_modules():
            if hasattr(module, "lora_layer"):
                # e.g. "unet.down_blocks.0.attentions.0...to_q.lora_layer.weight"
                # We'll store them in a standard way
                prefix_name = f"unet.{name}"
                for param_name, param in module.lora_layer.state_dict().items():
                    unet_lora_state_dict[f"{prefix_name}.{param_name}"] = param.cpu()

        # Save to safetensors
        from safetensors.torch import save_file

        save_file(
            unet_lora_state_dict,
            os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"),
        )

        # Save learned embedding
        # We'll do the entire embedding for now. Then we can selectively extract the slice.
        embedding_weights = {}
        learned_embeds = (
            accelerator.unwrap_model(text_encoder)
            .get_input_embeddings()
            .weight[placeholder_token_id]
        )
        embedding_weights[args.placeholder_token] = learned_embeds.detach().cpu()
        # save as safetensors
        # We'll store them all in "learned_embeds.safetensors"
        # If we had multiple tokens we might store them all
        from safetensors import torch as st_torch

        st_torch.save_file(
            embedding_weights,
            os.path.join(args.output_dir, "learned_embeds.safetensors"),
        )

        logger.info(f"LoRA training complete. Weights saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
