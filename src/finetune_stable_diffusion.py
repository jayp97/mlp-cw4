#!/usr/bin/env python
# coding=utf-8
"""
finetune_stable_diffusion.py

LoRA fine-tuning on the entire dataset (all labels), using
a manual LoRA injection approach for cross-attention sub-layers.
We do NOT rely on AttnProcessor(2_0) having parameters, because in
modern diffusers versions, the default attention processors (like
AttnProcessor2_0) are parameter-free. Instead, we manually insert LoRA
down/up submodules into each cross-attention linear layer (to_q, to_k,
to_v, to_out).

We create prompts like "(skin lesion, {full_label})" from the metadata CSV
for each image.

----------------------------------------
Requirements:
  - diffusers >= 0.14
  - accelerate
  - safetensors
  - torch
----------------------------------------

Example Usage:
  python finetune_stable_diffusion.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --metadata_file="data/raw/HAM10000_metadata.csv" \
    --train_data_dir="data/processed_sd/images" \
    --resolution=512 \
    --learning_rate=5e-5 \
    --max_train_steps=1000 \
    --batch_size=1 \
    --lora_rank=4 \
    --seed=42 \
    --output_dir="models/stable_diffusion_lora"

After completion, the LoRA adapters (only) are saved into
--output_dir/lora_weights.safetensors
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

# We'll import the LoRA saving helpers:
from diffusers.loaders import StableDiffusionLoraLoaderMixin

# -------------------------------------------------------------------
# MAPPING: short lesion codes -> full textual label for prompts
# -------------------------------------------------------------------
LABEL_MAP = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "nv": "Melanocytic Nevus",
    "mel": "Melanoma",
    "vasc": "Vascular Lesion",
}


# -------------------------------------------------------------------
# LoRA Hook: We'll define a forward_pre_hook to add LoRA's contribution
# -------------------------------------------------------------------
def lora_forward_pre_hook(module, input):
    """
    We intercept the forward pass of the original linear layer.
    'module' has lora_down and lora_up, each a small linear layer.
    input is a tuple of (x, ) i.e. the input tensor(s).
    """
    x = input[0]  # The original input to the linear layer
    # The standard (original) forward output:
    out_normal = module.original_forward(x)

    # The LoRA forward pass: out_lora = lora_up( lora_down( x ) )
    out_lora = module.lora_up(module.lora_down(x)) * module.lora_alpha

    return (out_normal + out_lora,)


def add_lora_to_linear(linear_layer, rank=4, alpha=1.0):
    """
    Insert LoRA submodules (lora_down, lora_up) into the given linear_layer.
    We'll attach a forward_pre_hook so that:
      out = original_forward(x) + alpha * lora_up(lora_down(x))
    """
    if not isinstance(linear_layer, torch.nn.Linear):
        return

    # 1) Create LoRA down & up
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features

    lora_down = torch.nn.Linear(in_features, rank, bias=False)
    lora_up = torch.nn.Linear(rank, out_features, bias=False)

    # 2) Zero init so that they start as no-op
    torch.nn.init.zeros_(lora_down.weight)
    torch.nn.init.zeros_(lora_up.weight)

    # 3) Convert them to the same dtype as the parent's weight
    lora_down = lora_down.to(linear_layer.weight.dtype)
    lora_up = lora_up.to(linear_layer.weight.dtype)

    # 4) Attach them
    linear_layer.lora_down = lora_down
    linear_layer.lora_up = lora_up
    linear_layer.lora_alpha = alpha

    # 5) Overwrite forward with a forward_pre_hook
    if not hasattr(linear_layer, "original_forward"):
        linear_layer.original_forward = linear_layer.forward

        def dummy_forward(x):
            # The real logic is in the forward_pre_hook
            return x

        linear_layer.forward = dummy_forward

    # 6) Register the hook once
    if not getattr(linear_layer, "_lora_hook_registered", False):
        linear_layer.register_forward_pre_hook(lora_forward_pre_hook)
        linear_layer._lora_hook_registered = True


# -------------------------------------------------------------------
# Our dataset class (full dataset, all labels)
# -------------------------------------------------------------------
class AllLesionDataset(Dataset):
    """
    For each row in the metadata, we load the image and build a prompt:
       (skin lesion, {full_label})
    Where full_label is the full text from LABEL_MAP or the dx if missing.
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
        dx_code = str(row["dx"]).lower().strip()
        full_label = LABEL_MAP.get(dx_code, dx_code)
        prompt = f"(skin lesion, {full_label})"

        # Tokenize prompt for conditioning
        tokenized_text = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        # Load image
        img_path = os.path.join(self.data_root, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.BICUBIC)
        arr = np.array(image).astype(np.uint8)
        arr = (arr / 127.5 - 1.0).astype(np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {"pixel_values": tensor, "input_ids": tokenized_text}


def main():
    # ----------------------------------------------------------------
    # Parse args, set up Accelerator, fix random seeds
    # ----------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Metadata CSV (e.g. HAM10000_metadata.csv).",
    )
    parser.add_argument(
        "--train_data_dir", type=str, required=True, help="Folder with 512x512 images."
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

    args = parser.parse_args()
    accelerator = Accelerator()
    set_seed(args.seed)

    # ----------------------------------------------------------------
    # 1) Load the base SD components (tokenizer, text encoder, VAE, UNet)
    # ----------------------------------------------------------------
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    # Freeze text encoder (not training it)
    for param in text_encoder.parameters():
        param.requires_grad = False

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    # freeze original
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    device = accelerator.device
    vae.to(device, dtype=torch.float16)
    unet.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    # ----------------------------------------------------------------
    # 2) Insert LoRA submodules into cross-attn layers
    # ----------------------------------------------------------------
    for name, submodule in unet.named_modules():
        if (
            hasattr(submodule, "to_q")
            and isinstance(submodule.to_q, torch.nn.Linear)
            and hasattr(submodule, "to_k")
            and isinstance(submodule.to_k, torch.nn.Linear)
            and hasattr(submodule, "to_v")
            and isinstance(submodule.to_v, torch.nn.Linear)
        ):
            # Add LoRA to submodule.to_q, submodule.to_k, submodule.to_v, and submodule.to_out
            def patch_linear_if_exists(layer):
                if isinstance(layer, torch.nn.Linear):
                    add_lora_to_linear(layer, rank=args.lora_rank, alpha=1.0)

            patch_linear_if_exists(submodule.to_q)
            patch_linear_if_exists(submodule.to_k)
            patch_linear_if_exists(submodule.to_v)

            if hasattr(submodule, "to_out"):
                # to_out can be (linear, dropout) in a list or a single linear
                if (
                    isinstance(submodule.to_out, torch.nn.ModuleList)
                    and len(submodule.to_out) > 0
                ):
                    # typically [nn.Linear, nn.Dropout]
                    if isinstance(submodule.to_out[0], torch.nn.Linear):
                        add_lora_to_linear(
                            submodule.to_out[0], rank=args.lora_rank, alpha=1.0
                        )
                elif isinstance(submodule.to_out, torch.nn.Linear):
                    # direct linear
                    add_lora_to_linear(submodule.to_out, rank=args.lora_rank, alpha=1.0)

    # ----------------------------------------------------------------
    # 3) Create dataset & dataloader
    # ----------------------------------------------------------------
    dataset = AllLesionDataset(
        data_root=args.train_data_dir,
        metadata_file=args.metadata_file,
        resolution=args.resolution,
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # ----------------------------------------------------------------
    # 4) Create noise scheduler & gather LoRA parameters
    # ----------------------------------------------------------------
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # gather LoRA parameters by scanning for lora_down/lora_up
    lora_params = []
    for _, mod in unet.named_modules():
        if hasattr(mod, "lora_down") and hasattr(mod, "lora_up"):
            lora_params.append(mod.lora_down.weight)
            lora_params.append(mod.lora_up.weight)

    if len(lora_params) == 0:
        raise ValueError(
            "No LoRA parameters found. Possibly code mismatch or no cross-attn layers found."
        )

    print(f"Found {len(lora_params)} LoRA param tensors in total.")

    # Create optimizer & lr scheduler
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)
    max_train_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    # Prepare with Accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    unet.train()
    vae.eval()
    text_encoder.eval()

    # ----------------------------------------------------------------
    # 5) Fine-tuning loop
    # ----------------------------------------------------------------
    global_step = 0
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )

    data_iter = iter(dataloader)  # We'll re-init every epoch, or just do infinite
    for step in range(max_train_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(unet):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            input_ids = batch["input_ids"].to(device)

            # 1) encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values.half()).latent_dist.sample() * 0.18215

            # 2) sample random noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # 3) random timesteps
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            # 4) add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 5) text encoder forward
            with torch.no_grad():
                text_out = text_encoder(input_ids=input_ids)
                encoder_hidden_states = text_out[0].half()

            # 6) U-Net forward to predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # 7) Backprop
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})
        global_step += 1
        if global_step >= max_train_steps:
            break

    # ----------------------------------------------------------------
    # 6) Save only the LoRA weights
    #    This produces a separate .safetensors file containing just
    #    the LoRA submodules. We'll call it 'lora_weights.safetensors'.
    # ----------------------------------------------------------------
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        # 1) Extract the LoRA weights from unet
        unet_lora_dict = StableDiffusionLoraLoaderMixin.unet_attn_processors_state_dict(
            unet
        )

        # 2) Save them
        StableDiffusionLoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_dict,
            is_main_process=True,
            safe_serialization=True,
            weight_name="lora_weights.safetensors",
        )

        print(
            f"LoRA fine-tuning complete!\nLoRA-only weights saved to: {args.output_dir}/lora_weights.safetensors"
        )


if __name__ == "__main__":
    main()
