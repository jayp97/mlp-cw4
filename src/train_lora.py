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
accelerate launch src/train_lora.py \
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
        mixed_precision="no",  # Changed to "no" to avoid type mismatches
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

        # Save weights directly as standard PyTorch format
        torch_path = os.path.join(args.output_dir, "pytorch_lora_weights.bin")

        # First, extract all the LoRA parameters in a structured format
        lora_state_dict = {}

        for name, module in unwrapped_unet.named_modules():
            if hasattr(module, "lora_layer"):
                # Store original module name and parameter name
                module_name = name
                for p_name, p_val in module.lora_layer.state_dict().items():
                    key = f"{module_name}|{p_name}"
                    lora_state_dict[key] = p_val.cpu()

        # Save as PyTorch bin file
        torch.save(lora_state_dict, torch_path)
        logger.info(f"LoRA PyTorch weights saved to {torch_path}")

        # Also save an adapter_config.json file to help with loading
        adapter_config = {
            "base_model_name_or_path": args.pretrained_model_name_or_path,
            "lora_scale": 1.0,
            "lora_rank": args.rank,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
        }

        import json

        config_path = os.path.join(args.output_dir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)
        logger.info(f"Adapter config saved to {config_path}")

        # Also save a README with instructions
        readme_content = """# LoRA weights for Stable Diffusion HAM10000 Dataset

## Usage Instructions

These weights need to be loaded using the companion script. Due to LoRA format compatibility 
issues, we've saved the weights in a directly loadable PyTorch format.

Run the generate_synthetic_images.py script with:

```bash
python src/generate_synthetic_images.py \\
  --pretrained_model="runwayml/stable-diffusion-v1-5" \\
  --lora_weights="models/stable_diffusion_lora/pytorch_lora_weights.bin" \\
  --lesion_code="df" \\
  --num_images=20 \\
  --guidance_scale=7.5 \\
  --num_inference_steps=50 \\
  --seed=42 \\
  --output_dir="data/synthetic/images_dermatofibroma"
```
"""
        with open(os.path.join(args.output_dir, "README.md"), "w") as f:
            f.write(readme_content)

        logger.info(f"LoRA training complete! Results saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
