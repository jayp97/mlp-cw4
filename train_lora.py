import os
import argparse
import torch
from accelerate import Accelerator
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import random
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DermatologyDataset(Dataset):
    def __init__(self, image_dir, prompt_file, tokenizer, size=512, shuffle=True):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.size = size

        # Load prompts
        self.image_paths = []
        self.prompts = []

        with open(prompt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",", 1)  # Split at first comma
                if len(parts) != 2:
                    continue

                image_file, prompt = parts
                image_path = os.path.join(image_dir, image_file)

                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                    self.prompts.append(prompt)

        if shuffle:
            data = list(zip(self.image_paths, self.prompts))
            random.shuffle(data)
            self.image_paths, self.prompts = zip(*data)

        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        logger.info(f"Loaded dataset with {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        prompt = self.prompts[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Tokenize the prompt
        prompt_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": image, "input_ids": prompt_ids}


def train_lora(
    model_id="runwayml/stable-diffusion-v1-5",
    train_data_dir="data/fine_tuning_class_specific",
    class_name=None,
    output_dir="models/lora",
    learning_rate=1e-4,
    batch_size=1,
    num_epochs=100,
    gradient_accumulation_steps=4,
    save_steps=500,
    mixed_precision="fp16",
    seed=42,
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_text_encoder_r=8,
    lora_text_encoder_alpha=32,
    lora_text_encoder_dropout=0.1,
    train_text_encoder=True,
):
    """
    Fine-tune Stable Diffusion using LoRA for generating skin lesion images.

    Args:
        model_id: HuggingFace model ID for Stable Diffusion
        train_data_dir: Directory containing training data
        class_name: Specific lesion class to train on
        output_dir: Directory to save the trained model
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Number of steps to accumulate gradients
        save_steps: Save checkpoint every N steps
        mixed_precision: Mixed precision training ("fp16" or "bf16")
        seed: Random seed
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_text_encoder_r: LoRA rank for text encoder
        lora_text_encoder_alpha: LoRA alpha for text encoder
        lora_text_encoder_dropout: LoRA dropout for text encoder
        train_text_encoder: Whether to train the text encoder
    """
    # Set random seed
    torch.manual_seed(seed)
    random.seed(seed)

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    # Load VAE - THIS IS ESSENTIAL FOR PROPER LATENT ENCODING
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

    # Freeze parameters
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)  # Make sure VAE is frozen

    # Configure LoRA for UNet
    unet_lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.train()

    # Configure LoRA for text encoder (optional)
    if train_text_encoder:
        text_encoder_lora_config = LoraConfig(
            r=lora_text_encoder_r,
            lora_alpha=lora_text_encoder_alpha,
            lora_dropout=lora_text_encoder_dropout,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
        text_encoder.train()

    # Load the scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Prepare dataset
    if class_name:
        image_dir = os.path.join(train_data_dir, class_name)
        prompt_file = os.path.join(train_data_dir, f"{class_name}_prompts.txt")
    else:
        # Find a valid class directory by looking for prompt files
        prompt_files = [
            f for f in os.listdir(train_data_dir) if f.endswith("_prompts.txt")
        ]
        if not prompt_files:
            raise ValueError(
                f"No prompt files found in {train_data_dir}. Please run data_preparation.py first."
            )

        class_name = prompt_files[0].replace("_prompts.txt", "")
        image_dir = os.path.join(train_data_dir, class_name)
        prompt_file = os.path.join(train_data_dir, prompt_files[0])

    logger.info(f"Training on class: {class_name}")
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Prompt file: {prompt_file}")

    dataset = DermatologyDataset(
        image_dir=image_dir, prompt_file=prompt_file, tokenizer=tokenizer
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Configure optimizer
    params_to_optimize = list(unet.parameters())
    if train_text_encoder:
        params_to_optimize.extend(list(text_encoder.parameters()))

    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

    # Prepare everything for accelerator
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    # Move text_encoder and vae to device for encoding (but not through accelerator since we don't train them)
    device = accelerator.device
    text_encoder.to(device)
    vae.to(device)

    # Get total training steps
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / gradient_accumulation_steps
    )
    max_train_steps = num_epochs * num_update_steps_per_epoch

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        unet.train()
        if train_text_encoder:
            text_encoder.train()

        progress_bar = tqdm(
            total=len(dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # Get input data
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)

                # Encode text
                if train_text_encoder:
                    encoder_hidden_states = text_encoder(input_ids)[0]
                else:
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(input_ids)[0]

                # Encode images to latent space using VAE
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the model prediction
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Calculate loss
                loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

                accelerator.backward(loss)

                # Update parameters
                optimizer.step()
                optimizer.zero_grad()

            # Update progress bar
            progress_bar.update(1)
            global_step += 1

            # Log info
            logs = {"loss": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)

            # Save checkpoint
            if global_step % save_steps == 0:
                # Save LoRA weights for UNet
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)

                # Save UNet LoRA weights and config
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet.save_pretrained(os.path.join(ckpt_dir, "unet_lora"))

                # Save Text Encoder LoRA weights and config if trained
                if train_text_encoder:
                    unwrapped_text_encoder = text_encoder
                    unwrapped_text_encoder.save_pretrained(
                        os.path.join(ckpt_dir, "text_encoder_lora")
                    )

                logger.info(f"Saved checkpoint at step {global_step}")

        # Save after each epoch
        epoch_dir = os.path.join(output_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Save UNet LoRA weights and config
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(os.path.join(epoch_dir, "unet_lora"))

        # Save Text Encoder LoRA weights and config if trained
        if train_text_encoder:
            unwrapped_text_encoder = text_encoder
            unwrapped_text_encoder.save_pretrained(
                os.path.join(epoch_dir, "text_encoder_lora")
            )

        logger.info(f"Saved model at epoch {epoch+1}")

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    # Save UNet LoRA weights and config
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(os.path.join(final_dir, "unet_lora"))

    # Save Text Encoder LoRA weights and config if trained
    if train_text_encoder:
        unwrapped_text_encoder = text_encoder
        unwrapped_text_encoder.save_pretrained(
            os.path.join(final_dir, "text_encoder_lora")
        )

    # Save model info - class name and other details
    with open(os.path.join(final_dir, "model_info.txt"), "w") as f:
        f.write(f"Class: {class_name}\n")
        f.write(f"Base model: {model_id}\n")
        f.write(f"Training steps: {global_step}\n")
        f.write(f"Training epochs: {num_epochs}\n")

    logger.info("Training complete!")
    return os.path.join(final_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion with LoRA for skin lesion images"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="data/fine_tuning_class_specific",
        help="Directory with training data",
    )
    parser.add_argument(
        "--class_name", type=str, default=None, help="Specific lesion class to train on"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/lora",
        help="Output directory for model",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )

    args = parser.parse_args()
    train_lora(**vars(args))
