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
from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import random
import logging
import math
import csv
from datetime import datetime

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
    num_epochs=40,  # Reduced from 75 to 40
    gradient_accumulation_steps=8,  # Increased from 4 to 8
    save_steps=500,
    mixed_precision="fp16",  # Changed to fp16 from bf16 for better compatibility
    seed=42,
    lora_r=8,  # Reduced from 16 to 8
    lora_alpha=32,
    lora_dropout=0.05,
    lora_text_encoder_r=8,
    lora_text_encoder_alpha=32,
    lora_text_encoder_dropout=0.1,
    train_text_encoder=False,  # Changed to False by default
    use_8bit_adam=True,  # Changed to True by default
    clip_grad_norm=1.0,
    warmup_steps=100,  # Reduced from 500 to 100
    train_split_path=None,
    # New parameter to select most efficient target modules
    target_modules_preset="efficient",  # options: "minimal", "efficient", "comprehensive"
):
    """
    Fine-tune Stable Diffusion using LoRA for generating skin lesion images.
    Optimized for faster training without sacrificing quality.

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
        use_8bit_adam: Whether to use 8-bit Adam optimizer
        clip_grad_norm: Maximum gradient norm for clipping
        warmup_steps: Number of warmup steps for learning rate scheduler
        train_split_path: Path to train split CSV
        target_modules_preset: Which set of modules to target with LoRA
    """
    # Set random seed
    torch.manual_seed(seed)
    random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup loss tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_log_file = os.path.join(output_dir, f"loss_log_{timestamp}.csv")
    with open(loss_log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss", "timestamp"])

    # Initialize Accelerator - using fp16 for better compatibility
    compute_dtype = (
        torch.float16
        if mixed_precision == "fp16"
        else torch.bfloat16 if mixed_precision == "bf16" else torch.float32
    )

    # Configure accelerator with gradient clipping
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        # Add gradient_clipping here instead of doing it manually
        kwargs_handlers=(
            [{"clip_grad_norm": clip_grad_norm}] if clip_grad_norm > 0 else None
        ),
    )

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    # Load VAE (consistent with accelerator precision)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

    # Freeze parameters
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # Configure target modules based on preset
    if target_modules_preset == "minimal":
        # Minimal set of modules for fastest training
        unet_target_modules = ["to_q", "to_v"]
    elif target_modules_preset == "comprehensive":
        # Full set of modules for best quality
        unet_target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
        ]
    else:  # "efficient" (default)
        # Balanced set for speed and quality
        unet_target_modules = ["to_q", "to_v", "to_out.0"]

    # Configure LoRA for UNet
    unet_lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=unet_target_modules,
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.train()

    # Configure LoRA for text encoder (optional)
    if train_text_encoder:
        text_encoder_lora_config = LoraConfig(
            r=lora_text_encoder_r,
            lora_alpha=lora_text_encoder_alpha,
            lora_dropout=lora_text_encoder_dropout,
            target_modules=["q_proj", "v_proj"],  # Reduced from 4 to 2 modules
        )
        text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
        text_encoder.train()

    # Load the scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Prepare dataset - modified to use train split if provided
    if train_split_path and os.path.exists(train_split_path):
        # Use only training images for fine-tuning
        logger.info(f"Using train split from: {train_split_path}")
        train_df = pd.read_csv(train_split_path)

        if class_name:
            # Filter to only include the specified class
            train_df = train_df[train_df["dx"] == class_name]

        # Get list of image IDs to use
        train_image_ids = set(train_df["image_id"].tolist())

        # Filter prompt file to only include training images
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

        # Create a filtered prompt file with only training images
        filtered_prompt_file = os.path.join(
            train_data_dir, f"{class_name}_train_prompts.txt"
        )

        with open(prompt_file, "r") as src, open(filtered_prompt_file, "w") as dst:
            for line in src:
                parts = line.strip().split(",", 1)
                if len(parts) != 2:
                    continue

                image_file, prompt = parts
                image_id = os.path.splitext(image_file)[0]

                # Include only if in training set
                if image_id in train_image_ids:
                    dst.write(f"{image_file},{prompt}\n")

        # Use the filtered prompt file
        prompt_file = filtered_prompt_file
    else:
        # Use the original logic for dataset preparation
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

    # Configure optimizer with warmup
    params_to_optimize = list(unet.parameters())
    if train_text_encoder:
        params_to_optimize.extend(list(text_encoder.parameters()))

    # Use 8-bit Adam if requested (helps with memory usage)
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(params_to_optimize, lr=learning_rate)
            logger.info("Using 8-bit Adam optimizer")
        except ImportError:
            logger.warning("bitsandbytes not installed, falling back to regular AdamW")
            optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

    # Get total training steps
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / gradient_accumulation_steps
    )
    max_train_steps = num_epochs * num_update_steps_per_epoch

    # Learning rate scheduler with warmup
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare everything for accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # Move text_encoder and vae to device for encoding (but not through accelerator)
    device = accelerator.device
    text_encoder.to(device)
    vae.to(device)

    # Create training stats tracking
    epoch_losses = []
    min_loss = float("inf")
    min_loss_epoch = 0
    min_loss_step = 0

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

        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # Get input data
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)

                # Encode text (with memory optimization)
                with torch.no_grad() if not train_text_encoder else torch.enable_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # Encode images to latent space (with memory optimization)
                with torch.no_grad():
                    # Don't convert the dtype here to avoid issues
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

                # Add noise to the latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the model prediction
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Calculate loss
                loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

                # Backward pass and optimization
                accelerator.backward(loss)

                # Update parameters (gradient clipping is handled by accelerator)
                optimizer.step()
                lr_scheduler.step()  # Update learning rate
                optimizer.zero_grad()

            # Update progress bar
            progress_bar.update(1)
            global_step += 1

            # Track loss
            loss_value = loss.detach().item()
            epoch_loss += loss_value

            # Log loss to CSV
            with open(loss_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch + 1,
                        global_step,
                        loss_value,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ]
                )

            # Save best model
            if loss_value < min_loss:
                min_loss = loss_value
                min_loss_epoch = epoch + 1
                min_loss_step = global_step

                # Save best model
                best_dir = os.path.join(output_dir, "best")
                os.makedirs(best_dir, exist_ok=True)

                # Save UNet LoRA weights
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet.save_pretrained(os.path.join(best_dir, "unet_lora"))

                # Save Text Encoder LoRA weights if trained
                if train_text_encoder:
                    unwrapped_text_encoder = text_encoder
                    unwrapped_text_encoder.save_pretrained(
                        os.path.join(best_dir, "text_encoder_lora")
                    )

                # Save model info
                with open(os.path.join(best_dir, "model_info.txt"), "w") as f:
                    f.write(f"Class: {class_name}\n")
                    f.write(f"Base model: {model_id}\n")
                    f.write(f"Best loss: {min_loss}\n")
                    f.write(f"Epoch: {min_loss_epoch}\n")
                    f.write(f"Step: {min_loss_step}\n")
                    f.write(
                        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )

            # Log info
            logs = {
                "loss": loss_value,
                "step": global_step,
                "lr": lr_scheduler.get_last_lr()[0],
            }
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

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - Average loss: {avg_epoch_loss:.6f}"
        )

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

        # Save epoch summary
        with open(os.path.join(epoch_dir, "epoch_info.txt"), "w") as f:
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"Average loss: {avg_epoch_loss:.6f}\n")
            f.write(f"Steps: {global_step}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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

    # Save overall training summary
    with open(os.path.join(output_dir, "training_summary.txt"), "w") as f:
        f.write(f"Class: {class_name}\n")
        f.write(f"Base model: {model_id}\n")
        f.write(f"Training steps: {global_step}\n")
        f.write(f"Training epochs: {num_epochs}\n")
        f.write(
            f"Best loss: {min_loss:.6f} (Epoch {min_loss_epoch}, Step {min_loss_step})\n"
        )
        f.write(f"Final loss: {epoch_losses[-1]:.6f}\n")
        f.write(f"Loss file: {os.path.basename(loss_log_file)}\n")

        # Add epoch loss summary
        f.write("\nEpoch loss summary:\n")
        for i, loss in enumerate(epoch_losses):
            f.write(f"Epoch {i+1}: {loss:.6f}\n")

    # Save model info in final dir
    with open(os.path.join(final_dir, "model_info.txt"), "w") as f:
        f.write(f"Class: {class_name}\n")
        f.write(f"Base model: {model_id}\n")
        f.write(f"Training steps: {global_step}\n")
        f.write(f"Training epochs: {num_epochs}\n")
        f.write(
            f"Best loss: {min_loss:.6f} (Epoch {min_loss_epoch}, Step {min_loss_step})\n"
        )

    logger.info("Training complete!")
    logger.info(
        f"Best loss: {min_loss:.6f} (Epoch {min_loss_epoch}, Step {min_loss_step})"
    )
    logger.info(f"Loss log saved to: {loss_log_file}")

    return os.path.join(final_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion with LoRA for skin lesion images (optimized)"
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
        "--num_epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam optimizer",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default=None,
        help="Path to train split CSV",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train text encoder",
    )
    parser.add_argument(
        "--target_modules_preset",
        type=str,
        default="efficient",
        choices=["minimal", "efficient", "comprehensive"],
        help="Which set of modules to target with LoRA",
    )

    args = parser.parse_args()
    train_lora(
        model_id=args.model_id,
        train_data_dir=args.train_data_dir,
        class_name=args.class_name,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        use_8bit_adam=args.use_8bit_adam,
        train_split_path=args.train_split,
        lora_r=args.lora_r,
        train_text_encoder=args.train_text_encoder,
        target_modules_preset=args.target_modules_preset,
    )
