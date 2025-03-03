"""
Improvements Implemented:
- Removed the PEFT library usage entirely. Instead, we use the official LoRA support from the Diffusers library.
- Added LoRAAttnProcessor to each attention module in the UNet, making those layers trainable for LoRA.
- Save and load LoRA weights with unet.save_attn_procs() / unet.load_attn_procs() -- the official Diffusers approach.
- Retained seed fixing, mixed-precision training, and other improvements for stable training.
- Added a note that you can set TRAIN_STEPS=100 for a quick test and then revert to 1000 or more for a final run.
"""

import os
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Diffusers / Transformers
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    LoRAAttnProcessor,  # This is part of the official LoRA functionality in Diffusers
)
from transformers import CLIPTextModel, CLIPTokenizer


# -------------------------------
# Set random seed for reproducibility
# -------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ------------------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------------------
HF_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"  # Base Stable Diffusion model
SYNTHETIC_PATH = "data/synthetic/images_dermatofibroma/"  # Where generated images go
MODEL_SAVE_PATH = "models/stable_diffusion_lora/"  # Where LoRA weights are saved
os.makedirs(SYNTHETIC_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Training config
# For a quick test, you can set TRAIN_STEPS to e.g. 100. Then, if everything works,
# set it to 1000 (or higher) to overwrite the existing weights with a full run.
TRAIN_STEPS = 100
BATCH_SIZE = 1  # Typically small for LoRA fine-tuning
LEARNING_RATE = 5e-5

# For stable diffusion v1.5, 512x512 is the typical resolution
IMAGE_SIZE = 512

# LoRA rank: higher means more capacity for adaptation
LORA_RANK = 4

# Prompt used during training and generation.
PROMPT_TEMPLATE = (
    "A dermatofibroma lesion, dermoscopy image, high quality clinical photograph"
)

# Device setup: Use CUDA if available, otherwise MPS on Apple Silicon, else CPU
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {DEVICE}")


# ------------------------------------------------------------------------------
# Check if required data directories and files are accessible
# ------------------------------------------------------------------------------
def check_required_data():
    """
    Check if the necessary data directories and files are accessible before proceeding.
    """
    image_dir = "data/processed_sd/images/"
    metadata_file = "data/raw/HAM10000_metadata.csv"
    if not (os.path.isdir(image_dir) and os.access(image_dir, os.R_OK)):
        raise FileNotFoundError(
            f"Required directory '{image_dir}' does not exist or is not accessible."
        )
    if not (os.path.isfile(metadata_file) and os.access(metadata_file, os.R_OK)):
        raise FileNotFoundError(
            f"Required file '{metadata_file}' does not exist or is not accessible."
        )


# ------------------------------------------------------------------------------
# Custom Collate Function
# ------------------------------------------------------------------------------
def custom_collate_fn(batch):
    """
    This function unzips the batch of (image, prompt) tuples into two lists.
    """
    images, prompts = zip(*batch)
    return list(images), list(prompts)


# ------------------------------------------------------------------------------
# 1) Custom Dataset for Fine-Tuning
# ------------------------------------------------------------------------------
class DermatofibromaDataset(Dataset):
    """
    Loads dermatofibroma images from `data/processed_sd/images/` (HAM10000, preprocessed to 512x512)
    and provides them along with a text prompt.
    """

    def __init__(
        self,
        metadata_path="data/raw/HAM10000_metadata.csv",
        image_dir="data/processed_sd/images/",
    ):
        import pandas as pd

        # Load the main metadata CSV
        self.metadata = pd.read_csv(metadata_path)
        # Filter for dermatofibroma rows (where dx == 'df')
        self.df_metadata = self.metadata[self.metadata["dx"] == "df"]
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df_metadata)

    def __getitem__(self, idx):
        image_id = self.df_metadata.iloc[idx]["image_id"]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        # Load image and convert to RGB
        image = Image.open(image_path).convert("RGB")
        prompt = PROMPT_TEMPLATE
        return image, prompt


# ------------------------------------------------------------------------------
# 2) Fine-Tuning with LoRA (Diffusers Built-In) and Mixed Precision
# ------------------------------------------------------------------------------
def create_lora_layers(unet, r=4):
    """
    Sets the LoRAAttnProcessor on each attention module in the UNet, with the given rank r.
    This makes the relevant parameters trainable for LoRA fine-tuning.
    """
    for name, submodule in unet.named_modules():
        if hasattr(submodule, "set_processor"):
            # The official Diffusers approach: each cross/self-attn can have an AttnProcessor
            submodule.set_processor(
                LoRAAttnProcessor(hidden_size=submodule.out_channels, rank=r)
            )


def finetune_stable_diffusion_dermatofibroma():
    """
    Fine-tune Stable Diffusion on the dermatofibroma class using Diffusers LoRA.
    The resulting LoRA weights are saved to MODEL_SAVE_PATH.
    """

    # A. Load Stable Diffusion components from the base checkpoint
    print("Loading base UNet & text encoder from:", HF_MODEL_ID)
    unet = UNet2DConditionModel.from_pretrained(HF_MODEL_ID, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(HF_MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(HF_MODEL_ID, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(HF_MODEL_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(HF_MODEL_ID, subfolder="scheduler")

    # Freeze text encoder params to keep prompt embeddings stable
    for param in text_encoder.parameters():
        param.requires_grad = False

    # B. Add LoRAAttnProcessor to UNet for trainable parameters
    create_lora_layers(unet, r=LORA_RANK)

    # Move models to device
    unet.to(DEVICE)
    text_encoder.to(DEVICE)
    vae.to(DEVICE)

    # Filter trainable parameters from the unet (LoRA layers only)
    trainable_params = []
    for n, p in unet.named_parameters():
        if p.requires_grad:
            trainable_params.append(p)
    print(f"Found {len(trainable_params)} trainable params in UNet (LoRA layers).")

    # C. Set up optimizer and data
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    dataset = DermatofibromaDataset()  # Loads only DF images
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
    )

    # Setup mixed precision scaler if using CUDA
    scaler = torch.amp.GradScaler() if DEVICE == "cuda" else None

    # Training Loop
    unet.train()
    train_iter = iter(train_loader)
    for step in tqdm(range(TRAIN_STEPS), desc="Fine-tuning"):
        try:
            images, prompts = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, prompts = next(train_iter)

        # 1) Tokenize text
        input_ids = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(DEVICE)

        # 2) Get text embeddings (frozen encoder, so no text encoder grads)
        with torch.no_grad():
            text_embeddings = text_encoder(input_ids)[0]  # shape [B, seq_len, hid_dim]

        # 3) Convert images to Tensors
        import numpy as np

        images_np = np.stack([np.array(img) for img in images])  # [B, 512, 512, 3]
        images_np = np.moveaxis(images_np, -1, 1)  # => [B, 3, 512, 512]
        images_torch = torch.from_numpy(images_np).float().to(DEVICE) / 255.0

        # 4) Encode images to latents
        with torch.no_grad():
            latents_dist = vae.encode(images_torch).latent_dist
            latents = latents_dist.sample() * 0.18215

        # 5) Add random noise
        noise = torch.randn_like(latents)
        timesteps = (
            torch.randint(0, noise_scheduler.config.num_train_timesteps, (BATCH_SIZE,))
            .long()
            .to(DEVICE)
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 6) Predict noise with LoRA-enabled UNet
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
        else:
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # 7) Backprop and update
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Log every 50 steps
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    # D. Save LoRA weights
    print(f"Saving LoRA weights to {MODEL_SAVE_PATH} ...")
    unet.save_attn_procs(MODEL_SAVE_PATH)
    print(f"LoRA weights saved to {MODEL_SAVE_PATH}")


# ------------------------------------------------------------------------------
# 3) Generating Synthetic Images
# ------------------------------------------------------------------------------
def generate_synthetic_dermatofibroma(num_images=50):
    """
    Generate synthetic dermatofibroma images using the fine-tuned Stable Diffusion model.
    Uses LoRA weights from unet.save_attn_procs(MODEL_SAVE_PATH).
    Args:
        num_images (int): Number of synthetic images to generate.
    """
    # A. Load base pipeline
    print("Loading base pipeline ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,  # Turn off safety for medical images
    ).to(DEVICE)

    # B. Load LoRA weights from disk
    print("Loading LoRA weights from:", MODEL_SAVE_PATH)
    try:
        pipe.unet.load_attn_procs(MODEL_SAVE_PATH)
        print("Successfully loaded LoRA weights into the UNet.")
    except Exception as e:
        print(f"Failed to load LoRA weights from {MODEL_SAVE_PATH}.\nError: {e}")
        print("Using base model weights instead.")

    # C. Generate images
    for i in range(num_images):
        with torch.inference_mode():
            result = pipe(
                PROMPT_TEMPLATE,
                num_inference_steps=150,  # More steps for better quality
                guidance_scale=8.0,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            )
        image = result.images[0]

        # Save the generated image
        save_path = os.path.join(SYNTHETIC_PATH, f"synthetic_df_{i+1}.jpg")
        image.save(save_path)
        print(f"Generated synthetic image: {save_path}")


# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------
def main():
    """
    Main function to fine-tune Stable Diffusion with LoRA, then generate synthetic images.
    """
    check_required_data()  # Ensure necessary data files/dirs are accessible

    # Step 1: Fine-tune
    finetune_stable_diffusion_dermatofibroma()

    # Step 2: Generate images
    generate_synthetic_dermatofibroma(num_images=50)


if __name__ == "__main__":
    main()
