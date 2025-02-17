"""
stable_diffusion_generation.py

This script fine-tunes Stable Diffusion using LoRA on the dermatofibroma class from the HAM10000 dataset
and generates synthetic images to overbalance the class. The synthetic images are saved in the
`data/synthetic/images_dermatofibroma/` directory for later use in training the EfficientNetV2 model.
"""

import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from PIL import Image

# Hugging Face imports
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.loaders import LoraLoaderMixin

# Configuration
HF_MODEL_ID = "runwayml/stable-diffusion-v1-5"  # Base Stable Diffusion model
SYNTHETIC_PATH = (
    "data/synthetic/images_dermatofibroma/"  # Path to save synthetic images
)
MODEL_SAVE_PATH = (
    "models/stable_diffusion_lora/"  # Path to save fine-tuned LoRA weights
)
os.makedirs(SYNTHETIC_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Training config
TRAIN_STEPS = 100  # Number of fine-tuning steps (adjust based on your dataset size)
LORA_RANK = 4  # LoRA rank dimension (controls the size of the fine-tuned model)
BATCH_SIZE = 1  # Batch size for fine-tuning
IMAGE_SIZE = 224  # Image size (matches HAM10000 processed image size)
PROMPT_TEMPLATE = (
    "A dermatofibroma lesion, dermoscopy image, high quality clinical photograph"
)

# Device setup
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")


class DermatofibromaDataset(Dataset):
    """
    Custom dataset class for loading dermatofibroma images and their metadata.
    Assumes the HAM10000 dataset is preprocessed and stored in `data/processed/images/`.
    """

    def __init__(
        self,
        metadata_path="data/raw/HAM10000_metadata.csv",
        image_dir="data/processed/images/",
    ):
        """
        Initialize the dataset.
        Args:
            metadata_path (str): Path to the HAM10000 metadata CSV file.
            image_dir (str): Directory containing preprocessed images.
        """
        import pandas as pd

        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        self.image_dir = image_dir

        # Filter for dermatofibroma class (dx = 'df')
        self.df_metadata = self.metadata[self.metadata["dx"] == "df"]

    def __len__(self):
        """Return the number of dermatofibroma images."""
        return len(self.df_metadata)

    def __getitem__(self, idx):
        """
        Load an image and its metadata.
        Args:
            idx (int): Index of the image to load.
        Returns:
            image (PIL.Image): The loaded image.
            prompt (str): A text prompt describing the image.
        """
        # Get image path and metadata
        image_id = self.df_metadata.iloc[idx]["image_id"]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Create a prompt for fine-tuning
        prompt = PROMPT_TEMPLATE

        return image, prompt


def finetune_stable_diffusion_dermatofibroma():
    """
    Fine-tune Stable Diffusion using LoRA on the dermatofibroma class.
    This function loads the base model, applies LoRA fine-tuning, and saves the fine-tuned weights.
    """
    # Initialize base model components
    unet = UNet2DConditionModel.from_pretrained(HF_MODEL_ID, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(HF_MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(HF_MODEL_ID, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(HF_MODEL_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(HF_MODEL_ID, subfolder="scheduler")

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Layers to apply LoRA
        init_lora_weights="gaussian",  # Initialize LoRA weights with Gaussian distribution
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()  # Print the number of trainable parameters

    # Setup optimizer and dataset
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    dataset = DermatofibromaDataset()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Move models to the appropriate device
    unet.to(DEVICE)
    text_encoder.to(DEVICE)
    vae.to(DEVICE)

    # Training loop
    unet.train()
    for step in tqdm(range(TRAIN_STEPS), desc="Fine-tuning"):
        # Load a batch of images and prompts
        images, prompts = next(iter(train_loader))

        # Tokenize and encode text prompts
        input_ids = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
        text_embeddings = text_encoder(input_ids)[0]

        # Convert images to latents (compressed representation)
        latents = vae.encode(images.to(DEVICE)).latent_dist.sample()
        latents = latents * 0.18215  # Scale latents

        # Add noise to latents
        noise = torch.randn_like(latents)
        timesteps = (
            torch.randint(0, noise_scheduler.num_train_timesteps, (BATCH_SIZE,))
            .long()
            .to(DEVICE)
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise and compute loss
        noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save fine-tuned LoRA weights
    unet.save_pretrained(MODEL_SAVE_PATH)
    print(f"Saved fine-tuned LoRA weights to {MODEL_SAVE_PATH}")


def generate_synthetic_dermatofibroma(num_images=50):
    """
    Generate synthetic dermatofibroma images using the fine-tuned Stable Diffusion model.
    Args:
        num_images (int): Number of synthetic images to generate.
    """
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,  # Disable safety checker for medical images
    )

    # Load fine-tuned LoRA weights if available
    if os.path.exists(MODEL_SAVE_PATH):
        pipe.unet.load_attn_procs(MODEL_SAVE_PATH)
        print("Loaded fine-tuned LoRA weights")

    # Move pipeline to the appropriate device
    pipe.to(DEVICE)

    # Generation loop
    for i in range(num_images):
        result = pipe(
            PROMPT_TEMPLATE,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        )

        # Save generated image
        image = result.images[0]
        save_path = os.path.join(SYNTHETIC_PATH, f"synthetic_df_{i+1}.jpg")
        image.save(save_path)
        print(f"Generated synthetic image: {save_path}")


def main():
    """
    Main function to fine-tune Stable Diffusion and generate synthetic images.
    """
    # Fine-tune Stable Diffusion with LoRA
    finetune_stable_diffusion_dermatofibroma()

    # Generate synthetic images
    generate_synthetic_dermatofibroma(num_images=50)


if __name__ == "__main__":
    main()
