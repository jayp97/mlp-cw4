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

# Hugging Face / Diffusers imports
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.loaders import LoraLoaderMixin

# ------------------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------------------
HF_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"  # Base Stable Diffusion model
SYNTHETIC_PATH = "data/synthetic/images_dermatofibroma/"  # Where generated images go
MODEL_SAVE_PATH = "models/stable_diffusion_lora/"  # Where LoRA weights are saved
os.makedirs(SYNTHETIC_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Training config
TRAIN_STEPS = 100  # Number of fine-tuning steps (adjust based on dataset size & budget)
LORA_RANK = 4  # Dimension controlling the size/effect of LoRA updates
BATCH_SIZE = 1  # Batch size for fine-tuning; typically small for LoRA

# For stable diffusion v1.5, 512x512 is the typical resolution
IMAGE_SIZE = 512

# This prompt is used both during training (fine-tuning) and generation
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
# Custom Collate Function
# ------------------------------------------------------------------------------
def custom_collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    This function simply unzips the batch of (image, prompt) tuples
    into two lists without attempting to automatically convert the PIL images.
    """
    images, prompts = zip(*batch)
    return list(images), list(prompts)


# ------------------------------------------------------------------------------
# 1) Custom Dataset for Fine-Tuning
# ------------------------------------------------------------------------------
class DermatofibromaDataset(Dataset):
    """
    Loads dermatofibroma images from `data/processed_sd/images/` (HAM10000,
    preprocessed to 512x512) and provides them, along with a text prompt.
    """

    def __init__(
        self,
        metadata_path="data/raw/HAM10000_metadata.csv",
        image_dir="data/processed_sd/images/",  # Now using 512x512 images
    ):
        """
        Args:
            metadata_path (str): Path to the HAM10000 metadata CSV file.
            image_dir (str): Directory containing the 512x512 images for SD.
        """
        import pandas as pd

        # Load the main metadata CSV
        self.metadata = pd.read_csv(metadata_path)
        # Filter out only dermatofibroma rows (where dx == 'df')
        self.df_metadata = self.metadata[self.metadata["dx"] == "df"]
        self.image_dir = image_dir

    def __len__(self):
        """Return how many DF images we have."""
        return len(self.df_metadata)

    def __getitem__(self, idx):
        """
        Returns a single image (as a PIL Image) and a text prompt for stable diffusion fine-tuning.
        """
        image_id = self.df_metadata.iloc[idx]["image_id"]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        # Load the image as PIL and convert to RGB
        image = Image.open(image_path).convert("RGB")
        prompt = PROMPT_TEMPLATE
        return image, prompt


# ------------------------------------------------------------------------------
# 2) Fine-Tuning with LoRA
# ------------------------------------------------------------------------------
def finetune_stable_diffusion_dermatofibroma():
    """
    Fine-tune Stable Diffusion on the dermatofibroma class using LoRA.
    The resulting LoRA weights are saved to MODEL_SAVE_PATH.
    """

    # --------------------------------------------------------------------------
    # A. Load Stable Diffusion components from a base checkpoint
    # --------------------------------------------------------------------------
    unet = UNet2DConditionModel.from_pretrained(HF_MODEL_ID, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(HF_MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(HF_MODEL_ID, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(HF_MODEL_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(HF_MODEL_ID, subfolder="scheduler")

    # --------------------------------------------------------------------------
    # B. Convert the UNet model to a LoRA model
    # --------------------------------------------------------------------------
    lora_config = LoraConfig(
        r=LORA_RANK,
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ],  # LoRA will be applied to these layers
        init_lora_weights="gaussian",  # Initialize LoRA weights with a Gaussian distribution
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()  # Print trainable parameters

    # --------------------------------------------------------------------------
    # C. Set up optimizer and data
    # --------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    dataset = DermatofibromaDataset()  # Loads only DF images (512x512)
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
    )
    # Move models to device
    unet.to(DEVICE)
    text_encoder.to(DEVICE)
    vae.to(DEVICE)

    # --------------------------------------------------------------------------
    # D. Training Loop
    # --------------------------------------------------------------------------
    unet.train()
    for step in tqdm(range(TRAIN_STEPS), desc="Fine-tuning"):
        # 1) Sample a batch (images, prompts)
        try:
            images, prompts = next(iter(train_loader))
        except StopIteration:
            train_iter = iter(train_loader)
            images, prompts = next(train_iter)

        # 2) Tokenize text (prompts)
        input_ids = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
        text_embeddings = text_encoder(input_ids)[0]  # shape [B, seq_len, hid_dim]

        # 3) Convert PIL images to Tensors and move to device
        # Ensuring images have shape [B, 3, 512, 512] in float32
        images_np = np.stack(
            [np.array(img) for img in images]
        )  # shape [B, 512, 512, 3]
        images_np = np.moveaxis(images_np, -1, 1)  # => [B, 3, 512, 512]
        images_torch = torch.from_numpy(images_np).float().to(DEVICE) / 255.0

        # 4) Encode images into latents with the VAE
        latents_dist = vae.encode(images_torch).latent_dist
        latents = latents_dist.sample() * 0.18215  # Scale factor for SD latents

        # 5) Add random noise to latents for diffusion
        noise = torch.randn_like(latents)
        timesteps = (
            torch.randint(0, noise_scheduler.num_train_timesteps, (BATCH_SIZE,))
            .long()
            .to(DEVICE)
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 6) Predict the noise using the LoRA-enabled U-Net
        noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

        # 7) Compute MSE loss between predicted noise and the actual noise
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # 8) Backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --------------------------------------------------------------------------
    # E. Save LoRA weights
    # --------------------------------------------------------------------------
    unet.save_pretrained(MODEL_SAVE_PATH)
    print(f"Saved fine-tuned LoRA weights to {MODEL_SAVE_PATH}")


# ------------------------------------------------------------------------------
# 3) Generating Synthetic Images
# ------------------------------------------------------------------------------
def generate_synthetic_dermatofibroma(num_images=50):
    """
    Generate synthetic dermatofibroma images using the fine-tuned Stable Diffusion model.
    Args:
        num_images (int): Number of synthetic images to generate.
    """
    # A. Load the base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,  # Disable safety checker for medical images
    )

    # B. Load the LoRA weights if we've fine-tuned them
    if os.path.exists(MODEL_SAVE_PATH):
        pipe.unet.load_attn_procs(MODEL_SAVE_PATH)
        print("Loaded fine-tuned LoRA weights into the pipeline.")

    # Move pipeline to device
    pipe.to(DEVICE)

    # C. Generation loop
    for i in range(num_images):
        result = pipe(
            PROMPT_TEMPLATE,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=IMAGE_SIZE,  # 512
            width=IMAGE_SIZE,  # 512
        )
        image = result.images[0]

        # D. Save the generated image
        save_path = os.path.join(SYNTHETIC_PATH, f"synthetic_df_{i+1}.jpg")
        image.save(save_path)
        print(f"Generated synthetic image: {save_path}")


def main():
    """
    Main function to fine-tune Stable Diffusion with LoRA, then generate synthetic images.
    """
    finetune_stable_diffusion_dermatofibroma()  # Step 1: Fine-tune
    generate_synthetic_dermatofibroma(num_images=50)  # Step 2: Generate images


if __name__ == "__main__":
    main()
