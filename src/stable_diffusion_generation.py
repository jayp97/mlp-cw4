import os
import torch
import cv2
import numpy as np
import random
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
from safetensors.torch import load_file as safe_load_file


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
# Label mapping: convert short label codes to full names
# ------------------------------------------------------------------------------
LABEL_MAP = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "nv": "Melanocytic Nevus",
    "mel": "Melanoma",
    "vasc": "Vascular Lesion",
}

# ------------------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------------------
HF_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"  # Base Stable Diffusion model
SYNTHETIC_PATH = "data/synthetic/images_dermatofibroma/"  # Where generated images go
MODEL_SAVE_PATH = "models/stable_diffusion_lora/"  # Where LoRA weights are saved
os.makedirs(SYNTHETIC_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Training config
TRAIN_STEPS = 1000
LORA_RANK = 4
BATCH_SIZE = 1

# For stable diffusion v1.5, 512x512 is the typical resolution
IMAGE_SIZE = 512

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
    Custom collate function for the DataLoader.
    This function unzips the batch of (image, prompt) tuples into two lists.
    """
    images, prompts = zip(*batch)
    return list(images), list(prompts)


# ------------------------------------------------------------------------------
# 1) Custom Dataset for Fine-Tuning (using all images)
# ------------------------------------------------------------------------------
class AllLesionDataset(Dataset):
    """
    Loads all HAM10000 images from 'data/processed_sd/images/' and incorporates
    the full lesion label in the text prompt.
    """

    def __init__(
        self,
        metadata_path="data/raw/HAM10000_metadata.csv",
        image_dir="data/processed_sd/images/",  # Using 512x512 images
    ):
        import pandas as pd

        # Load metadata for all lesions
        self.metadata = pd.read_csv(metadata_path)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row["image_id"]
        dx_label = row["dx"].lower()  # e.g., "df"
        # Map the short label to its full name if available
        full_label = LABEL_MAP.get(dx_label, dx_label)

        # Load image and convert to RGB
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        # Incorporate the full label directly into the prompt
        prompt = (
            f"A {full_label} lesion, dermoscopy image, high quality clinical photograph"
        )

        return image, prompt


# ------------------------------------------------------------------------------
# Helper: rename keys from the PEFT "base_model.model" prefix to match Diffusers
# ------------------------------------------------------------------------------
def rename_peft_unet_keys(state_dict):
    """
    Strips the 'base_model.model.' prefix from the state dict keys.
    """
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("base_model.model."):
            new_key = k.replace("base_model.model.", "")
        else:
            new_key = k
        new_sd[new_key] = v
    return new_sd


def rename_peft_text_encoder_keys(state_dict):
    """
    Strips the 'base_model.model.' prefix from the text encoder state dict keys.
    """
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("base_model.model."):
            new_key = k.replace("base_model.model.", "")
        else:
            new_key = k
        new_sd[new_key] = v
    return new_sd


def load_peft_lora_weights(pipe, lora_folder: str, device="cuda"):
    """
    Manually load LoRA weights from the PEFT-saved structure for both the UNet and,
    optionally, the text encoder.
    """
    # UNet LoRA
    unet_lora_path = os.path.join(lora_folder, "unet_lora", "adapter_model.safetensors")
    if os.path.isfile(unet_lora_path):
        unet_state_dict = safe_load_file(unet_lora_path, device="cpu")
        unet_state_dict = rename_peft_unet_keys(unet_state_dict)
        pipe.unet.load_state_dict(unet_state_dict, strict=False)
        print(f"Loaded UNet LoRA weights from {unet_lora_path}")
    else:
        print(f"No UNet LoRA file found at {unet_lora_path}, using base UNet.")

    # Text Encoder LoRA (if available)
    text_enc_lora_path = os.path.join(
        lora_folder, "text_encoder_lora", "adapter_model.safetensors"
    )
    if os.path.isfile(text_enc_lora_path):
        text_enc_state_dict = safe_load_file(text_enc_lora_path, device="cpu")
        text_enc_state_dict = rename_peft_text_encoder_keys(text_enc_state_dict)
        pipe.text_encoder.load_state_dict(text_enc_state_dict, strict=False)
        print(f"Loaded Text Encoder LoRA weights from {text_enc_lora_path}")
    else:
        print(
            f"No Text Encoder LoRA file found at {text_enc_lora_path}, using base text encoder."
        )

    pipe.to(device)


# ------------------------------------------------------------------------------
# 2) Fine-Tuning with LoRA and Mixed Precision Training
# ------------------------------------------------------------------------------
def finetune_stable_diffusion():
    """
    Fine-tune Stable Diffusion on all HAM10000 lesion images using LoRA.
    The resulting LoRA weights are saved to MODEL_SAVE_PATH.
    """

    # A. Load base components from the checkpoint
    unet = UNet2DConditionModel.from_pretrained(HF_MODEL_ID, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(HF_MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(HF_MODEL_ID, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(HF_MODEL_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(HF_MODEL_ID, subfolder="scheduler")

    # Freeze the text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False

    # B. Convert UNet to a LoRA model
    lora_config = LoraConfig(
        r=LORA_RANK,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # C. Set up optimizer and data
    optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-5)
    dataset = AllLesionDataset()  # Use entire dataset with full labels
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
    )

    # Move models to device
    unet.to(DEVICE)
    text_encoder.to(DEVICE)
    vae.to(DEVICE)

    scaler = torch.amp.GradScaler() if DEVICE == "cuda" else None

    # D. Training loop
    unet.train()
    train_iter = iter(train_loader)

    for step in tqdm(range(TRAIN_STEPS), desc="Fine-tuning"):
        try:
            images, prompts = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, prompts = next(train_iter)

        input_ids = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(DEVICE)

        text_embeddings = text_encoder(input_ids)[0]

        images_np = np.stack([np.array(img) for img in images])
        images_np = np.moveaxis(images_np, -1, 1)
        images_torch = torch.from_numpy(images_np).float().to(DEVICE) / 255.0

        latents_dist = vae.encode(images_torch).latent_dist
        latents = latents_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        timesteps = (
            torch.randint(0, noise_scheduler.config.num_train_timesteps, (BATCH_SIZE,))
            .long()
            .to(DEVICE)
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        if DEVICE == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
        else:
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    unet.save_pretrained(MODEL_SAVE_PATH)
    print(f"Saved fine-tuned LoRA weights to {MODEL_SAVE_PATH}")


# ------------------------------------------------------------------------------
# 3) Generating Synthetic Images with Enhanced Inference Settings
# ------------------------------------------------------------------------------
def generate_synthetic_images(num_images=20):
    """
    Generate synthetic skin lesion images using the fine-tuned Stable Diffusion model.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,
        use_safetensors=True,
    )

    load_peft_lora_weights(pipe, MODEL_SAVE_PATH, device=DEVICE)
    pipe.to(DEVICE)

    for i in range(num_images):
        result = pipe(
            "A Dermatofibroma lesion, dermoscopy image, high quality clinical photograph",
            num_inference_steps=150,
            guidance_scale=8.0,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        )
        image = result.images[0]
        save_path = os.path.join(SYNTHETIC_PATH, f"synthetic_image_{i+1}.jpg")
        image.save(save_path)
        print(f"Generated synthetic image: {save_path}")


# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------
def main():
    """
    Main function to fine-tune Stable Diffusion with LoRA, then generate synthetic images.
    """
    check_required_data()
    finetune_stable_diffusion()
    generate_synthetic_images(20)


if __name__ == "__main__":
    main()
