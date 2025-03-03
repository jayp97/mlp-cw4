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
# Configuration constants
# ------------------------------------------------------------------------------
HF_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"  # Base Stable Diffusion model
SYNTHETIC_PATH = "data/synthetic/images_dermatofibroma/"  # Where generated images go
MODEL_SAVE_PATH = "models/stable_diffusion_lora/"  # Where LoRA weights are saved
os.makedirs(SYNTHETIC_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Training config
# For a quick test, you can set TRAIN_STEPS to 100. Then, if everything works as expected,
# run with 1000 steps to overwrite the existing weights manually.
TRAIN_STEPS = 100
LORA_RANK = 4  # Dimension controlling the size/effect of LoRA updates
BATCH_SIZE = 1  # Batch size for fine-tuning; typically small for LoRA

# For stable diffusion v1.5, 512x512 is the typical resolution
IMAGE_SIZE = 512

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
    Custom collate function for the DataLoader.
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
        image_dir="data/processed_sd/images/",  # Using 512x512 images
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
# Helper: rename keys from the PEFT "base_model.model" prefix to match Diffusers
# ------------------------------------------------------------------------------
def rename_peft_unet_keys(state_dict):
    """
    PEFT saves LoRA weights under a 'base_model.model.*' prefix.
    Diffusers expects the UNet's keys to start at e.g. 'down_blocks.0...'
    without 'unet.' or 'base_model.model.'. This function strips the prefix.
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
    If you also trained LoRA on the text encoder with PEFT, you'd need
    to strip the same prefix for text encoder layers.
    (Your code currently doesn't train text encoder, so it may be unused.)
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
    Manually load LoRA weights from the PEFT-saved structure:
      - unet_lora/adapter_model.safetensors
      - text_encoder_lora/adapter_model.safetensors
    Remaps the key prefixes and loads them into pipe.unet and pipe.text_encoder.
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

    # Text Encoder LoRA (only if you fine-tuned it)
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
def finetune_stable_diffusion_dermatofibroma():
    """
    Fine-tune Stable Diffusion on the dermatofibroma class using LoRA.
    The resulting LoRA weights are saved to MODEL_SAVE_PATH.
    """

    # A. Load Stable Diffusion components from the base checkpoint
    unet = UNet2DConditionModel.from_pretrained(HF_MODEL_ID, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(HF_MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(HF_MODEL_ID, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(HF_MODEL_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(HF_MODEL_ID, subfolder="scheduler")

    # Freeze text encoder to keep prompt embeddings stable
    for param in text_encoder.parameters():
        param.requires_grad = False

    # B. Convert the UNet model to a LoRA model
    lora_config = LoraConfig(
        r=LORA_RANK,
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ],  # LoRA will be applied to these attention sub-layers
        init_lora_weights="gaussian",  # Initialize LoRA weights with a Gaussian distribution
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()  # Print trainable parameters

    # C. Set up optimizer and data
    optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-5)
    dataset = DermatofibromaDataset()  # Loads only DF images (512x512)
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
    )

    # Move models to device
    unet.to(DEVICE)
    text_encoder.to(DEVICE)
    vae.to(DEVICE)

    # Setup mixed precision scaler if using CUDA
    scaler = torch.amp.GradScaler() if DEVICE == "cuda" else None

    # D. Training Loop with persistent DataLoader iterator and loss logging
    unet.train()
    train_iter = iter(train_loader)

    for step in tqdm(range(TRAIN_STEPS), desc="Fine-tuning"):
        try:
            images, prompts = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, prompts = next(train_iter)

        # 1) Tokenize text (prompts)
        input_ids = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(DEVICE)

        # Obtain text embeddings (text encoder is frozen)
        text_embeddings = text_encoder(input_ids)[0]  # shape [B, seq_len, hid_dim]

        # 2) Convert PIL images to Tensors and move to device
        images_np = np.stack([np.array(img) for img in images])  # [B, 512, 512, 3]
        images_np = np.moveaxis(images_np, -1, 1)  # [B, 3, 512, 512]
        images_torch = torch.from_numpy(images_np).float().to(DEVICE) / 255.0

        # 3) Encode images into latents with the VAE
        latents_dist = vae.encode(images_torch).latent_dist
        latents = latents_dist.sample() * 0.18215  # Scale factor for SD latents

        # 4) Add random noise to latents for diffusion
        noise = torch.randn_like(latents)
        timesteps = (
            torch.randint(0, noise_scheduler.config.num_train_timesteps, (BATCH_SIZE,))
            .long()
            .to(DEVICE)
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 5) Predict the noise using the LoRA-enabled UNet
        if DEVICE == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
        else:
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # 6) Backpropagation and optimizer step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Logging training loss every 50 steps
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    # E. Save LoRA weights
    # This saves out a folder containing LoRA weights in "unet_lora/adapter_model.safetensors"
    # plus the config, plus (optionally) text encoder LoRA if you also had trained that.
    unet.save_pretrained(MODEL_SAVE_PATH)
    print(f"Saved fine-tuned LoRA weights to {MODEL_SAVE_PATH}")


# ------------------------------------------------------------------------------
# 3) Generating Synthetic Images with Enhanced Inference Settings
# ------------------------------------------------------------------------------
def generate_synthetic_dermatofibroma(num_images=50):
    """
    Generate synthetic dermatofibroma images using the fine-tuned Stable Diffusion model.
    Args:
        num_images (int): Number of synthetic images to generate.
    """
    # A. Load the base pipeline with use_safetensors enabled.
    pipe = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,  # Disable safety checker for medical images
        use_safetensors=True,
    )

    # B. Load the fine-tuned LoRA weights (remapped for Diffusers).
    #    We do NOT call pipe.load_lora_weights(...) because that expects
    #    a different naming convention. Instead, we manually rename and load.
    load_peft_lora_weights(pipe, MODEL_SAVE_PATH, device=DEVICE)

    # Move pipeline to device (in case it isn't already)
    pipe.to(DEVICE)

    # C. Generate images with increased inference steps for better quality
    for i in range(num_images):
        result = pipe(
            PROMPT_TEMPLATE,
            num_inference_steps=150,  # Increased inference steps from 100 to 150
            guidance_scale=8.0,  # Adjust guidance scale as needed
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
    check_required_data()  # Ensure necessary data files and directories are accessible
    finetune_stable_diffusion_dermatofibroma()  # Step 1: Fine-tune
    generate_synthetic_dermatofibroma(num_images=50)  # Step 2: Generate images


if __name__ == "__main__":
    main()
