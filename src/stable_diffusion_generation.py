import os
import torch
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# PEFT / LoRA
from peft import LoraConfig, get_peft_model

# Hugging Face / Diffusers
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.loaders import LoraLoaderMixin
from transformers import CLIPTextModel, CLIPTokenizer

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
HF_MODEL_ID = "runwayml/stable-diffusion-v1-5"  # Base Stable Diffusion
SYNTHETIC_PATH = "data/synthetic/images_dermatofibroma/"
MODEL_SAVE_PATH = "models/stable_diffusion_lora/"
os.makedirs(SYNTHETIC_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Training config
TRAIN_STEPS = 1000
LORA_RANK = 4
BATCH_SIZE = 2
ACCUMULATION_STEPS = 4
LR = 1e-5
EPOCHS = 2  # Example: two epochs

IMAGE_SIZE = 512
PROMPT_TEMPLATE = (
    "Dermatofibroma lesion under dermoscopy, clinical photograph, high quality"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ------------------------------------------------------------------------------
# Debug Functions
# ------------------------------------------------------------------------------
def debug_file_paths():
    """Check that a sample image file exists at the specified path."""
    image_dir = "data/processed_sd/images/"
    sample_image = "ISIC_0025366.jpg"
    image_path = os.path.join(image_dir, sample_image)
    if os.path.exists(image_path):
        print(f"✅ File found: {image_path}")
    else:
        print(f"❌ File NOT found: {image_path}")


def debug_metadata():
    """Print metadata information to confirm filtering of dermatofibroma images."""
    import pandas as pd

    metadata_path = "data/raw/HAM10000_metadata.csv"
    df_metadata = pd.read_csv(metadata_path)
    df_filtered = df_metadata[df_metadata["dx"] == "df"]
    print(f"✅ Found {len(df_filtered)} dermatofibroma images in metadata.")
    print("Sample image IDs:", df_filtered["image_id"].head(10).tolist())


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
class DermatofibromaDataset(Dataset):
    """
    Loads 'df' (dermatofibroma) images from `data/processed_sd/images/`, each with a text prompt.
    """

    def __init__(
        self,
        metadata_path="data/raw/HAM10000_metadata.csv",
        image_dir="data/processed_sd/images/",
    ):
        import pandas as pd

        self.metadata = pd.read_csv(metadata_path)
        self.df_metadata = self.metadata[self.metadata["dx"] == "df"]
        self.image_dir = image_dir
        self.prompt = PROMPT_TEMPLATE

        # Check if each row's corresponding image actually exists
        self.valid_rows = []
        for idx in range(len(self.df_metadata)):
            image_id = self.df_metadata.iloc[idx]["image_id"]
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            if os.path.exists(image_path):
                self.valid_rows.append(idx)

        print(f"Found {len(self.valid_rows)} valid DF images for training.")

    def __len__(self):
        return len(self.valid_rows)

    def __getitem__(self, idx):
        real_idx = self.valid_rows[idx]
        image_id = self.df_metadata.iloc[real_idx]["image_id"]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        return image, self.prompt


def collate_fn(batch):
    images, prompts = zip(*batch)
    return list(images), list(prompts)


# ------------------------------------------------------------------------------
# Fine-tuning function
# ------------------------------------------------------------------------------
def finetune_stable_diffusion_dermatofibroma():
    """
    Fine-tune Stable Diffusion on the dermatofibroma class using LoRA via PEFT,
    then rename the file (pytorch_model.bin or adapter_model.bin) to 'pytorch_lora_weights.bin'
    so diffusers can load it with load_attn_procs().
    """
    # 1) Load base models
    unet = UNet2DConditionModel.from_pretrained(HF_MODEL_ID, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(HF_MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(HF_MODEL_ID, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(HF_MODEL_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(HF_MODEL_ID, subfolder="scheduler")

    # 2) LoRA configs (no init_lora_weights => default)
    lora_config_unet = LoraConfig(
        r=LORA_RANK,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    lora_config_text = LoraConfig(
        r=LORA_RANK,
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
    )

    # Convert them to LoRA
    unet = get_peft_model(unet, lora_config_unet)
    text_encoder = get_peft_model(text_encoder, lora_config_text)

    unet.to(DEVICE)
    text_encoder.to(DEVICE)
    vae.to(DEVICE)

    # 3) Optimizer
    params_to_optimize = list(unet.parameters()) + list(text_encoder.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=LR)

    # 4) Data
    dataset = DermatofibromaDataset()
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # 5) Training Loop
    global_step = 0
    unet.train()
    text_encoder.train()

    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch+1}/{EPOCHS}...")
        for images, prompts in tqdm(train_loader, desc="Training"):
            # Tokenize text
            input_ids = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(DEVICE)
            text_embeddings = text_encoder(input_ids)[0]

            # Convert images to latents
            images_np = np.stack([np.array(img) for img in images])
            images_np = np.moveaxis(images_np, -1, 1)
            images_torch = torch.from_numpy(images_np).float().to(DEVICE) / 255.0

            with torch.no_grad():
                latents_dist = vae.encode(images_torch).latent_dist
                latents = latents_dist.sample() * 0.18215

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

            # Compute MSE loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()

            if (global_step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step >= TRAIN_STEPS:
                break

        if global_step >= TRAIN_STEPS:
            break

    # 6) Save LoRA weights
    unet_dir = os.path.join(MODEL_SAVE_PATH, "unet_lora")
    text_dir = os.path.join(MODEL_SAVE_PATH, "text_encoder_lora")
    unet.save_pretrained(unet_dir)
    text_encoder.save_pretrained(text_dir)
    print("LoRA fine-tuning complete; weights saved.")

    # 7) Rename possible PEFT output (pytorch_model.bin or adapter_model.bin)
    rename_peft_to_diffusers_lora(unet_dir)
    rename_peft_to_diffusers_lora(text_dir)


def rename_peft_to_diffusers_lora(folder: str):
    """
    1) PEFT might produce 'pytorch_model.bin' or 'adapter_model.bin'.
    2) Diffusers expects 'pytorch_lora_weights.bin'.
    We'll rename whichever file we find to the required name.
    """
    possible_files = ["pytorch_model.bin", "adapter_model.bin"]
    lora_model_path = os.path.join(folder, "pytorch_lora_weights.bin")

    found_file = None
    for candidate in possible_files:
        candidate_path = os.path.join(folder, candidate)
        if os.path.exists(candidate_path):
            found_file = candidate_path
            break

    if found_file:
        # If a 'pytorch_lora_weights.bin' is already there, remove it first
        if os.path.exists(lora_model_path):
            os.remove(lora_model_path)
        os.rename(found_file, lora_model_path)
        print(f"Renamed {found_file} -> {lora_model_path}")
    else:
        print(f"No LoRA bin file found in {folder}; skipping rename.")


# ------------------------------------------------------------------------------
# Generate function
# ------------------------------------------------------------------------------
def generate_synthetic_dermatofibroma(num_images=50):
    """
    Load the fine-tuned LoRA pipeline from models/stable_diffusion_lora/
    and generate synthetic images of dermatofibroma.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,
    ).to(DEVICE)

    unet_lora_path = os.path.join(MODEL_SAVE_PATH, "unet_lora")
    text_encoder_lora_path = os.path.join(MODEL_SAVE_PATH, "text_encoder_lora")

    # If these directories exist, load the LoRA weights
    if os.path.isdir(unet_lora_path):
        pipe.unet.load_attn_procs(unet_lora_path)
    else:
        print(f"Warning: {unet_lora_path} not found. Using base UNet weights.")

    if os.path.isdir(text_encoder_lora_path):
        pipe.text_encoder.load_attn_procs(text_encoder_lora_path)
    else:
        print(
            f"Warning: {text_encoder_lora_path} not found. Using base text encoder weights."
        )

    print("Loaded fine-tuned LoRA (UNet + Text Encoder) if available.")

    pipe.enable_attention_slicing()

    for i in range(num_images):
        result = pipe(
            PROMPT_TEMPLATE,
            num_inference_steps=100,
            guidance_scale=8.0,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        )
        image = result.images[0]
        save_path = os.path.join(SYNTHETIC_PATH, f"synthetic_df_{i+1}.jpg")
        image.save(save_path)
        print(f"Generated synthetic image: {save_path}")


# ------------------------------------------------------------------------------
# Main and Argument Parser
# ------------------------------------------------------------------------------
def main():
    finetune_stable_diffusion_dermatofibroma()
    generate_synthetic_dermatofibroma(num_images=50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stable Diffusion Fine-tuning & Generation"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run debug checks and exit."
    )
    args = parser.parse_args()

    if args.debug:
        print("Running debug checks...")
        debug_file_paths()
        debug_metadata()
    else:
        main()
