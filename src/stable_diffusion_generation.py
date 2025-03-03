import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Hugging Face / Diffusers
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    LoRAAttnProcessor,
    AttnProcessor2_0,
)
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
TRAIN_STEPS = 1000  # total steps before stopping
BATCH_SIZE = 2
ACCUMULATION_STEPS = 4  # gradient accumulation
LR = 1e-5
EPOCHS = 2  # example: 2 epochs
LORA_RANK = 4  # rank for LoRA
LORA_ALPHA = 1  # scaling factor for LoRA

IMAGE_SIZE = 512
PROMPT_TEMPLATE = (
    "Dermatofibroma lesion under dermoscopy, clinical photograph, high quality"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ------------------------------------------------------------------------------
# Debug Functions (optional)
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
    """Print metadata info to confirm filtering of dermatofibroma images."""
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
    Loads 'df' (dermatofibroma) images from `data/processed_sd/images/`,
    each with a text prompt.
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

        # Verify that each dermatofibroma image actually exists
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
# Functions to inject LoRA layers into UNet / Text Encoder
# ------------------------------------------------------------------------------
def enable_lora_for_unet(unet, r=4, alpha=1):
    """
    Replaces AttnProcessor2_0 with LoRAAttnProcessor in all cross/self-attn modules
    within the UNet.
    """
    for name, module in unet.attn_processors.items():
        if isinstance(module, AttnProcessor2_0):
            # The module knows its own hidden/cross_attention dimensions
            hidden_size = module.out_dim
            cross_attention_dim = module.out_dim

            # Replace it with a LoRAAttnProcessor:
            unet.attn_processors[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=r,
                lora_alpha=alpha,
            )


def enable_lora_for_text_encoder(text_encoder, r=4, alpha=1):
    """
    For each self-attention module in the text encoder,
    replace with LoRAAttnProcessor.
    """
    # For CLIPTextModel, each layer has .self_attn or similar
    # We'll do a simple approach: check submodules for AttnProcessor2_0.
    for name, submodule in text_encoder.named_modules():
        if isinstance(submodule, AttnProcessor2_0):
            hidden_size = submodule.out_dim
            cross_attention_dim = submodule.out_dim
            parent_module = text_encoder.get_submodule(name.rsplit(".", 1)[0])
            # e.g. parent might be a CLIPEncoderLayer

            # Replace the entire submodule key with LoRAAttnProcessor
            # Not all text encoder modules have it, so be safe:
            if hasattr(parent_module, "attn_processor"):
                parent_module.attn_processor = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=r,
                    lora_alpha=alpha,
                )


# ------------------------------------------------------------------------------
# Fine-tuning function
# ------------------------------------------------------------------------------
def finetune_stable_diffusion_dermatofibroma():
    """
    Fine-tune (LoRA) the Stable Diffusion UNet (and text encoder, optional)
    on the dermatofibroma class. LoRA weights get saved to the specified folders
    in a format recognized by Diffusers' load_attn_procs().
    """
    # 1) Load the base models
    print("Loading UNet, VAE, Text Encoder...")
    unet = UNet2DConditionModel.from_pretrained(HF_MODEL_ID, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(HF_MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(HF_MODEL_ID, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(HF_MODEL_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(HF_MODEL_ID, subfolder="scheduler")

    unet.to(DEVICE)
    text_encoder.to(DEVICE)
    vae.to(DEVICE)

    # 2) Inject LoRA layers into UNet and Text Encoder
    enable_lora_for_unet(unet, r=LORA_RANK, alpha=LORA_ALPHA)
    enable_lora_for_text_encoder(text_encoder, r=LORA_RANK, alpha=LORA_ALPHA)

    unet.train()
    text_encoder.train()

    # 3) Prepare an optimizer for the newly introduced LoRA parameters
    #    Usually we freeze the base model weights and train only LoRA
    #    modules. For simplicity, let's do that explicitly:
    lora_params = []
    for param in unet.parameters():
        if param.requires_grad:
            lora_params.append(param)
    for param in text_encoder.parameters():
        if param.requires_grad:
            lora_params.append(param)

    optimizer = torch.optim.AdamW(lora_params, lr=LR)

    # 4) Data
    dataset = DermatofibromaDataset()
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    global_step = 0
    print("Starting training...")

    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch+1}/{EPOCHS}...")
        for images, prompts in tqdm(train_loader, desc="Training"):
            # 4a) Tokenize text
            input_ids = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(DEVICE)
            # text encoder forward
            text_embeddings = text_encoder(input_ids)[0]

            # 4b) Convert images to latents
            images_np = np.stack([np.array(img) for img in images])  # shape [B,H,W,C]
            images_np = np.moveaxis(images_np, -1, 1)  # -> [B,C,H,W]
            images_torch = torch.from_numpy(images_np).float().to(DEVICE) / 255.0

            with torch.no_grad():
                latents_dist = vae.encode(images_torch).latent_dist
                latents = latents_dist.sample() * 0.18215

            # 4c) Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 4d) UNet predicts the noise
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

            # 4e) MSE loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()

            # 4f) Gradient accumulation
            if (global_step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step >= TRAIN_STEPS:
                break

        if global_step >= TRAIN_STEPS:
            break

    print("Training complete.")

    # 5) Save LoRA weights in diffusers format
    unet_out_dir = os.path.join(MODEL_SAVE_PATH, "unet_lora")
    text_out_dir = os.path.join(MODEL_SAVE_PATH, "text_encoder_lora")

    unet.save_attn_procs(unet_out_dir)
    text_encoder.save_attn_procs(text_out_dir)

    print(
        f"LoRA fine-tuning complete; weights saved to:\n {unet_out_dir}\n {text_out_dir}"
    )


# ------------------------------------------------------------------------------
# Generate function
# ------------------------------------------------------------------------------
def generate_synthetic_dermatofibroma(num_images=50):
    """
    Load the fine-tuned LoRA pipeline from models/stable_diffusion_lora/
    and generate synthetic images of dermatofibroma.
    """
    print("Loading base pipeline for generation...")
    pipe = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,  # WARNING: disables the built-in safety checker
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
        print(f"Warning: {text_encoder_lora_path} not found. Using base text weights.")

    print("Loaded LoRA (UNet + Text Encoder) if available.")
    pipe.enable_attention_slicing()

    os.makedirs(SYNTHETIC_PATH, exist_ok=True)

    for i in range(num_images):
        result = pipe(
            PROMPT_TEMPLATE,
            num_inference_steps=50,  # can adjust steps
            guidance_scale=7.5,  # can adjust guidance
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
        description="Stable Diffusion Fine-tuning & Generation with LoRA"
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
