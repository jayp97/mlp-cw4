import os
import torch
import cv2
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
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
HF_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SYNTHETIC_PATH = "data/synthetic/images_dermatofibroma/"
MODEL_SAVE_PATH = "models/stable_diffusion_lora/"
os.makedirs(SYNTHETIC_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Training config
TRAIN_STEPS = 1000
LORA_RANK = 4  # Not used because older LoRAAttnProcessor doesn't accept it
BATCH_SIZE = 2
ACCUMULATION_STEPS = 4
LR = 1e-5
EPOCHS = 2

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
    image_dir = "data/processed_sd/images/"
    sample_image = "ISIC_0025366.jpg"
    image_path = os.path.join(image_dir, sample_image)
    if os.path.exists(image_path):
        print(f"✅ File found: {image_path}")
    else:
        print(f"❌ File NOT found: {image_path}")


def debug_metadata():
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
# LoRA Implementation
# ------------------------------------------------------------------------------
def inject_lora(unet, text_encoder):
    """
    Inject LoRA into the UNet and Text Encoder.
    To avoid the error 'LoRAAttnProcessor2_0' object has no attribute '__call__',
    we force any instance of LoRAAttnProcessor2_0 to be replaced by the callable
    LoRAAttnProcessor.
    """
    # For UNet: replace all attention processors with LoRAAttnProcessor.
    new_processors = {}
    for name in unet.attn_processors.keys():
        new_processors[name] = LoRAAttnProcessor()
    unet.set_attn_processor(new_processors)
    # Additionally, traverse all submodules and if any processor is of type
    # LoRAAttnProcessor2_0, replace it.
    for module in unet.modules():
        if hasattr(module, "processor") and isinstance(
            module.processor, LoRAAttnProcessor2_0
        ):
            module.processor = LoRAAttnProcessor()

    # For text encoder: replace any processor of type LoRAAttnProcessor2_0.
    for name, module in text_encoder.named_modules():
        if hasattr(module, "processor") and isinstance(
            module.processor, LoRAAttnProcessor2_0
        ):
            module.processor = LoRAAttnProcessor()
    # Also, for any attention layers that lack a processor, assign one.
    for name, module in text_encoder.named_modules():
        if "attention.self" in name and "text_model.encoder.layers" in name:
            if not hasattr(module, "processor") or module.processor is None:
                module.processor = LoRAAttnProcessor()


def finetune_stable_diffusion_dermatofibroma():
    # Load base models
    unet = UNet2DConditionModel.from_pretrained(HF_MODEL_ID, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(HF_MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(HF_MODEL_ID, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(HF_MODEL_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(HF_MODEL_ID, subfolder="scheduler")

    # Inject LoRA (with processor replacement)
    inject_lora(unet, text_encoder)

    unet.to(DEVICE)
    text_encoder.to(DEVICE)
    vae.to(DEVICE)

    # Setup optimizer
    params_to_optimize = [
        {"params": [p for p in unet.parameters() if p.requires_grad]},
        {"params": [p for p in text_encoder.parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=LR)

    # Prepare data
    dataset = DermatofibromaDataset()
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Training loop
    global_step = 0
    print("Starting training...")
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

            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()

            # Optimize
            if (global_step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step >= TRAIN_STEPS:
                break

        if global_step >= TRAIN_STEPS:
            break

    # Save LoRA weights
    print("Saving LoRA weights...")
    unet_lora_path = os.path.join(MODEL_SAVE_PATH, "unet_lora")
    text_lora_path = os.path.join(MODEL_SAVE_PATH, "text_encoder_lora")
    os.makedirs(unet_lora_path, exist_ok=True)
    os.makedirs(text_lora_path, exist_ok=True)

    unet.save_attn_procs(unet_lora_path)

    text_lora_state_dict = {}
    for name, module in text_encoder.named_modules():
        if hasattr(module, "processor") and isinstance(
            module.processor, (LoRAAttnProcessor, LoRAAttnProcessor2_0)
        ):
            text_lora_state_dict.update(module.processor.state_dict(name))
    torch.save(
        text_lora_state_dict, os.path.join(text_lora_path, "pytorch_lora_weights.bin")
    )

    print("Training complete!")


# ------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------
def generate_synthetic_dermatofibroma(num_images=50):
    print("Initializing pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,
    ).to(DEVICE)

    # Load LoRA weights
    unet_lora_path = os.path.join(MODEL_SAVE_PATH, "unet_lora")
    text_lora_path = os.path.join(MODEL_SAVE_PATH, "text_encoder_lora")

    if os.path.exists(unet_lora_path):
        pipe.unet.load_attn_procs(unet_lora_path)
    if os.path.exists(os.path.join(text_lora_path, "pytorch_lora_weights.bin")):
        pipe.load_lora_weights(text_lora_path)

    print("Generating images...")
    os.makedirs(SYNTHETIC_PATH, exist_ok=True)
    for i in range(num_images):
        result = pipe(
            PROMPT_TEMPLATE,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        )
        image = result.images[0]
        save_path = os.path.join(SYNTHETIC_PATH, f"synthetic_df_{i+1}.jpg")
        image.save(save_path)
        print(f"Saved: {save_path}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    finetune_stable_diffusion_dermatofibroma()
    generate_synthetic_dermatofibroma(num_images=50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stable Diffusion LoRA Training")
    parser.add_argument("--debug", action="store_true", help="Run debug checks")
    args = parser.parse_args()

    if args.debug:
        debug_file_paths()
        debug_metadata()
    else:
        main()
