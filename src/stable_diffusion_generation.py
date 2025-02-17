"""
stable_diffusion_generation.py

Demonstrates how to generate synthetic images for the dermatofibroma class,
using a base or fine-tuned Stable Diffusion pipeline.
"""

import os
import torch
import cv2
import numpy as np

# If using Hugging Face diffusers:
try:
    from diffusers import StableDiffusionPipeline

    # from diffusers import StableDiffusionLoRAPipeline  # If you specifically want LoRA
except ImportError:
    print("Please install 'diffusers' and 'transformers' to run stable diffusion code.")

# Example stable diffusion model ID
HF_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SYNTHETIC_PATH = "data/synthetic/images_dermatofibroma/"
MODEL_SAVE_PATH = "models/stable_diffusion_lora/"


def finetune_stable_diffusion_dermatofibroma():
    """
    Placeholder: Implement fine-tuning of stable diffusion (LoRA or DreamBooth)
    to generate more realistic dermatofibroma images.
    """
    print("[INFO] Fine-tuning placeholder. Add your LoRA or DreamBooth code here.")


def generate_synthetic_dermatofibroma(
    num_images=50, prompt="Dermatofibroma lesion, dermoscopy image"
):
    """
    Generate synthetic images using a stable diffusion pipeline.
    If you have a fine-tuned pipeline, load it instead of the base v1-5.
    """
    os.makedirs(SYNTHETIC_PATH, exist_ok=True)

    # If you had a fine-tuned pipeline, load it here:
    # pipeline = StableDiffusionLoRAPipeline.from_pretrained(MODEL_SAVE_PATH, torch_dtype=torch.float16)
    # For demonstration, just use base SD v1-5:
    pipeline = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.float16
    )

    # Choose device
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)

    for i in range(num_images):
        result = pipeline(prompt, guidance_scale=7.5, num_inference_steps=50)
        image_pil = result.images[0]

        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        save_path = os.path.join(SYNTHETIC_PATH, f"synthetic_df_{i+1}.jpg")
        cv2.imwrite(save_path, image_cv)
        print(f"Generated synthetic image: {save_path}")


def main():
    # 1) Fine-tune (optional / placeholder)
    finetune_stable_diffusion_dermatofibroma()
    # 2) Generate synthetic images
    generate_synthetic_dermatofibroma(num_images=50)


if __name__ == "__main__":
    main()
