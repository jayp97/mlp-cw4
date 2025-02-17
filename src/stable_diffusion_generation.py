"""
stable_diffusion_generation.py

This script demonstrates how to generate synthetic images for the
'dermatofibroma' class using a Stable Diffusion pipeline. It includes
a placeholder for fine-tuning with LoRA or DreamBooth, if desired.
"""

import os
import torch
import cv2
import numpy as np

# Example: Using Hugging Face diffusers
try:
    from diffusers import StableDiffusionPipeline

    # from diffusers import StableDiffusionLoRAPipeline  # If you want LoRA training
except ImportError:
    print(
        "Please install diffusers via 'pip install diffusers' if you want to run stable diffusion code."
    )

# Example model ID (public stable diffusion checkpoint)
HF_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Define paths
SYNTHETIC_PATH = "data/synthetic/images_dermatofibroma/"
MODEL_SAVE_PATH = "models/stable_diffusion_lora/"


def finetune_stable_diffusion_dermatofibroma():
    """
    Placeholder function to show where you'd fine-tune stable diffusion on
    dermatofibroma images using LoRA or DreamBooth. Actual code depends on
    huggingface/diffusers specifics.
    """
    print(
        "[INFO] Fine-tuning placeholder. Implement your LoRA/DreamBooth training here."
    )
    # e.g., load pipeline, train, save weights to MODEL_SAVE_PATH


def generate_synthetic_dermatofibroma(
    num_images=50, prompt="Dermatofibroma lesion, dermoscopy image"
):
    """
    Generate synthetic dermatofibroma images using a (optionally) fine-tuned pipeline.
    By default, loads the standard stable diffusion v1.5 pipeline.
    """
    os.makedirs(SYNTHETIC_PATH, exist_ok=True)

    # If you had a fine-tuned pipeline, you would load it here:
    # pipeline = StableDiffusionLoRAPipeline.from_pretrained(MODEL_SAVE_PATH, torch_dtype=torch.float16)
    # For demonstration, use the base pipeline:
    pipeline = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.float16
    )

    # Set device
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
    # 1. (Optionally) fine-tune the pipeline
    finetune_stable_diffusion_dermatofibroma()
    # 2. Generate synthetic images
    generate_synthetic_dermatofibroma(num_images=50)


if __name__ == "__main__":
    main()
