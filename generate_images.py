import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm import tqdm
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_images(
    model_id="runwayml/stable-diffusion-v1-5",
    lora_model_path="models/lora/final",
    class_name=None,
    num_images=10,
    output_dir="data/synthetic",
    prompt_template="{class_name} skin lesion, dermatology image, high quality",
    negative_prompt="low quality, blurry, distorted, deformed, unrealistic",
    guidance_scale=7.5,
    num_inference_steps=50,
    seed=None,
    height=512,
    width=512,
):
    """
    Generate synthetic skin lesion images using fine-tuned Stable Diffusion.

    Args:
        model_id: HuggingFace model ID for Stable Diffusion
        lora_model_path: Path to LoRA model weights
        class_name: Lesion class to generate
        num_images: Number of images to generate
        output_dir: Directory to save generated images
        prompt_template: Template for image generation prompt
        negative_prompt: Negative prompt for image generation
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of inference steps
        seed: Random seed for generation
        height: Height of generated images
        width: Width of generated images
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    # Try to load model info if class name is not specified
    if class_name is None:
        model_info_path = os.path.join(lora_model_path, "model_info.txt")
        if os.path.exists(model_info_path):
            with open(model_info_path, "r") as f:
                for line in f:
                    if line.startswith("Class:"):
                        class_name = line.split(":", 1)[1].strip()
                        break

    if class_name is None:
        logger.error(
            "Please specify a class name using --class_name or make sure model_info.txt exists"
        )
        return

    # Configure output directory
    output_subdir = os.path.join(output_dir, f"images_{class_name.lower()}")
    os.makedirs(output_subdir, exist_ok=True)

    logger.info(f"Generating {num_images} images for class: {class_name}")

    # Load base model
    logger.info(f"Loading base model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,  # Disable safety checker for medical images
    )

    # Load LoRA weights for UNet
    logger.info(f"Loading LoRA weights from: {lora_model_path}")
    unet_lora_path = os.path.join(lora_model_path, "unet_lora")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora_path)

    # Load LoRA weights for text encoder if available
    text_encoder_lora_path = os.path.join(lora_model_path, "text_encoder_lora")
    if os.path.exists(text_encoder_lora_path):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_lora_path
        )

    # Move pipeline to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Generate images
    prompt = prompt_template.format(class_name=class_name)
    logger.info(f"Using prompt: {prompt}")
    logger.info(f"Using negative prompt: {negative_prompt}")

    # Create a file to track generation parameters
    params_file = os.path.join(output_subdir, "generation_params.txt")
    with open(params_file, "w") as f:
        f.write(f"Class: {class_name}\n")
        f.write(f"Base model: {model_id}\n")
        f.write(f"LoRA model: {lora_model_path}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Negative prompt: {negative_prompt}\n")
        f.write(f"Guidance scale: {guidance_scale}\n")
        f.write(f"Inference steps: {num_inference_steps}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Height: {height}\n")
        f.write(f"Width: {width}\n")

    # Generate images with progress bar
    for i in tqdm(range(num_images), desc="Generating images"):
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
        ).images[0]

        # Save image
        image_path = os.path.join(output_subdir, f"{class_name.lower()}_{i+1:04d}.png")
        image.save(image_path)

    logger.info(
        f"Generated {num_images} images for class '{class_name}' at: {output_subdir}"
    )
    return output_subdir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic skin lesion images"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="models/lora/final",
        help="Path to LoRA model weights",
    )
    parser.add_argument(
        "--class_name", type=str, default=None, help="Lesion class to generate"
    )
    parser.add_argument(
        "--num_images", type=int, default=10, help="Number of images to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="{class_name} skin lesion, dermatology image, high quality",
        help="Prompt template",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, blurry, distorted, deformed, unrealistic",
        help="Negative prompt",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")

    args = parser.parse_args()
    generate_images(**vars(args))
