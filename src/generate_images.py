import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import PeftModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm import tqdm
import random
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_images(
    model_id="runwayml/stable-diffusion-v1-5",
    lora_model_path="models/lora/final",
    class_name=None,
    num_images=10,
    output_dir="data/synthetic",
    prompt_template="{class_name} skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, medical imaging, sharp focus",
    negative_prompt="low quality, blurry, distorted, deformed, unrealistic, cartoon, drawing, painting, watermark, text, signature, border, frame, error, cropped, amateur, pixelated, jpeg artifacts, oversaturated",
    guidance_scale=7.5,
    num_inference_steps=75,
    seed=None,
    height=512,
    width=512,
    scheduler_type="ddim",
    lora_scale=0.9,
    use_test_prompts=False,
    output_format="png",
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
        scheduler_type: Type of scheduler to use (ddim or pndm)
        lora_scale: Scale factor for LoRA weights
        use_test_prompts: Whether to use different test prompts
        output_format: Output image format (png or jpg)
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        # Also set numpy seed for full reproducibility
        np.random.seed(seed)
        # Set torch cuda seed if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Set torch to deterministic mode if possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    # Use float16 precision for memory efficiency
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,  # Disable safety checker for medical images
    )

    # Set scheduler
    if scheduler_type.lower() == "ddim":
        logger.info("Using DDIM scheduler")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type.lower() == "pndm":
        logger.info("Using PNDM scheduler (default)")
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using default")

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
        logger.info("Loaded LoRA weights for text encoder")

    # Move pipeline to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Set LoRA scale factor
    if hasattr(pipe, "set_adapters_scale"):
        pipe.set_adapters_scale(lora_scale)
        logger.info(f"Set LoRA scale factor to {lora_scale}")

    # Generate multiple prompt variations if testing
    if use_test_prompts:
        # Create a set of different prompts to test which ones work best
        prompts = [
            prompt_template.format(class_name=class_name),
            f"{class_name} skin lesion, dermatological photo, high quality",
            f"{class_name} skin lesion, medical photograph, detailed",
            f"{class_name} dermatology lesion, clinical image, sharp focus",
            f"close-up photo of {class_name} skin lesion, medical imaging",
            f"high detail {class_name} lesion on skin, dermatology image",
            f"clinical photograph of {class_name}, dermatology",
            f"{class_name} on human skin, medical documentation, high resolution",
            f"detailed view of {class_name} lesion, dermatology examination",
            f"medical image of {class_name}, dermatological assessment",
        ]

        # Create a set of different negative prompts to test
        negative_prompts = [
            negative_prompt,
            "low quality, blurry, distorted, unrealistic",
            "cartoon, drawing, illustration, painting, artificial",
            "poor lighting, overexposed, underexposed, cropped",
            "text, watermark, signature, border",
        ]

        # Adjust CFG values to test
        cfg_values = [6.5, 7.0, 7.5, 8.0, 8.5]

        # Ensure at least one image is generated with each combination
        combinations = min(
            num_images, len(prompts) * len(negative_prompts) * len(cfg_values)
        )
        logger.info(
            f"Testing {combinations} different prompt/negative prompt/cfg combinations"
        )
    else:
        # Use a single prompt template
        prompts = [prompt_template.format(class_name=class_name)]
        negative_prompts = [negative_prompt]
        cfg_values = [guidance_scale]

    # Generate images
    logger.info(f"Using prompt template: {prompt_template}")
    logger.info(f"Using negative prompt: {negative_prompt}")
    logger.info(f"Guidance scale: {guidance_scale}")
    logger.info(f"Inference steps: {num_inference_steps}")

    # Create a file to track generation parameters
    params_file = os.path.join(output_subdir, "generation_params.txt")
    with open(params_file, "w") as f:
        f.write(f"Class: {class_name}\n")
        f.write(f"Base model: {model_id}\n")
        f.write(f"LoRA model: {lora_model_path}\n")
        f.write(f"Prompt template: {prompt_template}\n")
        f.write(f"Negative prompt: {negative_prompt}\n")
        f.write(f"Guidance scale: {guidance_scale}\n")
        f.write(f"Inference steps: {num_inference_steps}\n")
        f.write(f"Scheduler: {scheduler_type}\n")
        f.write(f"LoRA scale: {lora_scale}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Height: {height}\n")
        f.write(f"Width: {width}\n")

        if use_test_prompts:
            f.write("\nTest prompts:\n")
            for i, prompt in enumerate(prompts):
                f.write(f"{i+1}: {prompt}\n")

            f.write("\nTest negative prompts:\n")
            for i, neg_prompt in enumerate(negative_prompts):
                f.write(f"{i+1}: {neg_prompt}\n")

            f.write("\nTest CFG values:\n")
            for i, cfg in enumerate(cfg_values):
                f.write(f"{i+1}: {cfg}\n")

    # Generate images with progress bar
    for i in tqdm(range(num_images), desc="Generating images"):
        if use_test_prompts:
            # Cycle through prompts, negative prompts, and cfg values
            prompt_idx = i % len(prompts)
            neg_prompt_idx = (i // len(prompts)) % len(negative_prompts)
            cfg_idx = (i // (len(prompts) * len(negative_prompts))) % len(cfg_values)

            prompt = prompts[prompt_idx]
            neg_prompt = negative_prompts[neg_prompt_idx]
            cfg = cfg_values[cfg_idx]

            # Add testing info to filename
            suffix = f"_p{prompt_idx+1}_n{neg_prompt_idx+1}_c{cfg_idx+1}"
        else:
            prompt = prompts[0]
            neg_prompt = negative_prompts[0]
            cfg = cfg_values[0]
            suffix = ""

        # Set the seed for this specific image
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed + i)
        else:
            generator = None

        # Generate image
        image = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            guidance_scale=cfg,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        # Save image with appropriate format
        if output_format.lower() == "jpg":
            image_path = os.path.join(
                output_subdir, f"{class_name.lower()}_{i+1:04d}{suffix}.jpg"
            )
            image.save(image_path, quality=95)
        else:
            image_path = os.path.join(
                output_subdir, f"{class_name.lower()}_{i+1:04d}{suffix}.png"
            )
            image.save(image_path)

        # Save image metadata
        metadata_path = os.path.join(
            output_subdir, f"{class_name.lower()}_{i+1:04d}{suffix}_metadata.txt"
        )
        with open(metadata_path, "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Negative prompt: {neg_prompt}\n")
            f.write(f"CFG: {cfg}\n")
            f.write(f"Inference steps: {num_inference_steps}\n")
            f.write(f"Seed: {seed + i if seed is not None else 'random'}\n")

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
        default="{class_name} skin lesion, dermatology image, high quality, close-up, clinical photograph, detailed skin texture, medical imaging, sharp focus",
        help="Prompt template",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, blurry, distorted, deformed, unrealistic, cartoon, drawing, painting, watermark, text, signature, border, frame, error, cropped",
        help="Negative prompt",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=75, help="Number of inference steps"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "pndm"],
        help="Scheduler type (ddim or pndm)",
    )
    parser.add_argument(
        "--lora_scale", type=float, default=0.9, help="Scale factor for LoRA weights"
    )
    parser.add_argument(
        "--test_prompts", action="store_true", help="Test different prompt variations"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="Output image format",
    )

    args = parser.parse_args()
    generate_images(
        model_id=args.model_id,
        lora_model_path=args.lora_model_path,
        class_name=args.class_name,
        num_images=args.num_images,
        output_dir=args.output_dir,
        prompt_template=args.prompt_template,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        height=args.height,
        width=args.width,
        scheduler_type=args.scheduler,
        lora_scale=args.lora_scale,
        use_test_prompts=args.test_prompts,
        output_format=args.output_format,
    )
