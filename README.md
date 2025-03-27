# Skin Lesion Image Generation with Stable Diffusion

This repository contains scripts to fine-tune Stable Diffusion for generating synthetic skin lesion images from the HAM10000 dataset. The code allows you to fine-tune the model on specific lesion classes and generate new synthetic images.

## Project Structure

```
mlp-cw4/
├── data/
│   ├── raw/
│   │   ├── images/                    # Original HAM10000 images (part1 & part2 merged)
│   │   └── HAM10000_metadata.csv      # Labels + metadata for HAM10000
│   ├── processed/
│   │   └── images/                    # Resized images (224x224, for example)
│   ├── processed_sd/
│   │   └── images/                    # Resized images (512x512)
│   └── synthetic/
│       └── images_dermatofibroma/     # Synthetic images generated
├── data_preparation.py                # Script to prepare data for fine-tuning
├── train_lora.py                      # Script to fine-tune the model using LoRA
├── generate_images.py                 # Script to generate synthetic images
├── pipeline.py                        # Script for full end-to-end pipeline
├── evaluate_images.py                 # Script for image evaluation
├── visualise_loss.py                  # Script for visualising training loss
├── requirements.txt                   # Required packages
└── README.md                          # Instructions
```

## Understanding the Code

### 1. Data Preparation (`data_preparation.py`)

This script prepares the HAM10000 dataset for fine-tuning Stable Diffusion:

- It reads the `HAM10000_metadata.csv` to get labels for each image
- It organizes the processed images (512x512) from your `processed_sd/images` folder
- It creates text prompts for each image (e.g., "dermatofibroma skin lesion")
- It can prepare data for all lesion types or focus on a specific class

### 2. LoRA Fine-tuning (`train_lora.py`)

This script fine-tunes Stable Diffusion using Low-Rank Adaptation (LoRA):

- LoRA is a parameter-efficient fine-tuning method that significantly reduces memory requirements
- It only trains a small number of parameters while keeping most of the model frozen
- It applies LoRA to both the UNet and text encoder components of Stable Diffusion
- It supports mixed precision training for faster training times
- It saves checkpoints throughout training

### 3. Image Generation (`generate_images.py`)

This script generates synthetic skin lesion images using the fine-tuned model:

- It loads the base Stable Diffusion model and applies the LoRA weights
- It generates images based on a prompt template for the specified skin lesion class
- It allows customization of generation parameters (guidance scale, steps, etc.)
- It saves the generated images to the specified output directory

### 4. End-to-End Pipeline (`pipeline.py`)

This script runs the complete workflow from data preparation to image generation:

- It calls the appropriate functions from the other scripts
- It manages all intermediate outputs
- It provides a simple interface for running the full pipeline with a single command

## Installation

1. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: End-to-End Pipeline

For convenience, you can run the entire pipeline with a single command:

```bash
python pipeline.py --specific_class dermatofibroma --num_epochs 50 --num_images 50
```

This will:

1. Prepare the dataset for the specified class
2. Fine-tune the model
3. Generate the requested number of synthetic images

### Option 2: Step-by-Step Approach

If you prefer more control, you can run each step separately:

#### 1. Prepare Data

```bash
python data_preparation.py --specific_class dermatofibroma
```

This creates a dataset focused on the specified lesion class.

#### 2. Fine-tune the Model

```bash
python train_lora.py --class_name dermatofibroma --num_epochs 50
```

Key parameters:

- `--model_id`: Base Stable Diffusion model (default: "runwayml/stable-diffusion-v1-5")
- `--class_name`: Lesion class to train on
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--batch_size`: Batch size (default: 1)
- `--output_dir`: Directory to save the model

#### 3. Generate Synthetic Images

```bash
python generate_images.py --class_name dermatofibroma --num_images 50
```

Key parameters:

- `--lora_model_path`: Path to the fine-tuned LoRA model
- `--class_name`: Lesion class to generate
- `--num_images`: Number of images to generate
- `--output_dir`: Directory to save generated images
- `--seed`: Random seed for reproducibility
- `--guidance_scale`: Controls how much the image generation follows the text prompt

## Running on Google Cloud

To run this project on Google Cloud, you can use either Google Colab or a VM instance with a GPU.

### Option 1: Google Colab

1. Upload the scripts to Google Drive
2. Create a new Colab notebook
3. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Navigate to your project directory:
   ```
   %cd /content/drive/MyDrive/path/to/project
   ```
5. Install requirements:
   ```
   !pip install -r requirements.txt
   ```
6. Run the pipeline script:
   ```
   !python pipeline.py --specific_class dermatofibroma --num_epochs 50 --num_images 50
   ```

### Option 2: Google Cloud VM

1. Create a VM instance with GPU:

   - Go to Compute Engine > VM instances
   - Click "Create instance"
   - Select a GPU type (e.g., NVIDIA T4 or V100)
   - Choose a Deep Learning VM Image
   - Set boot disk size (at least 50GB recommended)

2. SSH into the VM and set up the project:

   ```bash
   # Clone or upload your code
   git clone [your-repo-url]
   cd [your-repo-directory]

   # Install dependencies
   pip install -r requirements.txt
   ```

3. Run the pipeline:

   ```bash
   python pipeline.py --specific_class dermatofibroma --num_epochs 50 --num_images 50
   ```

4. For large training runs, consider using `screen` or `tmux` to keep the process running even if you disconnect:

   ```bash
   # Start a new screen session
   screen -S training

   # Run your training
   python pipeline.py --specific_class dermatofibroma --num_epochs 100 --num_images 100

   # Detach from screen (training continues): Ctrl+A, then D
   # Reattach later: screen -r training
   ```

### Optimizing for Google Cloud

1. **Increase batch size**: If using a powerful GPU like V100 or A100, increase the batch size and adjust learning rate accordingly.

   ```bash
   python train_lora.py --batch_size 4 --learning_rate 5e-5
   ```

2. **Use mixed precision**: The default is already fp16, but you can confirm:

   ```bash
   python train_lora.py --mixed_precision fp16
   ```

3. **Save storage**: If you're limited on disk space, clean up checkpoints after training:

   ```bash
   # Keep only the final model
   find models/lora -type d -name "checkpoint-*" -exec rm -rf {} +
   find models/lora -type d -name "epoch-*" -exec rm -rf {} +
   ```

4. **Monitor resources**:

   ```bash
   # Monitor GPU usage
   watch -n 1 nvidia-smi

   # Monitor disk space
   df -h
   ```

## Customization Tips

1. **Prompt engineering**: You can modify the prompt template for better results:

   ```bash
   python generate_images.py --prompt_template "{class_name} skin lesion, dermatoscopic image, high resolution, medical photography"
   ```

2. **Fine-tune on multiple classes**: If you want the model to learn the differences between classes:

   ```bash
   # First prepare all classes
   python data_preparation.py
   # Then train (without specifying a class)
   python train_lora.py --num_epochs 100
   ```

3. **Generation parameters**: Adjust generation parameters for different results:

   ```bash
   python generate_images.py --class_name dermatofibroma --guidance_scale 9 --num_inference_steps 75
   ```

4. **Create variants**: Generate multiple sets with different seeds:
   ```bash
   python generate_images.py --class_name dermatofibroma --num_images 20 --seed 42 --output_dir "data/synthetic_set1"
   python generate_images.py --class_name dermatofibroma --num_images 20 --seed 123 --output_dir "data/synthetic_set2"
   ```

## Troubleshooting

1. **Out of memory errors**: Reduce batch size, use gradient accumulation, or switch to a different precision mode:

   ```bash
   python train_lora.py --batch_size 1 --gradient_accumulation_steps 8
   ```

2. **Slow training**: Ensure you're using a GPU; check with:

   ```bash
   python -c "import torch; print('GPU available:', torch.cuda.is_available())"
   ```

3. **Poor image quality**: Increase training epochs, adjust the guidance scale, or try different prompts:

   ```bash
   python generate_images.py --guidance_scale 8.5 --num_inference_steps 70
   ```

4. **File not found errors**: Check paths and ensure the data structure matches expectations:
   ```bash
   # Verify files exist
   ls -la data/processed_sd/images
   cat data/raw/HAM10000_metadata.csv | head
   ```
