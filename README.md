# MLP-CW4 Project

This repository demonstrates a complete pipeline for:

1. **Data Preprocessing**

   - Resizes and organizes HAM10000 images into `data/processed/images/`.

2. **Synthetic Image Generation**

   - Optionally fine-tunes a Stable Diffusion model (LoRA or DreamBooth)
   - Generates synthetic dermatofibroma images into `data/synthetic/images_dermatofibroma/`.

3. **Classification with EfficientNetV2**

   - Uses timm (`tf_efficientnetv2_l.in21k`) to train on real + synthetic images.

4. **Evaluation**
   - Reports overall and per-class accuracy on the test set.

## Project Structure

```
mlp-cw4/ ├── data/ │ ├── raw/ │ │ └── images/ # HAM10000 images + metadata │ ├── processed/ │ └── synthetic/ │ └── images_dermatofibroma/ ├── models/ │ ├── stable_diffusion_lora/ │ ├── efficientnet_checkpoints/ │ └── final/ ├── src/ │ ├── data_preprocessing.py │ ├── stable_diffusion_generation.py │ ├── classification.py │ ├── evaluation.py │ └── utils.py ├── notebooks/ ├── environment.yml └── README.md
```
