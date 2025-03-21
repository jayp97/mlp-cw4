# MLP-CW4: Dermatofibroma Overbalancing with Synthetic Images

This project tests the hypothesis:

> _By synthetically generating images for the dermatofibroma class (using fine-tuned Stable Diffusion / FLUX), we can overbalance that class in the HAM10000 dataset and improve classification accuracy in an EfficientNetV2 model._

We also measure how different real-to-synthetic ratios (e.g. 1:0, 1:0.5, 1:1, 1:1.5, 1:2) affect classification performance.

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
│
├── models/
│   ├── stable_diffusion_lora/         # (Optional) Fine-tuned SD LoRA weights
│   ├── efficientnet_checkpoints/      # Checkpoints for classification
│   └── final/                         # Final model(s)
│
├── src/
│   ├── data_preprocessing.py          # Resizing + metadata loading
│   ├── stable_diffusion_generation.py # Synthetic generation (fine-tuned SD / FLUX)
│   ├── classification.py              # EfficientNetV2 training/fine-tuning
│   ├── evaluation.py                  # Model evaluation
│   └── utils.py                       # (Optional) helper code
│
├── experiments/
│   └── ratio_experiments.py           # Runs multiple real:synthetic ratios
│
├── notebooks/
│   ├── Data_Preprocessing.ipynb
│   ├── Synthetic_Generation.ipynb
│   ├── Classification_Training.ipynb
│   ├── Evaluation.ipynb
│   └── Experiments.ipynb
│
├── environment.yml (or requirements.txt)
└── README.md
```

## Steps

1. **Data Preprocessing**

   - `python src/data_preprocessing.py`
   - Resizes all images in `data/raw/images/` to 224x224, saves in `data/processed/images/`.

2. **Synthetic Image Generation**

   - `python src/stable_diffusion_generation.py`
   - Optionally fine-tune stable diffusion on your DF images, then generate synthetic images stored in `data/synthetic/images_dermatofibroma/`.

3. **Classification**

   - `python src/classification.py` (or run the notebook)
   - Trains an EfficientNetV2 model with the specified ratio of real-to-synthetic DF images.

4. **Evaluation**

   - `python src/evaluation.py` (or run the notebook)
   - Prints overall accuracy and class-wise accuracy, focusing on DF.

5. **Experiments**
   - Vary the `synthetic_ratio` from 0 up to 2.0 in increments of 0.5.
   - Check the effect on DF accuracy (and overall accuracy).

## Environment

conda env create -f environment.yml conda activate mlp-cw4

## Notes

- You might need a GPU with sufficient VRAM (or Apple Silicon MPS) to fine-tune stable diffusion.
- The code uses `timm` to load a `tf_efficientnetv2_l.in21k` pretrained model.
- Adjust hyperparameters like image size, batch size, or number of epochs based on your compute resources.
