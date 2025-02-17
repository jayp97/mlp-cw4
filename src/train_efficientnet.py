# src/train_efficientnet.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

from efficientnet_pytorch import EfficientNet  # Alternatively: timm

# or from timm import create_model

PROCESSED_PATH = "data/processed/images/"
SYNTHETIC_PATH = "data/synthetic/images_dermatofibroma/"
METADATA_PATH = "data/raw/HAM10000_metadata.csv"
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "models/efficientnet_checkpoints/"


class SkinLesionDataset(Dataset):
    def __init__(self, img_dir, metadata_csv, transform=None, synthetic_ratio=0):
        """
        synthetic_ratio indicates how many synthetic images we want to add relative
        to the real images for the dermatofibroma (df) class.
        0 = no synthetic
        1 = equal amounts real:synthetic
        3 = 3 times synthetic, etc.
        """
        self.img_dir = img_dir
        self.transform = transform
        df = pd.read_csv(metadata_csv)

        # For each image_id, the .jpg is in data/processed/images/
        # The dx field has the class label
        df["filename"] = df["image_id"] + ".jpg"
        self.data = df

        self.synthetic_ratio = synthetic_ratio
        if synthetic_ratio > 0:
            # Only add synthetic for 'df' class
            df_df = self.data[self.data["dx"] == "df"]  # Dermatofibroma rows
            real_count = len(df_df)
            synthetic_needed = real_count * synthetic_ratio
            # Gather synthetic images
            synthetic_images = [
                os.path.join(SYNTHETIC_PATH, f)
                for f in os.listdir(SYNTHETIC_PATH)
                if f.endswith(".jpg")
            ]
            # If you have fewer synthetic images than needed, re-sample them
            synthetic_sample = synthetic_images * (
                int(synthetic_needed // len(synthetic_images)) + 1
            )
            synthetic_sample = synthetic_sample[: int(synthetic_needed)]

            # Build a small DataFrame for synthetic images
            synth_df = pd.DataFrame(
                {
                    "filename": [os.path.basename(x) for x in synthetic_sample],
                    "dx": ["df"] * len(synthetic_sample),
                }
            )
            self.data = pd.concat([self.data, synth_df], ignore_index=True)

        self.labels_map = {
            "akiec": 0,
            "bcc": 1,
            "bkl": 2,
            "df": 3,  # The class of interest
            "mel": 4,
            "nv": 5,
            "vasc": 6,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row["filename"]
        label_str = row["dx"]
        label = self.labels_map[label_str]

        # If synthetic image
        if os.path.exists(os.path.join(SYNTHETIC_PATH, filename)):
            img_path = os.path.join(SYNTHETIC_PATH, filename)
        else:
            img_path = os.path.join(self.img_dir, filename)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(synthetic_ratio=0):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # ensure size
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # typical ImageNet stats
        ]
    )

    dataset = SkinLesionDataset(
        img_dir=PROCESSED_PATH,
        metadata_csv=METADATA_PATH,
        transform=transform,
        synthetic_ratio=synthetic_ratio,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load EfficientNetV2L (you may switch to timm if needed)
    model = EfficientNet.from_pretrained(
        "efficientnet-b7"
    )  # closest to V2-L or use timm's 'efficientnet_v2_l'

    # Adjust final layer for 7 classes
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 7)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    checkpoint_path = os.path.join(
        MODEL_SAVE_PATH, f"efficientnet_synth_{synthetic_ratio}.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    # Example: train with 0 synthetic first
    train_model(synthetic_ratio=0)
