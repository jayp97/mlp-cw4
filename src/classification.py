"""
classification.py

This script handles classification training with EfficientNetV2 (via timm).
It combines real images (processed) and optional synthetic images for
the dermatofibroma class.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

import timm
import timm.data


class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset that can optionally include synthetic dermatofibroma images.
    """

    def __init__(
        self,
        img_dir="data/processed/images/",
        metadata_csv="data/raw/HAM10000_metadata.csv",
        transform=None,
        synthetic_ratio=0,
        synthetic_dir="data/synthetic/images_dermatofibroma/",
    ):
        self.img_dir = img_dir
        self.synthetic_dir = synthetic_dir
        self.transform = transform

        # Load metadata
        df = pd.read_csv(metadata_csv)
        df["filename"] = df["image_id"] + ".jpg"
        self.data = df

        # If synthetic_ratio > 0, replicate synthetic images for 'df' class
        self.synthetic_ratio = synthetic_ratio
        if synthetic_ratio > 0:
            df_df = self.data[self.data["dx"] == "df"]
            real_count = len(df_df)
            synthetic_needed = int(real_count * synthetic_ratio)

            # Gather synthetic images
            synthetic_images = [
                os.path.join(self.synthetic_dir, f)
                for f in os.listdir(self.synthetic_dir)
                if f.endswith(".jpg")
            ]
            if len(synthetic_images) == 0:
                print(f"Warning: No synthetic images found in {self.synthetic_dir}")

            # Resample if needed
            expanded_list = (
                synthetic_images
                * ((synthetic_needed // max(1, len(synthetic_images))) + 1)
            )[:synthetic_needed]

            # Create a DataFrame for synthetic images
            synth_df = pd.DataFrame(
                {
                    "filename": [os.path.basename(x) for x in expanded_list],
                    "dx": ["df"] * len(expanded_list),
                }
            )

            # Merge
            self.data = pd.concat([self.data, synth_df], ignore_index=True)

        # Map labels to numeric classes
        self.labels_map = {
            "akiec": 0,
            "bcc": 1,
            "bkl": 2,
            "df": 3,  # Dermatofibroma
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

        synthetic_path = os.path.join(self.synthetic_dir, filename)
        if os.path.exists(synthetic_path):
            img_path = synthetic_path
        else:
            img_path = os.path.join(self.img_dir, filename)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def train_efficientnetv2(
    synthetic_ratio=0,
    num_epochs=5,
    batch_size=16,
    learning_rate=1e-4,
    checkpoint_name="efficientnet_v2.pth",
):
    """
    Train an EfficientNetV2-L model using timm. The synthetic_ratio param
    indicates how many synthetic 'df' images to add relative to real df images.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model using timm (EfficientNetV2-L pretrained on ImageNet-21k)
    model = timm.create_model(
        "tf_efficientnetv2_l.in21k", pretrained=True, num_classes=7
    )
    model.to(device)

    # Setup transforms from timm
    data_config = timm.data.resolve_data_config({}, model=model)
    train_transform = timm.data.create_transform(**data_config, is_training=True)

    # Create dataset & dataloader
    dataset = SkinLesionDataset(
        transform=train_transform, synthetic_ratio=synthetic_ratio
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs("models/efficientnet_checkpoints/", exist_ok=True)
    ckpt_path = os.path.join("models/efficientnet_checkpoints/", checkpoint_name)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


def main():
    """
    If running this file directly, we'll train the model with default params.
    """
    train_efficientnetv2(
        synthetic_ratio=1,
        num_epochs=5,
        batch_size=16,
        learning_rate=1e-4,
        checkpoint_name="efficientnet_v2_synth_1.pth",
    )


if __name__ == "__main__":
    main()
