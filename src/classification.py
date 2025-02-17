"""
classification.py

Fine-tunes an EfficientNetV2 model (via timm) on the HAM10000 dataset.
Also optionally adds synthetic dermatofibroma images to overbalance that class.
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
    Custom PyTorch Dataset that:
      - Reads from data/processed/images (real images)
      - Optionally mixes in synthetic dermatofibroma images at a certain ratio
    """

    def __init__(
        self,
        img_dir="data/processed/images/",
        metadata_csv="data/raw/HAM10000_metadata.csv",
        transform=None,
        synthetic_ratio=0.0,
        synthetic_dir="data/synthetic/images_dermatofibroma/",
    ):
        """
        synthetic_ratio: float, e.g. 0.5, 1.0, 1.5, 2.0, etc.
          If 1.0 => for each real DF image, add 1 synthetic DF image
          If 0.5 => for each real DF image, add 0.5 synthetic DF images (rounded down or up)
          etc.
        """
        self.img_dir = img_dir
        self.synthetic_dir = synthetic_dir
        self.transform = transform

        df = pd.read_csv(metadata_csv)
        # The CSV should have 'image_id' plus 'dx' columns
        df["filename"] = df["image_id"] + ".jpg"
        self.data = df

        self.synthetic_ratio = synthetic_ratio
        if synthetic_ratio > 0:
            # Only add synthetic images for the 'df' class
            df_df = self.data[self.data["dx"] == "df"]
            real_count = len(df_df)
            # Number of synthetic images to add
            synthetic_needed = int(round(real_count * synthetic_ratio))

            synthetic_images = [
                os.path.join(self.synthetic_dir, f)
                for f in os.listdir(self.synthetic_dir)
                if f.endswith(".jpg")
            ]
            if len(synthetic_images) == 0:
                print("[WARNING] No synthetic images found in", self.synthetic_dir)

            # If we don't have enough synthetic images, we'll replicate them
            # so that we have 'synthetic_needed' total
            factor = (synthetic_needed // max(len(synthetic_images), 1)) + 1
            extended_list = synthetic_images * factor
            extended_list = extended_list[:synthetic_needed]  # trim to exact count

            # Build a small DataFrame for these synthetic samples
            synth_df = pd.DataFrame(
                {
                    "filename": [os.path.basename(x) for x in extended_list],
                    "dx": ["df"] * len(extended_list),
                }
            )
            self.data = pd.concat([self.data, synth_df], ignore_index=True)

        # Map labels to numeric indices
        self.labels_map = {
            "akiec": 0,
            "bcc": 1,
            "bkl": 2,
            "df": 3,
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

        # If the file is synthetic
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
    synthetic_ratio=0.0,
    num_epochs=5,
    batch_size=16,
    learning_rate=1e-4,
    checkpoint_name="efficientnet_v2.pth",
):
    """
    Trains an EfficientNetV2-L model using the timm library, optionally adding
    synthetic dermatofibroma images at the specified ratio.

    Args:
      synthetic_ratio (float): e.g., 0.0 (no synthetic), 1.0, 1.5, 2.0
      num_epochs (int)
      batch_size (int)
      learning_rate (float)
      checkpoint_name (str)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model: tf_efficientnetv2_l.in21k => pretrained on ImageNet-21k
    model = timm.create_model(
        "tf_efficientnetv2_l.in21k", pretrained=True, num_classes=7
    )
    model.to(device)

    # Create data transforms from timm
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
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_loss:.4f}")

    # Save the checkpoint
    os.makedirs("models/efficientnet_checkpoints/", exist_ok=True)
    ckpt_path = os.path.join("models/efficientnet_checkpoints/", checkpoint_name)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


def main():
    """
    If run directly, we'll do a sample training with ratio=0.0 (i.e. no synthetic).
    """
    train_efficientnetv2(
        synthetic_ratio=0.0,
        num_epochs=5,
        batch_size=16,
        learning_rate=1e-4,
        checkpoint_name="efficientnet_v2_synth_0.0.pth",
    )


if __name__ == "__main__":
    main()
