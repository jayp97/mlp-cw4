"""
evaluation.py

Evaluates a trained EfficientNetV2 model on the entire HAM10000 dataset
(real images only) to measure class-wise and overall accuracy.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

import timm
import timm.data


class SkinLesionTestDataset(torch.utils.data.Dataset):
    """
    A dataset for evaluating the classifier on the real images only.
    """

    def __init__(
        self,
        img_dir="data/processed/images/",
        metadata_csv="data/raw/HAM10000_metadata.csv",
        transform=None,
    ):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(metadata_csv)
        df["filename"] = df["image_id"] + ".jpg"
        self.data = df

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

        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def evaluate_model(checkpoint_name="efficientnet_v2_synth_1.0.pth"):
    """
    Loads the specified checkpoint, evaluates on the entire dataset,
    and prints overall + per-class accuracies.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = timm.create_model(
        "tf_efficientnetv2_l.in21k", pretrained=True, num_classes=7
    )

    ckpt_path = os.path.join("models/efficientnet_checkpoints/", checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load the state dict
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create transforms for evaluation
    data_config = timm.data.resolve_data_config({}, model=model)
    eval_transform = timm.data.create_transform(**data_config, is_training=False)

    # Create dataset & dataloader
    test_dataset = SkinLesionTestDataset(transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    correct = 0
    total = 0
    num_classes = 7
    class_correct = np.zeros(num_classes, dtype=int)
    class_total = np.zeros(num_classes, dtype=int)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {checkpoint_name}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for l, p in zip(labels, preds):
                class_total[l] += 1
                if l == p:
                    class_correct[l] += 1

    overall_acc = 100.0 * correct / total
    print(f"\n[RESULT] Overall Accuracy: {overall_acc:.2f}% ({correct}/{total})")

    # Print per-class stats
    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    for i, cls_name in enumerate(class_names):
        if class_total[i] == 0:
            print(f"Class '{cls_name}': No samples found.")
        else:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(
                f"Class '{cls_name}': {acc:.2f}% ({class_correct[i]}/{class_total[i]})"
            )


def main():
    # Quick test
    evaluate_model("efficientnet_v2_synth_1.0.pth")


if __name__ == "__main__":
    main()
