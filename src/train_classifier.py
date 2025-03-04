#!/usr/bin/env python
# coding=utf-8
"""
train_classifier.py

Trains a skin lesion classifier using real and/or synthetic images.
The CSV file (for training and validation) contains metadata including image IDs and labels.
Images are assumed to be stored together in a single folder and are named as {image_id}.jpg or .png.

Optionally, you can filter by a given lesion code.

Example Usage:
python train_classifier.py \
  --train_csv="mlp-cw4/data/raw/HAM10000_metadata.csv" \
  --val_csv="mlp-cw4/data/raw/HAM10000_metadata.csv" \
  --image_dir="mlp-cw4/data/processed_sd/images" \
  --label_column="dx" \
  --lesion_code="df" \
  --batch_size=32 \
  --epochs=20 \
  --arch="vgg16" \
  --lr=1e-3 \
  --output_dir="mlp-cw4/models/final_classifier"
"""
import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="CSV file containing metadata for training images.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="CSV file containing metadata for validation images.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing all images (named as {image_id}.jpg or .png).",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="dx",
        help="Column name in CSV that indicates lesion label.",
    )
    parser.add_argument(
        "--lesion_code",
        type=str,
        default=None,
        help="If provided, only use rows where the label equals this code.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--arch", type=str, default="vgg16", choices=["vgg16", "resnet18", "vit_b_16"]
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./classifier_ckpt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class SkinLesionDataset(Dataset):
    def __init__(
        self, csv_file, image_dir, label_column="dx", lesion_code=None, transform=None
    ):
        self.csv = pd.read_csv(csv_file)
        if lesion_code is not None:
            self.csv = self.csv[
                self.csv[label_column].str.lower() == lesion_code.lower()
            ]
        self.image_dir = image_dir
        self.transform = transform
        self.labels = sorted(self.csv[label_column].unique())
        self.label2idx = {lbl: i for i, lbl in enumerate(self.labels)}

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        image_id = row["image_id"]
        label_str = row["dx"]
        label_idx = self.label2idx[label_str]
        img_path_jpg = os.path.join(self.image_dir, f"{image_id}.jpg")
        img_path_png = os.path.join(self.image_dir, f"{image_id}.png")
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            raise FileNotFoundError(f"Image {image_id} not found as jpg or png.")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = SkinLesionDataset(
        csv_file=args.train_csv,
        image_dir=args.image_dir,
        label_column=args.label_column,
        lesion_code=args.lesion_code,
        transform=train_transform,
    )
    val_dataset = SkinLesionDataset(
        csv_file=args.val_csv,
        image_dir=args.image_dir,
        label_column=args.label_column,
        lesion_code=args.lesion_code,
        transform=val_transform,
    )

    label_counts = train_dataset.csv[args.label_column].value_counts()
    label2weight = {lbl: 1.0 / count for lbl, count in label_counts.items()}
    sample_weights = [label2weight[row] for row in train_dataset.csv[args.label_column]]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    num_classes = len(train_dataset.labels)
    if args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif args.arch == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for images, labels in tqdm(train_loader, desc="Train"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Val"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels.data)
        val_loss /= len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, "best_model.pth")
            )
            print("New best model saved.")

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
