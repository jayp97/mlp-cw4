"""
classification.py

Fine-tunes an EfficientNetV2 model (via timm) on the HAM10000 dataset.
Add info about synthetic here!
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim .lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
import timm
import timm.data


class SkinLesionDataset(Dataset):
    """
    Custom PyTorch Dataset that:
      - Reads from data/processed/images (real images in train, val, test splits)
    """

    def __init__(
        self,
        root_dir,                     # e.g. ../data/processed/images/train
        metadata_csv="../data/raw/HAM10000_metadata.csv",
        transform=None,
        synthetic_ratio=0.0,
        synthetic_dir="../data/synthetic/images_dermatofibroma/",
        is_train=False
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.synthetic_ratio = synthetic_ratio
        self.synthetic_dir = synthetic_dir

        # Load the metadata for ALL images
        df = pd.read_csv(metadata_csv)
        if "filename" not in df.columns:
            df["filename"] = df["image_id"] + ".jpg"

        # Build a dict mapping filename -> label
        filename_to_dx = dict(zip(df["filename"], df["dx"]))

        real_filenames = [
            f for f in os.listdir(root_dir) if f.endswith(".jpg")
        ]

        # Build our data list: (filename, label_str)
        data_list = []
        for fn in real_filenames:
            if fn in filename_to_dx:
                dx_str = filename_to_dx[fn]
                data_list.append((fn, dx_str))
            else:
                # The folder has a .jpg not in CSV
                # handle or skip as you like
                pass

        # Sythetic images add if training mode 
        if is_train and synthetic_ratio > 0:
            # Count how many DF images we have in data_list
            real_df_count = sum(1 for (_, dx) in data_list if dx == "df")
            synthetic_needed = int(round(real_df_count * synthetic_ratio))

            # Gather all synthetic images
            synthetic_files = [
                f for f in os.listdir(self.synthetic_dir) if f.endswith(".jpg")
            ]

            if len(synthetic_files) == 0 and synthetic_needed > 0:
                print(f"[WARNING] No synthetic images in {self.synthetic_dir}")

            if synthetic_needed > 0 and len(synthetic_files) > 0:
                factor = (synthetic_needed // len(synthetic_files)) + 1
                extended_list = synthetic_files * factor
                extended_list = extended_list[:synthetic_needed]

                for syn_fn in extended_list:
                    data_list.append((syn_fn, "df"))

        self.data_list = data_list


        self.label_map = {
            "akiec": 0, "bcc": 1, "bkl": 2, "df": 3,
            "mel": 4,  "nv": 5,  "vasc": 6
        }

        # Prints class distribution

        class_counts = {}
        for _, dx_str in self.data_list:
            class_counts[dx_str] = class_counts.get(dx_str, 0) + 1

        print(f"[INFO] Class distribution in {root_dir}:")
        for lesion_name in self.label_map.keys():
            count = class_counts.get(lesion_name, 0)
            print(f"  {lesion_name}: {count}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        fn, dx_str = self.data_list[idx]
        label = self.label_map[dx_str]

        # If it's a synthetic file (exists in synthetic_dir), load from there
        synthetic_path = os.path.join(self.synthetic_dir, fn)
        if os.path.exists(synthetic_path):
            img_path = synthetic_path
        else:
            # else it's a real file in root_dir
            img_path = os.path.join(self.root_dir, fn)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

def evaluate(model, dataloader, criterion, device="cpu"):
    """
    Runs one pass over a validation or test set, returns avg loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def evaluate_test_metrics(model, dataloader, device="cpu"):
    """
    Computes:
      - Overall Accuracy
      - AUC-ROC (per class)
      - Precision, Recall, F1 (per class)

    Returns a dict of metrics, and prints them nicely.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)              # shape (B,7)
            probs   = torch.softmax(outputs, 1)  # shape (B,7)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    all_probs  = np.concatenate(all_probs, axis=0)

    overall_acc = accuracy_score(all_labels, all_preds)

    # recision/Recall/F1 for each class
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0,1,2,3,4,5,6]
    )

    # AUC per class
    y_true_bin = label_binarize(all_labels, classes=[0,1,2,3,4,5,6])  # shape => (N,7)

    roc_aucs = []
    for c in range(7):
        if y_true_bin[:, c].sum() == 0:
            roc_aucs.append(float('nan'))
        else:
            auc_c = roc_auc_score(y_true_bin[:, c], all_probs[:, c])
            roc_aucs.append(auc_c)

    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    print("\n===== DETAILED TEST METRICS =====")
    print(f"Overall Accuracy: {overall_acc:.2f}")
    for i, cls_name in enumerate(class_names):
        print(f"\nClass '{cls_name}':")
        print(f"  Precision: {class_precision[i]:.2f}")
        print(f"  Recall:    {class_recall[i]:.2f}")
        print(f"  F1-Score:  {class_f1[i]:.2f}")
        if not np.isnan(roc_aucs[i]):
            print(f"  AUC:       {roc_aucs[i]:.2f}")
        else:
            print(f"  AUC:       N/A (no samples)")

    metrics_dict = {
        "overall_acc": overall_acc,
        "precision_per_class": class_precision,
        "recall_per_class": class_recall,
        "f1_per_class": class_f1,
        "auc_per_class": roc_aucs
    }
    return metrics_dict


def train_efficientnetv2(
    synthetic_ratio=0.0,
    num_epochs=5,
    batch_size=16,
    learning_rate=1e-4,
    checkpoint_name="efficientnet_v2.pth",
    train_dir="../data/processed/images/train",
    val_dir="../data/processed/images/val",
    test_dir="../data/processed/images/test",
):
    """
    Fine-tunes an EfficientNetV2-L model on train/val/test splits in separate folders.
    Optionally adds synthetic DF images to the train dataset with `synthetic_ratio`.

    Flow:
      - Create train/val/test datasets & loaders
      - Train with a validation pass each epoch
      - Evaluate on test at the end
      - Save final checkpoint
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(
        "tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=7
    )
    model.to(device)

    # Freeze all parameters except the classifier head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    # Data transforms , is_training False means no augmentation
    data_config    = timm.data.resolve_data_config({}, model=model)
    train_transform = timm.data.create_transform(**data_config, is_training=False)
    eval_transform  = timm.data.create_transform(**data_config, is_training=False)

    # Create Datasets
    train_dataset = SkinLesionDataset(
        root_dir=train_dir,
        metadata_csv="../data/raw/HAM10000_metadata.csv",
        transform=train_transform,
        synthetic_ratio=synthetic_ratio,  # Only train
        synthetic_dir="../data/synthetic/images_dermatofibroma/",
        is_train=True
    )
    val_dataset = SkinLesionDataset(
        root_dir=val_dir,
        metadata_csv="../data/raw/HAM10000_metadata.csv",
        transform=eval_transform,
        synthetic_ratio=0.0,    # no synthetic
        is_train=False
    )
    test_dataset = SkinLesionDataset(
        root_dir=test_dir,
        metadata_csv="../data/raw/HAM10000_metadata.csv",
        transform=eval_transform,
        synthetic_ratio=0.0,    # no synthetic
        is_train=False
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val   set: {len(val_dataset)} samples")
    print(f"Test  set: {len(test_dataset)} samples")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.7)

    # Training Loop
    for epoch in range(num_epochs):

        if epoch == 5:
            print("[INFO] Unfreezing last two blocks of EfficientNetV2...")
            for param in model.blocks[-2:].parameters():
                param.requires_grad = True


            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=learning_rate * 0.1,  # lower lr after unfreeze
                                    weight_decay=0.01)

        elif epoch == 10:
            print("[INFO] Unfreezing all layers EfficientNetV2...")
            for param in model.blocks[-3].parameters():
                param.requires_grad = True

            # Re-create the optimizer so it now tracks newly unfrozen params
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=learning_rate * 0.01,  # lower lr after unfreeze
                                    weight_decay=0.01)
            scheduler = CosineAnnealingLR(optimizer, T_max=10)


                        
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if epoch>=10:
                scheduler.step()

            running_loss += loss.item()

        # End of epoch => compute train loss & do validation
        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device=device)

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device=device)
    print(f"\n[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")

    _ = evaluate_test_metrics(model, test_loader, device=device)

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
        num_epochs=40,
        batch_size=128,
        learning_rate=1e-2,
        checkpoint_name="efficientnet_v2_synth_0.0.pth",
    )


if __name__ == "__main__":
    main()
