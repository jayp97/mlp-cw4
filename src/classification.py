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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import timm
import timm.data
import argparse
from distutils.util import strtobool
import random
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


class SkinLesionDataset(Dataset):
    """
    Custom PyTorch Dataset that:
      - Reads from data/processed/images (real images in train, val, test splits)
    """

    def __init__(
        self,
        root_dir,                     # e.g. ../data/processed/images/train
        metadata_csv="data/raw/HAM10000_metadata.csv",
        transform=None,
        synthetic_ceiling=0.0,
        augmented_ceiling=0.0,
        synthetic_dir="data/processed_synth/",
        augmented_dir="data/augmented/",
        is_train=False
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.synthetic_ceiling = synthetic_ceiling
        self.synthetic_dir = synthetic_dir
        self.augmented_dir = augmented_dir

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

        # Sythetic images add if training mode 
        if is_train and synthetic_ceiling > 0:
            real_df_count = sum(1 for (_, dx) in data_list if dx == "df")
            real_akiec_count = sum(1 for (_, dx) in data_list if dx == "akiec")
            real_vasc_count = sum(1 for (_, dx) in data_list if dx == "vasc")

            # Determine how many synthetic images are needed
            synthetic_needed_df = int(max(0, synthetic_ceiling - real_df_count))
            synthetic_needed_akiec = int(max(0, synthetic_ceiling - real_akiec_count))
            synthetic_needed_vasc = int(max(0, synthetic_ceiling - real_vasc_count))

            # Gather all synthetic images
            synthetic_files = [f for f in os.listdir(self.synthetic_dir) if f.endswith(".jpg")]

            # Select and add synthetic images for 'df'
            if synthetic_needed_df > 0:
                synthetic_df_files = [f for f in synthetic_files if "df" in f]
                synthetic_selected_df = random.sample(synthetic_df_files, min(len(synthetic_df_files), synthetic_needed_df))
                data_list.extend([(f"synth_{fn}", "df") for fn in synthetic_selected_df])

            # Select and add synthetic images for 'akiec'
            if synthetic_needed_akiec > 0:
                synthetic_akiec_files = [f for f in synthetic_files if "akiec" in f]
                synthetic_selected_akiec = random.sample(synthetic_akiec_files, min(len(synthetic_akiec_files), synthetic_needed_akiec))
                data_list.extend([(f"synth_{fn}", "akiec") for fn in synthetic_selected_akiec])

            # Select and add synthetic images for 'vasc'
            if synthetic_needed_vasc > 0:
                synthetic_vasc_files = [f for f in synthetic_files if "vasc" in f]
                synthetic_selected_vasc = random.sample(synthetic_vasc_files, min(len(synthetic_vasc_files), synthetic_needed_vasc))
                data_list.extend([(f"synth_{fn}", "vasc") for fn in synthetic_selected_vasc])

            # Gather all synthetic images
            synthetic_files = [
                f for f in os.listdir(self.synthetic_dir) if f.endswith(".jpg")
            ]
        elif is_train and augmented_ceiling>0:
            real_counts = {
                "df": sum(1 for _, dx in data_list if dx == "df"),
                "akiec": sum(1 for _, dx in data_list if dx == "akiec"),
                "vasc": sum(1 for _, dx in data_list if dx == "vasc"),
            }

            aug_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                transforms.RandomAffine(degrees=10, shear=15, scale=(0.8, 1.2), translate=(0.2, 0.2))
                ])


            augmented_needed = {dx: max(0, augmented_ceiling - real_counts[dx]) for dx in real_counts}

            for lesion_type, needed in augmented_needed.items():
                if needed > 0:
                    original_images = [fn for fn, dx in data_list if dx == lesion_type]
                    selected_images = random.choices(original_images, k=int(needed))

                    for fn in selected_images:
                        img_path = os.path.join(self.root_dir, fn)
                        image = Image.open(img_path).convert("RGB")  # Load image
                        
                        augmented_image = aug_transform(image)  # Apply augmentations
                        
                        # Save augmented image with new filename
                        new_fn = f"aug_{fn}"
                        aug_path = os.path.join(self.augmented_dir, new_fn)
                        augmented_image.save(aug_path)

                        # Add new augmented image to dataset
                        data_list.append((new_fn, lesion_type))


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
        is_augmented = "aug" in fn
        is_synthetic = "synth" in fn

        if is_synthetic:
            fn_without_synth = fn.removeprefix("synth_")
            img_path = os.path.join(self.synthetic_dir, fn_without_synth)
        elif is_augmented:
            fn_without_synth = fn.removeprefix("synth_")
            img_path = os.path.join(self.augmented_dir, fn_without_synth)
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
    Computes and prints:
      - Overall accuracy
      - Macro precision, recall, F1
      - AUC-ROC per class, plus macro AUC
      - Per-class precision, recall, F1
      - Confusion matrix

    Returns a dict of metrics.
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

    # Overall Accuracy
    overall_acc = accuracy_score(all_labels, all_preds)

    # Per-class metrics
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0,1,2,3,4,5,6]
    )

    # Macro-average metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )

    # AUC per class
    y_true_bin = label_binarize(all_labels, classes=[0,1,2,3,4,5,6])
    roc_aucs = []
    for c in range(7):
        if y_true_bin[:, c].sum() == 0:
            roc_aucs.append(float('nan'))  # No samples for this class
        else:
            auc_c = roc_auc_score(y_true_bin[:, c], all_probs[:, c])
            roc_aucs.append(auc_c)

    # Macro-average AUC (ignoring NaN)
    valid_aucs = [auc for auc in roc_aucs if not np.isnan(auc)]
    if len(valid_aucs) > 0:
        macro_auc = np.mean(valid_aucs)
    else:
        macro_auc = float('nan')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    #  plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix_aug300.png")
    plt.show()

    # Print Summary
    print("\n===== DETAILED TEST METRICS =====")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1-Score:  {macro_f1:.4f}")
    print(f"Macro AUC:       {macro_auc:.4f}\n")

    # Per-class metrics
    for i, cls_name in enumerate(class_names):
        print(f"Class '{cls_name}':")
        print(f"  Precision: {class_precision[i]:.4f}")
        print(f"  Recall:    {class_recall[i]:.4f}")
        print(f"  F1-Score:  {class_f1[i]:.4f}")
        if not np.isnan(roc_aucs[i]):
            print(f"  AUC:       {roc_aucs[i]:.4f}")
        else:
            print("  AUC:       N/A (no samples)")
        print()

    # Optionally, you could also print a detailed classification report:
    # print(classification_report(all_labels, all_preds, target_names=class_names))

    metrics_dict = {
        "overall_acc": overall_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "precision_per_class": class_precision,
        "recall_per_class": class_recall,
        "f1_per_class": class_f1,
        "auc_per_class": roc_aucs,
        "confusion_matrix": cm,
    }

    return metrics_dict

def train_efficientnetv2(
    combine_train_val=True,
    synthetic_ceiling=0.0,
    augmented_ceiling=0.0,
    num_epochs=20,
    batch_size=64,
    learning_rate=5e-3,
    weight_decay_init = 0.7,
    weight_decay = 0.01,
    checkpoint_name="efficientnet_v2.pth",
    train_dir="data/processed/images/train",
    val_dir="data/processed/images/val",
    train_val_dir="data/processed/images/train_val",
    test_dir="data/processed/images/test",
    synthetic_dir="data/processed_synth"
):
    """
    Fine-tunes an EfficientNetV2-L model on train/val/test splits in separate folders.
    Optionally adds synthetic DF images to the train dataset with `synthetic_ratio`.

    Flow:
      - Create train/val/test datasets & loaders
      - Train on each epoch with gradual unfreezing
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
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    eval_transform  = timm.data.create_transform(**data_config, is_training=False)

    # Create Datasets
    if combine_train_val: 
        train_val_dataset = SkinLesionDataset(
            root_dir=train_val_dir,
            metadata_csv="data/raw/HAM10000_metadata.csv",
            transform=train_transform,
            synthetic_ceiling=synthetic_ceiling,  # Only train
            augmented_ceiling=augmented_ceiling,
            synthetic_dir=synthetic_dir,
            is_train=True
        )
    else:
        train_dataset = SkinLesionDataset(
            root_dir=train_dir,
            metadata_csv="data/raw/HAM10000_metadata.csv",
            transform=train_transform,
            synthetic_ceiling=synthetic_ceiling,  # Only train
            synthetic_dir=synthetic_dir,
            is_train=True
        )
        val_dataset = SkinLesionDataset(
            root_dir=val_dir,
            metadata_csv="data/raw/HAM10000_metadata.csv",
            transform=train_transform if combine_train_val else eval_transform,
            synthetic_ceiling=0.0,    # no synthetic
            is_train=False
        )
    test_dataset = SkinLesionDataset(
        root_dir=test_dir,
        metadata_csv="data/raw/HAM10000_metadata.csv",
        transform=eval_transform,
        synthetic_ceiling=0.0,    # no synthetic
        is_train=False
    )

    # Create DataLoaders
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Create DataLoaders: if combining, combine train and val datasets
    if combine_train_val:
        train_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
        # For evaluation during training, we'll use the test set
        eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"Combined Train+Val set: {len(train_val_dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
        print(f"Train set: {len(train_dataset)} samples")
        print(f"Val   set: {len(val_dataset)} samples")
        print(f"Test  set: {len(test_dataset)} samples")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_init)

    # Training Loop
    for epoch in range(num_epochs):

        if epoch == 5:
            print("[INFO] Unfreezing last two blocks of EfficientNetV2...")
            for param in model.blocks[-2:].parameters():
                param.requires_grad = True

            # Re-create the optimizer so it now tracks newly unfrozen params
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=learning_rate * 0.1,  # lower lr after unfreeze
                                    weight_decay=weight_decay)

        elif epoch == 10:
            print("[INFO] Unfreezing third last block of EfficientNetV2...")
            for param in model.blocks[-3].parameters():
                param.requires_grad = True

            # Re-create the optimizer so it now tracks newly unfrozen params
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=learning_rate * 0.01,  # even lower lr after unfreeze
                                    weight_decay=weight_decay)
            #scheduler = CosineAnnealingLR(optimizer, T_max=10)


                        
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #if epoch>=10:
                #scheduler.step()

            running_loss += loss.item()

        # End of epoch => compute train loss & do validation
        train_loss = running_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")

        if combine_train_val == False:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device=device)
            print(f"  Val   Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        else:
            test_loss, test_acc = evaluate(model, eval_loader, criterion, device=device)
            print(f"  Test   Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device=device)
    print(f"\n[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")

    _ = evaluate_test_metrics(model, test_loader, device=device)

    # Save the checkpoint only if using combined training + evaluation set
    if combine_train_val:
        os.makedirs("models/efficientnet_checkpoints/", exist_ok=True)
        ckpt_path = os.path.join("models/efficientnet_checkpoints/", checkpoint_name)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Model saved to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetV2 on the HAM10000 dataset.")
    parser.add_argument("--combine_train_val", type=lambda x: bool(strtobool(x)), default=True, help="Combine training and validation sets (True or False).")
    parser.add_argument("--synthetic_ceiling", type=float, default=0.0, help="Synthetic ceiling for training data.")
    parser.add_argument("--augmented_ceiling", type=float, default=0.0, help="augmented ceiling for training data.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay_init", type=float, default=0.7, help="Initial weight decay.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for training.")
    parser.add_argument("--checkpoint_name", type=str, default="efficientnet_v2.pth", help="Checkpoint filename.")

    args = parser.parse_args()

    train_efficientnetv2(
        combine_train_val=args.combine_train_val,
        synthetic_ceiling=args.synthetic_ceiling,
        augmented_ceiling=args.augmented_ceiling,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay_init=args.weight_decay_init,
        weight_decay=args.weight_decay,
        checkpoint_name=args.checkpoint_name,
    )

if __name__ == "__main__":
    main()
