import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import seaborn as sns
from scipy import stats
import cv2
from tqdm import tqdm
import random
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None, label=None):
        self.image_paths = image_paths
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # If label is provided, use it (for classification)
        if self.label is not None:
            return image, self.label, img_path

        return image, img_path


def get_image_paths(directory, extensions=(".jpg", ".jpeg", ".png")):
    """Get all image paths from directory with given extensions."""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def calculate_image_statistics(image):
    """Calculate basic image statistics."""
    # Convert to numpy if it's a torch tensor
    if isinstance(image, torch.Tensor):
        if image.shape[0] in [1, 3, 4]:  # If channel-first format
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()

    # Convert to float and scale to 0-1 if needed
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32) / 255.0

    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)

    # Calculate statistics for each channel
    means = []
    stds = []
    mins = []
    maxs = []
    contrasts = []

    for c in range(image.shape[2]):
        channel = image[:, :, c]
        means.append(np.mean(channel))
        stds.append(np.std(channel))
        mins.append(np.min(channel))
        maxs.append(np.max(channel))
        contrasts.append(maxs[-1] - mins[-1])

    # Calculate histogram-based metrics
    hist_r, _ = np.histogram(image[:, :, 0], bins=256, range=(0, 1))
    hist_g, _ = np.histogram(image[:, :, 1], bins=256, range=(0, 1))
    hist_b, _ = np.histogram(image[:, :, 2], bins=256, range=(0, 1))

    entropy_r = stats.entropy(hist_r + 1e-10)  # Add small constant to avoid log(0)
    entropy_g = stats.entropy(hist_g + 1e-10)
    entropy_b = stats.entropy(hist_b + 1e-10)

    # Convert to Lab color space for perceptual metrics
    rgb_image = (image * 255).astype(np.uint8)
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)

    # Calculate texture features using GLCM
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Use Laplacian for edge detection (proxy for texture complexity)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    edge_intensity = np.var(laplacian)

    return {
        "mean_r": means[0],
        "mean_g": means[1],
        "mean_b": means[2],
        "std_r": stds[0],
        "std_g": stds[1],
        "std_b": stds[2],
        "min_r": mins[0],
        "min_g": mins[1],
        "min_b": mins[2],
        "max_r": maxs[0],
        "max_g": maxs[1],
        "max_b": maxs[2],
        "contrast_r": contrasts[0],
        "contrast_g": contrasts[1],
        "contrast_b": contrasts[2],
        "entropy_r": entropy_r,
        "entropy_g": entropy_g,
        "entropy_b": entropy_b,
        "entropy_avg": (entropy_r + entropy_g + entropy_b) / 3,
        "edge_intensity": edge_intensity,
    }


def compute_similarity_metrics(real_images, synthetic_images):
    """Compute similarity metrics between real and synthetic images."""
    similarity_metrics = []

    # Select a subset of real images for comparison if there are many
    max_comparisons = min(1000, len(real_images) * len(synthetic_images))
    comparison_pairs = []

    # Create pairs of images to compare
    if len(real_images) * len(synthetic_images) <= max_comparisons:
        # If the total number of pairs is manageable, compare all pairs
        for real_path in real_images:
            for synth_path in synthetic_images:
                comparison_pairs.append((real_path, synth_path))
    else:
        # Otherwise, randomly sample pairs
        for _ in range(max_comparisons):
            real_path = random.choice(real_images)
            synth_path = random.choice(synthetic_images)
            comparison_pairs.append((real_path, synth_path))

    # Compute metrics for each pair
    for real_path, synth_path in tqdm(
        comparison_pairs, desc="Computing similarity metrics"
    ):
        real_img = cv2.imread(real_path)
        synth_img = cv2.imread(synth_path)

        # Resize if images have different dimensions
        if real_img.shape != synth_img.shape:
            synth_img = cv2.resize(synth_img, (real_img.shape[1], real_img.shape[0]))

        # Convert to grayscale for SSIM and PSNR
        real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
        synth_gray = cv2.cvtColor(synth_img, cv2.COLOR_BGR2GRAY)

        # Calculate metrics
        ssim_value = ssim(real_gray, synth_gray)
        psnr_value = psnr(real_gray, synth_gray)

        # Calculate histogram similarity
        hist_sim = 0
        for i in range(3):  # For each channel
            hist_real = cv2.calcHist([real_img], [i], None, [256], [0, 256])
            hist_synth = cv2.calcHist([synth_img], [i], None, [256], [0, 256])

            # Normalize histograms
            cv2.normalize(hist_real, hist_real, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_synth, hist_synth, 0, 1, cv2.NORM_MINMAX)

            # Compare histograms
            hist_sim += cv2.compareHist(hist_real, hist_synth, cv2.HISTCMP_CORREL) / 3

        similarity_metrics.append(
            {
                "real_image": os.path.basename(real_path),
                "synthetic_image": os.path.basename(synth_path),
                "ssim": ssim_value,
                "psnr": psnr_value,
                "hist_similarity": hist_sim,
            }
        )

    return similarity_metrics


def train_classifier(real_dir, synthetic_dir, num_epochs=10, batch_size=16):
    """Train a classifier to distinguish between real and synthetic images."""
    # Prepare data
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    real_paths = get_image_paths(real_dir)
    synthetic_paths = get_image_paths(synthetic_dir)

    # Balance datasets
    min_size = min(len(real_paths), len(synthetic_paths))
    if len(real_paths) > min_size:
        real_paths = random.sample(real_paths, min_size)
    if len(synthetic_paths) > min_size:
        synthetic_paths = random.sample(synthetic_paths, min_size)

    # Create datasets
    real_dataset = ImageDataset(real_paths, transform=transform, label=0)  # 0 for real
    synthetic_dataset = ImageDataset(
        synthetic_paths, transform=transform, label=1
    )  # 1 for synthetic

    # Split data into train/test
    train_ratio = 0.8
    real_train_size = int(len(real_dataset) * train_ratio)
    real_test_size = len(real_dataset) - real_train_size

    synthetic_train_size = int(len(synthetic_dataset) * train_ratio)
    synthetic_test_size = len(synthetic_dataset) - synthetic_train_size

    real_train_dataset, real_test_dataset = torch.utils.data.random_split(
        real_dataset, [real_train_size, real_test_size]
    )
    synthetic_train_dataset, synthetic_test_dataset = torch.utils.data.random_split(
        synthetic_dataset, [synthetic_train_size, synthetic_test_size]
    )

    # Combine datasets
    train_dataset = torch.utils.data.ConcatDataset(
        [real_train_dataset, synthetic_train_dataset]
    )
    test_dataset = torch.utils.data.ConcatDataset(
        [real_test_dataset, synthetic_test_dataset]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load a pre-trained model
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    logger.info("Training classifier...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels, _ in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Evaluate the model
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels, _ in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(
                probs[:, 1].cpu().numpy()
            )  # Probability of synthetic class

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(
        all_labels, all_preds, target_names=["Real", "Synthetic"]
    )

    # Calculate accuracy and other metrics
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    classification_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": cr,
        "all_labels": all_labels,
        "all_preds": all_preds,
        "all_probs": all_probs,
    }

    return classification_results, model


def analyze_images(
    real_dir,
    synthetic_dir,
    output_dir,
    class_name=None,
    run_classifier=True,
    num_epochs=5,
):
    """Analyze real and synthetic images and compare them."""
    os.makedirs(output_dir, exist_ok=True)

    # Get image paths
    real_paths = get_image_paths(real_dir)
    synthetic_paths = get_image_paths(synthetic_dir)

    if len(real_paths) == 0:
        logger.error(f"No images found in real directory: {real_dir}")
        return False

    if len(synthetic_paths) == 0:
        logger.error(f"No images found in synthetic directory: {synthetic_dir}")
        return False

    logger.info(
        f"Found {len(real_paths)} real images and {len(synthetic_paths)} synthetic images"
    )

    # Create transform
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    # Create datasets
    real_dataset = ImageDataset(real_paths, transform=transform)
    synthetic_dataset = ImageDataset(synthetic_paths, transform=transform)

    # Create data loaders
    real_loader = DataLoader(real_dataset, batch_size=1, shuffle=False)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=1, shuffle=False)

    # Calculate image statistics
    logger.info("Calculating image statistics for real images...")
    real_stats = []
    for batch in tqdm(real_loader, desc="Real images"):
        # Correctly unpack the batch
        img = batch[0].squeeze(0)  # First element is the image tensor
        img_path = batch[1]  # Second element is the img_path

        stats = calculate_image_statistics(img)
        stats["image"] = os.path.basename(img_path)
        stats["type"] = "real"
        real_stats.append(stats)

    logger.info("Calculating image statistics for synthetic images...")
    synthetic_stats = []
    for batch in tqdm(synthetic_loader, desc="Synthetic images"):
        # Correctly unpack the batch
        img = batch[0].squeeze(0)  # First element is the image tensor
        img_path = batch[1]  # Second element is the img_path

        stats = calculate_image_statistics(img)
        stats["image"] = os.path.basename(img_path)
        stats["type"] = "synthetic"
        synthetic_stats.append(stats)

    # Combine statistics
    all_stats = pd.DataFrame(real_stats + synthetic_stats)

    # Save statistics to CSV
    stats_file = os.path.join(output_dir, "image_statistics.csv")
    all_stats.to_csv(stats_file, index=False)
    logger.info(f"Image statistics saved to {stats_file}")

    # Create visualizations
    logger.info("Creating visualizations...")

    # 1. Distribution plots for key metrics
    for metric in [
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "entropy_avg",
        "edge_intensity",
    ]:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=all_stats,
            x=metric,
            hue="type",
            palette=["blue", "red"],
            common_norm=False,
        )
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dist_{metric}.png"))
        plt.close()

    # 2. Scatter plot of entropy vs. edge intensity
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=all_stats,
        x="entropy_avg",
        y="edge_intensity",
        hue="type",
        palette=["blue", "red"],
    )
    plt.title("Entropy vs. Edge Intensity")
    plt.xlabel("Average Entropy")
    plt.ylabel("Edge Intensity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_entropy_edge.png"))
    plt.close()

    # 3. Box plots for all metrics
    numeric_cols = all_stats.select_dtypes(include=[np.number]).columns.tolist()
    for metric in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=all_stats, x="type", y=metric)
        plt.title(f"Box Plot of {metric}")
        plt.xlabel("Image Type")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"box_{metric}.png"))
        plt.close()

    # 4. Correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    corr = all_stats[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        annot=False,
        fmt=".2f",
    )
    plt.title("Correlation Matrix of Image Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()

    # 5. Statistical tests
    test_results = {}
    for metric in numeric_cols:
        if metric not in ["image", "type"]:
            real_values = all_stats[all_stats["type"] == "real"][metric].dropna()
            synth_values = all_stats[all_stats["type"] == "synthetic"][metric].dropna()

            if len(real_values) > 0 and len(synth_values) > 0:
                # Perform t-test
                t_stat, p_val = stats.ttest_ind(
                    real_values, synth_values, equal_var=False
                )
                test_results[metric] = {
                    "real_mean": real_values.mean(),
                    "synthetic_mean": synth_values.mean(),
                    "difference": real_values.mean() - synth_values.mean(),
                    "real_std": real_values.std(),
                    "synthetic_std": synth_values.std(),
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                }

    # Save test results to CSV
    test_df = pd.DataFrame(test_results).T
    test_df = test_df.reset_index().rename(columns={"index": "metric"})
    test_file = os.path.join(output_dir, "statistical_tests.csv")
    test_df.to_csv(test_file, index=False)
    logger.info(f"Statistical test results saved to {test_file}")

    # Compute similarity metrics
    logger.info("Computing similarity metrics...")
    similarity_metrics = compute_similarity_metrics(real_paths, synthetic_paths)

    # Save similarity metrics to CSV
    similarity_df = pd.DataFrame(similarity_metrics)
    similarity_file = os.path.join(output_dir, "similarity_metrics.csv")
    similarity_df.to_csv(similarity_file, index=False)
    logger.info(f"Similarity metrics saved to {similarity_file}")

    # Plot similarity distributions
    for metric in ["ssim", "psnr", "hist_similarity"]:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=similarity_df, x=metric, kde=True)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dist_similarity_{metric}.png"))
        plt.close()

    # Calculate and display summary statistics
    summary = {
        "real_images": len(real_paths),
        "synthetic_images": len(synthetic_paths),
        "mean_ssim": similarity_df["ssim"].mean(),
        "mean_psnr": similarity_df["psnr"].mean(),
        "mean_hist_similarity": similarity_df["hist_similarity"].mean(),
    }

    # Train classifier if requested
    if run_classifier:
        logger.info("Training classifier to distinguish real from synthetic images...")
        classification_results, model = train_classifier(
            real_dir, synthetic_dir, num_epochs=num_epochs
        )

        # Add classification results to summary
        summary.update(
            {
                "classifier_accuracy": classification_results["accuracy"],
                "classifier_precision": classification_results["precision"],
                "classifier_recall": classification_results["recall"],
                "classifier_f1": classification_results["f1_score"],
            }
        )

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            classification_results["confusion_matrix"],
            annot=True,
            fmt="d",
            xticklabels=["Real", "Synthetic"],
            yticklabels=["Real", "Synthetic"],
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

        # Save classification report
        with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
            f.write(classification_results["classification_report"])

        # Plot ROC curve
        if (
            len(classification_results["all_labels"]) > 0
            and len(classification_results["all_probs"]) > 0
        ):
            from sklearn.metrics import roc_curve, auc

            fpr, tpr, _ = roc_curve(
                classification_results["all_labels"],
                classification_results["all_probs"],
            )
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--", lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "roc_curve.png"))
            plt.close()

            # Add ROC AUC to summary
            summary["roc_auc"] = roc_auc

    # Save summary to file
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("Image Analysis Summary\n")
        f.write("=====================\n\n")

        if class_name:
            f.write(f"Class: {class_name}\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  Real images: {summary['real_images']}\n")
        f.write(f"  Synthetic images: {summary['synthetic_images']}\n\n")

        f.write("Similarity Metrics:\n")
        f.write(
            f"  Mean SSIM: {summary['mean_ssim']:.4f} (higher is more similar, max=1)\n"
        )
        f.write(
            f"  Mean PSNR: {summary['mean_psnr']:.4f} dB (higher is more similar)\n"
        )
        f.write(
            f"  Mean Histogram Similarity: {summary['mean_hist_similarity']:.4f} (higher is more similar, max=1)\n\n"
        )

        if run_classifier:
            f.write("Classifier Performance:\n")
            f.write(f"  Accuracy: {summary['classifier_accuracy']:.4f}\n")
            f.write(f"  Precision: {summary['classifier_precision']:.4f}\n")
            f.write(f"  Recall: {summary['classifier_recall']:.4f}\n")
            f.write(f"  F1 Score: {summary['classifier_f1']:.4f}\n")
            if "roc_auc" in summary:
                f.write(f"  ROC AUC: {summary['roc_auc']:.4f}\n\n")

            f.write("Interpretation:\n")
            acc = summary["classifier_accuracy"]
            if acc > 0.9:
                f.write(
                    "  The classifier can very easily distinguish between real and synthetic images (>90% accuracy).\n"
                )
                f.write(
                    "  This suggests significant differences between the two sets.\n"
                )
            elif acc > 0.75:
                f.write(
                    "  The classifier can reliably distinguish between real and synthetic images (75-90% accuracy).\n"
                )
                f.write(
                    "  There are noticeable differences, but some overlap in characteristics.\n"
                )
            elif acc > 0.6:
                f.write(
                    "  The classifier has moderate success distinguishing images (60-75% accuracy).\n"
                )
                f.write(
                    "  The synthetic images share many characteristics with real ones, with some differences.\n"
                )
            else:
                f.write(
                    "  The classifier struggles to distinguish between real and synthetic images (<60% accuracy).\n"
                )
                f.write(
                    "  This suggests the synthetic images closely mimic the characteristics of real ones.\n"
                )

        f.write("\nKey Differences:\n")
        significant_diffs = test_df[test_df["significant"] == True].sort_values(
            by="p_value"
        )

        if len(significant_diffs) > 0:
            for _, row in significant_diffs.head(5).iterrows():
                f.write(
                    f"  {row['metric']}: Real mean={row['real_mean']:.4f}, Synthetic mean={row['synthetic_mean']:.4f} "
                )
                f.write(f"(p={row['p_value']:.4e})\n")
        else:
            f.write(
                "  No statistically significant differences found in the analyzed metrics.\n"
            )

    logger.info(f"Analysis complete. Results saved to {output_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and compare real and synthetic images"
    )
    parser.add_argument(
        "--real_dir", required=True, help="Directory containing real images"
    )
    parser.add_argument(
        "--synthetic_dir", required=True, help="Directory containing synthetic images"
    )
    parser.add_argument(
        "--output_dir", default="evaluation_results", help="Directory to save results"
    )
    parser.add_argument("--class_name", help="Class name (for reporting)")
    parser.add_argument(
        "--skip_classifier",
        action="store_true",
        help="Skip training a classifier (faster)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs for classifier training"
    )

    args = parser.parse_args()

    analyze_images(
        real_dir=args.real_dir,
        synthetic_dir=args.synthetic_dir,
        output_dir=args.output_dir,
        class_name=args.class_name,
        run_classifier=not args.skip_classifier,
        num_epochs=args.epochs,
    )
