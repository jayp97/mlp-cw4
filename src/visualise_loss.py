#!/usr/bin/env python
import os
import sys
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re


def read_loss_file(loss_file):
    """
    Read loss values from a CSV file.
    """
    if not os.path.exists(loss_file):
        print(f"Error: Loss file {loss_file} not found")
        return None

    epochs = []
    steps = []
    losses = []
    timestamps = []

    with open(loss_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        if not header or len(header) < 3:
            print("Error: CSV file does not have expected columns")
            return None

        has_timestamp = len(header) >= 4 and header[3].lower() == "timestamp"

        for row in reader:
            if len(row) >= 3:
                epochs.append(int(row[0]))
                steps.append(int(row[1]))
                losses.append(float(row[2]))

                if has_timestamp and len(row) >= 4:
                    timestamps.append(row[3])
                else:
                    timestamps.append(None)

    if not losses:
        print("Error: No loss values found in the file")
        return None

    return {
        "epochs": epochs,
        "steps": steps,
        "losses": losses,
        "timestamps": timestamps,
    }


def smooth_losses(losses, window_size=10):
    """
    Apply smoothing to loss values using a moving average.
    """
    if window_size <= 1:
        return losses

    weights = np.ones(window_size) / window_size
    return np.convolve(losses, weights, mode="valid")


def visualize_loss(
    loss_data, output_dir=None, model_name=None, class_name=None, smoothing=10
):
    """
    Visualize loss data in multiple formats.
    """
    if output_dir is None:
        output_dir = "."

    os.makedirs(output_dir, exist_ok=True)

    epochs = loss_data["epochs"]
    steps = loss_data["steps"]
    losses = loss_data["losses"]

    # Set plot style
    plt.style.use("ggplot")

    # Create plot title
    title_parts = []
    if model_name:
        title_parts.append(f"Model: {model_name}")
    if class_name:
        title_parts.append(f"Class: {class_name}")

    title = " - ".join(title_parts) if title_parts else "Training Loss"

    # Plot 1: Loss vs Steps
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, "b-", alpha=0.3, label="Raw Loss")

    # Add smoothed line if requested
    if smoothing > 1 and len(losses) > smoothing:
        smoothed_losses = smooth_losses(losses, smoothing)
        smoothed_steps = steps[smoothing - 1 :]
        plt.plot(
            smoothed_steps,
            smoothed_losses,
            "r-",
            linewidth=2,
            label=f"Smoothed (window={smoothing})",
        )

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss by Training Step")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save the plot
    step_plot_path = os.path.join(output_dir, "loss_by_step.png")
    plt.savefig(step_plot_path, dpi=300, bbox_inches="tight")
    print(f"Loss by step plot saved to: {step_plot_path}")

    # Plot 2: Average Loss per Epoch
    plt.figure(figsize=(12, 6))

    # Calculate average loss per epoch
    epoch_losses = {}
    for e, l in zip(epochs, losses):
        if e not in epoch_losses:
            epoch_losses[e] = []
        epoch_losses[e].append(l)

    avg_losses = [sum(vals) / len(vals) for e, vals in sorted(epoch_losses.items())]
    unique_epochs = sorted(epoch_losses.keys())

    # Plot average loss per epoch
    plt.plot(unique_epochs, avg_losses, "ro-", linewidth=2, markersize=8)

    # Add min/max range
    min_losses = [min(vals) for e, vals in sorted(epoch_losses.items())]
    max_losses = [max(vals) for e, vals in sorted(epoch_losses.items())]

    plt.fill_between(
        unique_epochs,
        min_losses,
        max_losses,
        color="blue",
        alpha=0.2,
        label="Min-Max Range",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Average Loss by Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend(["Average Loss", "Min-Max Range"])

    # Save the plot
    epoch_plot_path = os.path.join(output_dir, "loss_by_epoch.png")
    plt.savefig(epoch_plot_path, dpi=300, bbox_inches="tight")
    print(f"Loss by epoch plot saved to: {epoch_plot_path}")

    # Plot 3: Combined visualization
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(steps, losses, "b-", alpha=0.3, label="Raw Loss")
    if smoothing > 1 and len(losses) > smoothing:
        plt.plot(
            smoothed_steps,
            smoothed_losses,
            "r-",
            linewidth=2,
            label=f"Smoothed (window={smoothing})",
        )
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss by Training Step")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(unique_epochs, avg_losses, "ro-", linewidth=2, markersize=8)
    plt.fill_between(unique_epochs, min_losses, max_losses, color="blue", alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Loss by Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend(["Average Loss", "Min-Max Range"])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the combined plot
    combined_plot_path = os.path.join(output_dir, "loss_combined.png")
    plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
    print(f"Combined loss plot saved to: {combined_plot_path}")

    # Generate loss statistics summary
    loss_stats = {
        "min_loss": min(losses),
        "min_loss_step": steps[losses.index(min(losses))],
        "min_loss_epoch": epochs[losses.index(min(losses))],
        "max_loss": max(losses),
        "max_loss_step": steps[losses.index(max(losses))],
        "max_loss_epoch": epochs[losses.index(max(losses))],
        "final_loss": losses[-1],
        "final_step": steps[-1],
        "final_epoch": epochs[-1],
        "avg_loss": sum(losses) / len(losses),
        "total_steps": len(steps),
        "total_epochs": max(epochs),
    }

    # Calculate per-epoch statistics
    epoch_stats = []
    for epoch, loss_values in sorted(epoch_losses.items()):
        epoch_stats.append(
            {
                "epoch": epoch,
                "min_loss": min(loss_values),
                "max_loss": max(loss_values),
                "avg_loss": sum(loss_values) / len(loss_values),
                "steps": len(loss_values),
            }
        )

    # Write statistics to file
    stats_path = os.path.join(output_dir, "loss_statistics.txt")
    with open(stats_path, "w") as f:
        f.write(f"Loss Statistics Summary\n")
        f.write(f"======================\n\n")

        if title:
            f.write(f"{title}\n\n")

        f.write(f"Overall Statistics:\n")
        f.write(f"------------------\n")
        f.write(
            f"Minimum Loss: {loss_stats['min_loss']:.6f} (Epoch {loss_stats['min_loss_epoch']}, Step {loss_stats['min_loss_step']})\n"
        )
        f.write(
            f"Maximum Loss: {loss_stats['max_loss']:.6f} (Epoch {loss_stats['max_loss_epoch']}, Step {loss_stats['max_loss_step']})\n"
        )
        f.write(f"Average Loss: {loss_stats['avg_loss']:.6f}\n")
        f.write(
            f"Final Loss: {loss_stats['final_loss']:.6f} (Epoch {loss_stats['final_epoch']}, Step {loss_stats['final_step']})\n"
        )
        f.write(f"Total Steps: {loss_stats['total_steps']}\n")
        f.write(f"Total Epochs: {loss_stats['total_epochs']}\n\n")

        f.write(f"Per-Epoch Statistics:\n")
        f.write(f"-------------------\n")

        for stat in epoch_stats:
            f.write(f"Epoch {stat['epoch']}:\n")
            f.write(f"  Minimum Loss: {stat['min_loss']:.6f}\n")
            f.write(f"  Maximum Loss: {stat['max_loss']:.6f}\n")
            f.write(f"  Average Loss: {stat['avg_loss']:.6f}\n")
            f.write(f"  Steps: {stat['steps']}\n")
            f.write(f"\n")

    print(f"Loss statistics saved to: {stats_path}")
    return True


def extract_info_from_path(loss_file_path):
    """
    Try to extract model name and class name from the loss file path.
    """
    model_name = None
    class_name = None

    # Try to extract class name
    class_match = re.search(
        r"(?:class|_)[-_]?([a-zA-Z0-9]+)", loss_file_path, re.IGNORECASE
    )
    if class_match:
        class_name = class_match.group(1)

    # Try to extract model name
    model_match = re.search(
        r"(?:model|sd)[-_]?(v\d+\.\d+|v\d+|sd\d+)", loss_file_path, re.IGNORECASE
    )
    if model_match:
        model_name = model_match.group(1)

    return model_name, class_name


def main():
    parser = argparse.ArgumentParser(description="Visualize training loss data")
    parser.add_argument("loss_file", help="Path to CSV file with loss values")
    parser.add_argument("--output_dir", help="Directory to save the visualizations")
    parser.add_argument("--model", help="Model name (for plot titles)")
    parser.add_argument("--class_name", help="Class name (for plot titles)")
    parser.add_argument(
        "--smoothing", type=int, default=10, help="Smoothing window size for loss curve"
    )

    args = parser.parse_args()

    # Read loss data
    loss_data = read_loss_file(args.loss_file)
    if loss_data is None:
        return

    # Set output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.loss_file)
        if not args.output_dir:
            args.output_dir = "."

    # Try to extract model and class if not provided
    if args.model is None or args.class_name is None:
        model_guess, class_guess = extract_info_from_path(args.loss_file)

        if args.model is None and model_guess:
            args.model = model_guess

        if args.class_name is None and class_guess:
            args.class_name = class_guess

    # Visualize the loss data
    visualize_loss(
        loss_data,
        output_dir=args.output_dir,
        model_name=args.model,
        class_name=args.class_name,
        smoothing=args.smoothing,
    )


if __name__ == "__main__":
    main()
