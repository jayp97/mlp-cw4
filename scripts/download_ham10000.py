import os
import argparse
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi


def download_ham10000(raw_data_path):
    """
    Downloads the HAM10000 dataset from Kaggle and stores it in the specified directory.

    Parameters:
        raw_data_path (str): Path to store the downloaded dataset.
    """
    os.makedirs(raw_data_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("Downloading HAM10000 dataset...")
    api.dataset_download_files(
        "kmader/skin-cancer-mnist-ham10000", path=raw_data_path, unzip=True
    )

    # Ensure extracted images are inside `data/raw/images/`
    extracted_images_part1 = os.path.join(raw_data_path, "HAM10000_images_part_1")
    extracted_images_part2 = os.path.join(raw_data_path, "HAM10000_images_part_2")
    target_images_path = os.path.join(raw_data_path, "images")

    os.makedirs(target_images_path, exist_ok=True)

    # Move images if needed
    if os.path.exists(extracted_images_part1):
        for img in os.listdir(extracted_images_part1):
            shutil.move(os.path.join(extracted_images_part1, img), target_images_path)
        os.rmdir(extracted_images_part1)

    if os.path.exists(extracted_images_part2):
        for img in os.listdir(extracted_images_part2):
            shutil.move(os.path.join(extracted_images_part2, img), target_images_path)
        os.rmdir(extracted_images_part2)

    print(f"Download complete. Files are saved in {raw_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download HAM10000 dataset and store it in a specified folder."
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default="data/raw",
        help="Path to store the raw data (default: data/raw)",
    )
    args = parser.parse_args()
    download_ham10000(args.raw_data_path)
