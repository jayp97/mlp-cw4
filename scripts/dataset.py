import os
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi

def download_ham10000(raw_data_path):

    os.makedirs(raw_data_path, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    
    print("Downloading HAM10000 dataset...")
    api.dataset_download_files("kmader/skin-cancer-mnist-ham10000", path=raw_data_path, unzip=True)
    print(f"Download complete. Files are saved in {raw_data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download HAM10000 dataset and store it in a specified folder."
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default="data/raw",
        help="Path to store the raw data (default: data/raw)"
    )
    args = parser.parse_args()
    download_ham10000(args.raw_data_path)
