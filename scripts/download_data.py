"""Script to download and verify the Customer Support on Twitter dataset."""
import os
import sys
from pathlib import Path
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.utils.logger import setup_logger

logger = setup_logger("download_data")

def download_dataset() -> Path:
    """Downloads dataset from Hugging Face or verified mirror."""
    raw_dir = ROOT_DIR / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_parquet = raw_dir / "conversations_raw.parquet"
    
    if target_parquet.exists() and target_parquet.stat().st_size > 100000:
        logger.info(f"Raw dataset already exists at {target_parquet} ({target_parquet.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
        return target_parquet
    
    # URL from verified HF repository containing full conversation threads
    url = "https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation/resolve/main/data/train-00000-of-00001.parquet"
    logger.info(f"Downloading dataset from {url} to {target_parquet}...")
    
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(target_parquet, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (20*1024*1024) < 1024*1024:
                    logger.info(f"Progress: {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({(downloaded/total_size)*100:.1f}%)")
                    
    logger.info(f"Download complete! Saved to {target_parquet} ({target_parquet.stat().st_size / (1024*1024):.2f} MB).")
    return target_parquet

if __name__ == "__main__":
    download_dataset()
