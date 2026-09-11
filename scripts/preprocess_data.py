"""Script to preprocess raw dataset, rank brands, and extract structured support conversations."""
import os
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.brand_selector import BrandSelector
from src.data.cleaner import TextCleaner
from src.data.conversation_builder import ConversationBuilder
from src.data.loader import DataLoader
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("preprocess_data")

def main():
    config = load_config()
    raw_path = ROOT_DIR / "data" / "raw" / "conversations_raw.parquet"
    if not raw_path.exists():
        logger.error(f"Raw data not found at {raw_path}. Run scripts/download_data.py first.")
        sys.exit(1)

    loader = DataLoader(raw_path)
    df = loader.load_data(n_rows=100000) # Load rich representative slice of 100k conversations

    # Brand Analysis & Ranking
    selector = BrandSelector()
    brand_results_path = ROOT_DIR / "results" / "brand_selection.json"
    ranked_brands = selector.analyze_brands(df, output_path=brand_results_path)
    best_brand = selector.select_best_brand(ranked_brands)

    selected_brand_name = config.get("brand", {}).get("selected_brand", best_brand["brand"])
    logger.info(f"Target brand for system: '{selected_brand_name}'")

    # Conversation Reconstruction
    cleaner = TextCleaner(mask_pii=True)
    builder = ConversationBuilder(brand_name=selected_brand_name, cleaner=cleaner)
    pairs_df = builder.extract_support_pairs(df, max_pairs=15000)

    # Save processed conversations and knowledge base
    processed_dir = ROOT_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_parquet = processed_dir / "conversations.parquet"
    pairs_df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved {len(pairs_df)} processed conversation pairs to {output_parquet}")

    # Create compact sample evaluation artifact in data/sample/
    sample_dir = ROOT_DIR / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_df = pairs_df.head(1500)
    sample_df.to_parquet(sample_dir / "conversations_sample.parquet", index=False)
    logger.info(f"Saved compact reproduction sample ({len(sample_df)} pairs) to {sample_dir / 'conversations_sample.parquet'}")

if __name__ == "__main__":
    main()
