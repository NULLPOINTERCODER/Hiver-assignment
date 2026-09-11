"""Script to run intent discovery on AppleSupport customer queries."""
import os
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.intent.discovery import IntentDiscoverer
from src.utils.io import save_json, save_yaml
from src.utils.logger import setup_logger

logger = setup_logger("discover_intents")

def main():
    data_path = ROOT_DIR / "data" / "processed" / "conversations.parquet"
    if not data_path.exists():
        data_path = ROOT_DIR / "data" / "sample" / "conversations_sample.parquet"
    
    df = pd.read_parquet(data_path)
    customer_texts = df["clean_customer_message"].dropna().tolist()

    discoverer = IntentDiscoverer(n_clusters=10, random_state=42)
    clusters = discoverer.discover_clusters(customer_texts)

    results_path = ROOT_DIR / "results" / "intent_clusters.json"
    save_json(clusters, results_path)
    logger.info(f"Discovered intent clusters saved to {results_path}")

    for c in clusters:
        print(f"\n--- Cluster {c['cluster_id']} (size: {c['size']}) ---")
        print("Keywords:", ", ".join(c["keywords"]))
        print("Sample 1:", c["sample_queries"][0] if c["sample_queries"] else "")

if __name__ == "__main__":
    main()
