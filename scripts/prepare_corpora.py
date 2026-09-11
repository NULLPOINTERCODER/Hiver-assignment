"""Script to partition and quarantine dataset into Training and Knowledge Base corpora with zero leakage."""
import os
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.cleaner import TextCleaner
from src.data.sampler import DataSampler
from src.utils.io import load_yaml
from src.utils.logger import setup_logger

logger = setup_logger("prepare_corpora")

# High-precision domain keywords to auto-label training data from customer messages
INTENT_KEYWORDS = {
    "battery_power_issue": ["battery", "drain", "charge", "charging", "charger", "overheat", "percentage", "low power mode", "power off", "battery life"],
    "software_update_issue": ["ios", "update", "ios 11", "ios 10", "upgrade", "downgrade", "itunes", "restore", "stuck on apple logo", "firmware", "boot loop"],
    "account_security_issue": ["apple id", "password", "locked", "verification code", "2fa", "two-factor", "security questions", "hacked", "trusted device", "iforgot"],
    "billing_subscription_issue": ["charged", "refund", "subscription", "cancel", "itunes.com/bill", "payment declined", "bill", "invoice", "receipt", "in-app purchase", "apple music subscription"],
    "hardware_repair_issue": ["screen", "cracked", "broken", "shattered", "genius bar", "applecare", "repair", "hardware", "home button", "camera lens", "bent", "swollen"],
    "connectivity_network_issue": ["sim", "wifi", "wi-fi", "bluetooth", "airpods disconnect", "cellular", "no service", "searching...", "carrier", "airdrop", "hotspot"],
    "app_store_app_issue": ["app store", "download app", "app crash", "whatsapp", "instagram", "facebook app", "waiting...", "cannot connect to app store", "32-bit"],
    "icloud_backup_issue": ["icloud", "backup", "photos sync", "icloud storage", "photo library", "icloud drive", "syncing", "restore contacts", "icloud.com"],
    "audio_media_issue": ["speaker", "microphone", "sound", "volume", "airpods sound", "music pause", "audio", "earpiece", "muffled", "distorted", "headphone"],
    "general_inquiry_advice": ["trade in", "hours", "spanish", "store", "screenshot", "reachability", "apple pay", "qr code", "true tone", "recycle", "pencil", "student discount"]
}

def label_query_heuristically(text: str) -> str:
    """Labels query based on strongest domain keyword match."""
    t = text.lower()
    scores = {}
    for intent, kws in INTENT_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score > 0:
            scores[intent] = score
    if scores:
        return max(scores, key=scores.get)
    return "general_inquiry_advice"

def main():
    golden_path = ROOT_DIR / "evaluation" / "golden_set.csv"
    if not golden_path.exists():
        logger.error("Golden set does not exist.")
        sys.exit(1)
    golden_df = pd.read_csv(golden_path)

    processed_convs_path = ROOT_DIR / "data" / "processed" / "conversations.parquet"
    if not processed_convs_path.exists():
        processed_convs_path = ROOT_DIR / "data" / "sample" / "conversations_sample.parquet"

    df = pd.read_parquet(processed_convs_path)
    logger.info(f"Loaded {len(df)} total conversation pairs.")

    # Strict isolation: Remove any customer messages or conversation IDs appearing in golden set
    golden_messages = set(golden_df["customer_message"].str.lower().str.strip())
    df_clean = df[~df["clean_customer_message"].str.lower().str.strip().isin(golden_messages)].copy()
    
    # Auto-label intent for training and knowledge base
    df_clean["intent"] = df_clean["clean_customer_message"].apply(label_query_heuristically)

    sampler = DataSampler(random_seed=42)
    # Split remaining data into Training set and Knowledge Base
    df_shuffled = df_clean.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    train_df = df_shuffled.iloc[:4000].copy().reset_index(drop=True)
    kb_df = df_shuffled.iloc[4000:9000].copy().reset_index(drop=True)

    # Mathematically verify zero leakage
    sampler.verify_zero_leakage(golden_df, train_df, kb_df)

    # Save training and knowledge base sets
    train_path = ROOT_DIR / "data" / "processed" / "train_set.parquet"
    kb_path = ROOT_DIR / "data" / "processed" / "knowledge_base.parquet"
    
    train_df.to_parquet(train_path, index=False)
    kb_df.to_parquet(kb_path, index=False)
    logger.info(f"Saved {len(train_df)} training samples to {train_path}")
    logger.info(f"Saved {len(kb_df)} knowledge base resolution pairs to {kb_path}")

    # Also save compact sample knowledge base for reproduction
    sample_kb_path = ROOT_DIR / "data" / "sample" / "knowledge_base_sample.parquet"
    kb_df.head(1000).to_parquet(sample_kb_path, index=False)
    logger.info(f"Saved sample knowledge base to {sample_kb_path}")

if __name__ == "__main__":
    main()
