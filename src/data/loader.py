"""Dataset loading and schema detection utilities."""
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("data_loader")

SUPPORTED_SCHEMAS = {
    "kaggle_raw": ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_reply_to_tweet_id"],
    "reconstructed": ["conversation_id", "company", "conversation", "summary"],
    "support_pairs": ["conversation_id", "customer_message", "brand_response", "conversation_history", "brand"]
}

class DataLoader:
    """Robust data loader supporting multiple formats and schema detection."""
    
    def __init__(self, raw_path: Optional[Union[str, Path]] = None):
        self.raw_path = Path(raw_path) if raw_path else None

    @staticmethod
    def detect_schema(df: pd.DataFrame) -> str:
        """Detects the schema type from DataFrame columns."""
        cols = set(df.columns)
        if {"conversation_id", "company", "conversation"}.issubset(cols):
            return "reconstructed"
        elif {"tweet_id", "author_id", "inbound", "text"}.issubset(cols):
            return "kaggle_raw"
        elif {"customer_message", "brand_response"}.issubset(cols):
            return "support_pairs"
        return "unknown"

    def load_data(self, filepath: Optional[Union[str, Path]] = None, n_rows: Optional[int] = None) -> pd.DataFrame:
        """Loads data from CSV or Parquet into a Pandas DataFrame."""
        path = Path(filepath) if filepath else self.raw_path
        if not path or not path.exists():
            raise FileNotFoundError(f"File not found at: {path}")

        logger.info(f"Loading data from {path} (n_rows={n_rows})...")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            if n_rows is not None:
                df = df.iloc[:n_rows]
        elif path.suffix in [".csv", ".gz"]:
            df = pd.read_csv(path, nrows=n_rows)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        schema = self.detect_schema(df)
        logger.info(f"Loaded {len(df)} rows. Detected schema: '{schema}'")
        return df

    def stream_chunks(self, filepath: Optional[Union[str, Path]] = None, chunk_size: int = 50000) -> Generator[pd.DataFrame, None, None]:
        """Streams CSV or Parquet in chunks for memory-efficient processing."""
        path = Path(filepath) if filepath else self.raw_path
        if not path or not path.exists():
            raise FileNotFoundError(f"File not found at: {path}")

        if path.suffix == ".csv":
            for chunk in pd.read_csv(path, chunksize=chunk_size):
                yield chunk
        else:
            df = self.load_data(path)
            for i in range(0, len(df), chunk_size):
                yield df.iloc[i : i + chunk_size]
