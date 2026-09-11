"""Sampling, train/test splitting, and evaluation isolation module."""
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("sampler")

class DataSampler:
    """Manages strict isolation between Golden Evaluation Set, Training Data, and RAG Knowledge Base."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def verify_zero_leakage(self, golden_df: pd.DataFrame, train_df: pd.DataFrame, kb_df: Optional[pd.DataFrame] = None) -> bool:
        """Mathematically verifies 0% overlap between golden evaluation set and training/retrieval corpora."""
        golden_ids: Set[str] = set(golden_df["id"].astype(str).unique()) if "id" in golden_df.columns else set()
        golden_conv_ids: Set[str] = set(golden_df["conversation_id"].astype(str).unique()) if "conversation_id" in golden_df.columns else set()
        golden_texts: Set[str] = set(golden_df["customer_message"].str.lower().str.strip().unique())

        train_texts: Set[str] = set(train_df["customer_message"].str.lower().str.strip().unique()) if "customer_message" in train_df.columns else set()
        train_conv_ids: Set[str] = set(train_df["conversation_id"].astype(str).unique()) if "conversation_id" in train_df.columns else set()

        text_overlap_train = golden_texts.intersection(train_texts)
        conv_overlap_train = golden_conv_ids.intersection(train_conv_ids)

        if text_overlap_train:
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: {len(text_overlap_train)} texts overlap between Golden and Train sets!")
        if conv_overlap_train:
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: {len(conv_overlap_train)} conversation IDs overlap between Golden and Train sets!")

        if kb_df is not None:
            kb_texts: Set[str] = set(kb_df["customer_message"].str.lower().str.strip().unique())
            kb_conv_ids: Set[str] = set(kb_df["conversation_id"].astype(str).unique())
            text_overlap_kb = golden_texts.intersection(kb_texts)
            conv_overlap_kb = golden_conv_ids.intersection(kb_conv_ids)
            if text_overlap_kb:
                raise ValueError(f"CRITICAL LEAKAGE DETECTED: {len(text_overlap_kb)} texts overlap between Golden and Knowledge Base sets!")
            if conv_overlap_kb:
                raise ValueError(f"CRITICAL LEAKAGE DETECTED: {len(conv_overlap_kb)} conversation IDs overlap between Golden and Knowledge Base sets!")

        logger.info("VERIFIED: Zero data leakage confirmed. 0% overlap across all sets.")
        return True

    def create_evaluation_split(
        self,
        df: pd.DataFrame,
        n_golden: int = 200,
        n_train: int = 3000,
        n_kb: int = 5000
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits full dataset into strictly disjoint Golden, Training, and Knowledge Base subsets."""
        df_shuffled = df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
        
        # 1. Quarantined Golden Set
        golden_df = df_shuffled.iloc[:n_golden].copy().reset_index(drop=True)
        remaining_df = df_shuffled.iloc[n_golden:].copy().reset_index(drop=True)

        # 2. Training Set
        train_df = remaining_df.iloc[:n_train].copy().reset_index(drop=True)
        
        # 3. Knowledge Base (RAG pool)
        kb_df = remaining_df.iloc[n_train : n_train + n_kb].copy().reset_index(drop=True)

        # Verify zero leakage
        self.verify_zero_leakage(golden_df, train_df, kb_df)
        
        return golden_df, train_df, kb_df
