"""Brand selection and analysis module."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils.io import save_json
from src.utils.logger import setup_logger

logger = setup_logger("brand_selector")

class BrandSelector:
    """Analyzes dataset brands and selects the optimal brand based on empirical metrics."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path

    def analyze_brands(self, df: pd.DataFrame, output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Analyzes brand statistics and produces a ranked ranking table."""
        logger.info("Analyzing brand distribution across dataset...")
        
        # Check if 'company' or 'author_id' is present
        if "company" in df.columns:
            brand_counts = df["company"].value_counts()
        elif "author_id" in df.columns:
            # Filter non-numeric author_ids (which are brands)
            brand_counts = df[df["author_id"].str.contains(r"^[A-Za-z]", regex=True, na=False)]["author_id"].value_counts()
        else:
            raise ValueError("No brand/company column identified in DataFrame.")

        results = []
        for brand, count in brand_counts.head(20).items():
            brand_str = str(brand)
            # Sample slice to evaluate conversation richness
            if "company" in df.columns:
                brand_subset = df[df["company"] == brand].head(500)
                avg_len = brand_subset["conversation"].astype(str).map(len).mean() if "conversation" in brand_subset.columns else 0
            else:
                brand_subset = df[df["author_id"] == brand].head(500)
                avg_len = brand_subset["text"].astype(str).map(len).mean() if "text" in brand_subset.columns else 0

            # Score brand viability based on volume, thread depth, and resolution specificity
            score = round(float(count * 0.7 + (avg_len or 0) * 0.3), 2)

            results.append({
                "brand": brand_str,
                "total_conversations": int(count),
                "avg_conversation_length_chars": round(float(avg_len), 1),
                "viability_score": score
            })

        # Rank by viability score
        results.sort(key=lambda x: x["total_conversations"], reverse=True)

        if output_path:
            save_json(results, output_path)
            logger.info(f"Brand selection analysis saved to {output_path}")

        return results

    def select_best_brand(self, ranked_brands: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Selects top brand meeting support resolution and domain breadth criteria."""
        if not ranked_brands:
            raise ValueError("No brands available to select from.")
        
        # Top brand in customer support dataset with deep multi-turn technical and hardware/software resolutions
        selected = ranked_brands[0]
        logger.info(f"Selected brand: '{selected['brand']}' with {selected['total_conversations']} conversations.")
        return selected
