"""Conversation reconstruction and turn extraction pipeline."""
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.data.cleaner import TextCleaner
from src.utils.logger import setup_logger

logger = setup_logger("conversation_builder")

class ConversationBuilder:
    """Reconstructs structured support interactions from raw conversation threads."""

    def __init__(self, brand_name: str = "AppleSupport", cleaner: Optional[TextCleaner] = None):
        self.brand_name = brand_name
        self.cleaner = cleaner or TextCleaner(mask_pii=True)

    def parse_turns_from_text(self, conversation_field: Any) -> List[Dict[str, str]]:
        """Parses turns from list, json string, or text format."""
        if isinstance(conversation_field, list):
            turns = []
            for item in conversation_field:
                if isinstance(item, dict):
                    turns.append(item)
                elif isinstance(item, str):
                    turns.append({"text": item})
            return turns
        elif isinstance(conversation_field, str):
            conversation_str = conversation_field.strip()
            # Try JSON parse
            if conversation_str.startswith("[") and conversation_str.endswith("]"):
                try:
                    return json.loads(conversation_str)
                except Exception:
                    try:
                        return ast.literal_eval(conversation_str)
                    except Exception:
                        pass
            
            # Line-by-line speaker format: "Speaker: Text" or "@User: Text"
            lines = conversation_str.split("\n")
            turns = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    speaker, text = line.split(":", 1)
                    turns.append({"speaker": speaker.strip(), "text": text.strip()})
                else:
                    turns.append({"text": line})
            return turns
        return []

    def extract_support_pairs(self, df: pd.DataFrame, max_pairs: Optional[int] = None) -> pd.DataFrame:
        """Extracts customer query -> brand response pairs for knowledge base & classification."""
        logger.info(f"Extracting support interaction pairs for brand: '{self.brand_name}'...")
        records = []

        # Filter for the target brand if company column exists
        if "company" in df.columns:
            brand_df = df[df["company"].astype(str).str.lower() == self.brand_name.lower()]
        else:
            brand_df = df

        logger.info(f"Found {len(brand_df)} raw conversation threads for {self.brand_name}.")

        for _, row in brand_df.iterrows():
            conv_id = str(row.get("conversation_id", f"conv_{len(records)}"))
            raw_conv = row.get("conversation", "")
            turns = self.parse_turns_from_text(raw_conv)

            if not turns:
                continue

            # Identify customer message (first turn or non-brand turn) and brand response
            cust_msg = ""
            brand_resp = ""
            history = []

            for turn in turns:
                speaker = str(turn.get("speaker", turn.get("author", ""))).lower()
                text = str(turn.get("text", turn.get("content", ""))).strip()
                if not text:
                    continue

                is_brand = (self.brand_name.lower() in speaker) or (self.brand_name.lower() in text.lower() and len(text) > 20 and not cust_msg)
                
                if not cust_msg and not is_brand:
                    cust_msg = text
                elif cust_msg and not brand_resp and is_brand:
                    brand_resp = text
                
                history.append(f"{speaker or 'User'}: {text}")

            # Fallback if structure had first turn as customer, second turn as brand
            if not cust_msg and len(turns) >= 2:
                cust_msg = str(turns[0].get("text", ""))
                brand_resp = str(turns[1].get("text", ""))
            elif cust_msg and not brand_resp and len(turns) >= 2:
                brand_resp = str(turns[-1].get("text", ""))

            # Filter out empty or trivially short messages
            if len(cust_msg) > 12 and len(brand_resp) > 10:
                clean_cust = self.cleaner.clean_text(cust_msg)
                clean_resp = self.cleaner.clean_text(brand_resp)

                records.append({
                    "conversation_id": conv_id,
                    "brand": self.brand_name,
                    "customer_message": cust_msg,
                    "brand_response": brand_resp,
                    "clean_customer_message": clean_cust,
                    "clean_brand_response": clean_resp,
                    "history": "\n".join(history[:4])
                })

            if max_pairs and len(records) >= max_pairs:
                break

        pairs_df = pd.DataFrame(records)
        logger.info(f"Extracted {len(pairs_df)} high-quality customer-brand support pairs.")
        return pairs_df
