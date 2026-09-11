"""Unit tests for conversation reconstruction and speaker identification."""
import pandas as pd
import pytest
from src.data.conversation_builder import ConversationBuilder

def test_parse_turns_from_text():
    builder = ConversationBuilder(brand_name="AppleSupport")
    raw_str = "Customer: My phone is frozen\nAppleSupport: Please try a force restart."
    turns = builder.parse_turns_from_text(raw_str)
    assert len(turns) == 2
    assert turns[0]["speaker"] == "Customer"
    assert turns[1]["speaker"] == "AppleSupport"

def test_extract_support_pairs():
    builder = ConversationBuilder(brand_name="AppleSupport")
    df = pd.DataFrame([{
        "conversation_id": "conv_123",
        "company": "AppleSupport",
        "conversation": "User: Battery dies fast\nAppleSupport: We can help, check battery settings."
    }])
    pairs = builder.extract_support_pairs(df)
    assert len(pairs) == 1
    assert "battery" in pairs.iloc[0]["customer_message"].lower()
    assert "settings" in pairs.iloc[0]["brand_response"].lower()
