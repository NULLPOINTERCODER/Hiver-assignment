"""Unit tests for text cleaning, normalization, and PII masking."""
import pytest
from src.data.cleaner import TextCleaner

@pytest.fixture
def cleaner():
    return TextCleaner(mask_pii=True)

def test_clean_urls(cleaner):
    text = "Check this page https://support.apple.com/kb/HT201415 for details."
    cleaned = cleaner.clean_text(text)
    assert "[URL]" in cleaned
    assert "https://" not in cleaned

def test_mask_email(cleaner):
    text = "My Apple ID is john.doe@example.com, please assist."
    cleaned = cleaner.clean_text(text)
    assert "[EMAIL]" in cleaned
    assert "john.doe@example.com" not in cleaned

def test_mask_phone_number(cleaner):
    text = "Call me at (555) 123-4567 regarding my battery issue."
    cleaned = cleaner.clean_text(text)
    assert "[PHONE]" in cleaned
    assert "555" not in cleaned

def test_mask_order_ref(cleaner):
    text = "My repair case is Case#1002938482."
    cleaned = cleaner.clean_text(text)
    assert "[ORDER_REF]" in cleaned

def test_html_unescape(cleaner):
    text = "iPhone &amp; iPad issues &lt;urgent&gt;"
    cleaned = cleaner.clean_text(text)
    assert "&" in cleaned
    assert "<urgent>" in cleaned
