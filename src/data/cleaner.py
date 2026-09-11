"""Text cleaning, normalization, and PII masking pipeline."""
import html
import re
from typing import Dict, Optional, Tuple

class TextCleaner:
    """Production text cleaner for customer support messages and historical resolutions."""

    # Regex patterns for normalization and PII masking
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
    PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    ORDER_ID_PATTERN = re.compile(r"\b(?:order|case|ticket|tracking|ref|id)[#:\s-]*[A-Z0-9-]{5,20}\b", re.IGNORECASE)
    ANONYMIZED_USER_PATTERN = re.compile(r"@\d{4,10}\b")
    BRAND_HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    
    def __init__(self, mask_pii: bool = True, strip_mentions: bool = False):
        self.mask_pii = mask_pii
        self.strip_mentions = strip_mentions

    def clean_text(self, text: Optional[str]) -> str:
        """Cleans input text while preserving semantic structure and tokens."""
        if not text or not isinstance(text, str):
            return ""

        # 1. Unescape HTML entities
        text = html.unescape(text)

        # 2. Mask PII if enabled
        if self.mask_pii:
            text = self.mask_sensitive_data(text)

        # 3. Handle URLs
        text = self.URL_PATTERN.sub("[URL]", text)

        # 4. Handle Mentions
        if self.strip_mentions:
            text = self.BRAND_HANDLE_PATTERN.sub("", text)
        else:
            # Replace numeric anonymous twitter user handles with @Customer
            text = self.ANONYMIZED_USER_PATTERN.sub("@Customer", text)

        # 5. Normalize whitespace and newlines
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        return text

    def mask_sensitive_data(self, text: str) -> str:
        """Masks emails, order/case references, and phone numbers."""
        if not text:
            return ""
        text = self.EMAIL_PATTERN.sub("[EMAIL]", text)
        text = self.ORDER_ID_PATTERN.sub("[ORDER_REF]", text)
        text = self.PHONE_PATTERN.sub("[PHONE]", text)
        return text

    def process_record(self, original_text: str) -> Dict[str, str]:
        """Returns a dict containing both original and clean text."""
        return {
            "original_text": original_text,
            "clean_text": self.clean_text(original_text)
        }
