"""Text Preprocessing and Feature Extraction module."""

import re
from typing import List


class TextPreprocessor:
    """Preprocess text for Customer Support Ticket analysis."""

    def __init__(self, lower_case: bool = True):
        self.lower_case = lower_case

    def clean_text(self, text: str) -> str:
        """Clean raw text string by removing noise, extra spaces, and special chars."""
        if not isinstance(text, str):
            return ""

        if self.lower_case:
            text = text.lower()

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def transform(self, texts: List[str]) -> List[str]:
        """Apply cleaning to a list of text instances."""
        return [self.clean_text(t) for t in texts]
