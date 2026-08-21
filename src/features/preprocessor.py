"""Text preprocessing utilities for customer support tickets."""

import re
import unicodedata
from typing import List


class TextPreprocessor:
    """Clean and normalize multilingual customer support ticket text."""

    def __init__(self, lower_case: bool = True):
        self.lower_case = lower_case

    def combine_ticket_text(
        self,
        subject: object,
        body: object,
    ) -> str:
        """Combine subject and body with robust missing-value handling."""
        subject_text = subject if isinstance(subject, str) else ""
        body_text = body if isinstance(body, str) else ""

        if subject_text and body_text:
            return f"{subject_text}. {body_text}"

        if subject_text:
            return subject_text

        return body_text

    def clean_text(self, text: str) -> str:
        """Clean a single ticket text while preserving multilingual characters."""
        if not isinstance(text, str):
            return ""

        # Normalize Unicode characters.
        text = unicodedata.normalize("NFKC", text)

        # Convert escaped newline/tab sequences into spaces.
        text = re.sub(r"\\[nrt]", " ", text)

        # Remove HTML tags.
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs.
        text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Convert repeated whitespace to a single space.
        text = re.sub(r"\s+", " ", text).strip()

        # Convert to lowercase.
        if self.lower_case:
            text = text.lower()

        return text

    def transform(self, texts: List[str]) -> List[str]:
        """Apply cleaning to a list of ticket texts."""
        return [self.clean_text(text) for text in texts]