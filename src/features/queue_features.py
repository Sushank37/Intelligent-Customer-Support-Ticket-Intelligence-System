"""Feature engineering for Queue classification."""

from typing import List

import pandas as pd


METADATA_COLUMNS = [
    "type",
    "priority",
    "language",
    "business_type",
    "tag_1",
    "tag_2",
    "tag_3",
    "tag_4",
    "tag_5",
    "tag_6",
    "tag_7",
    "tag_8",
]


def build_queue_features(df: pd.DataFrame) -> List[str]:
    """
    Combine ticket text with structured metadata.

    Each categorical metadata value is converted into a descriptive
    token so that TF-IDF can learn relationships between metadata
    and queue labels.
    """

    required_columns = ["ticket_text"] + METADATA_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    features = []

    for _, row in df.iterrows():

        text = str(row["ticket_text"])

        metadata_tokens = []

        for column in METADATA_COLUMNS:

            value = row[column]

            if pd.isna(value):
                continue

            value = str(value).strip()

            if not value:
                continue

            # Replace spaces with underscores so each categorical
            # value becomes one TF-IDF token.
            value = value.replace(" ", "_")

            metadata_tokens.append(
                f"{column}_{value}"
            )

        combined_text = (
            text
            + " "
            + " ".join(metadata_tokens)
        )

        features.append(combined_text)

    return features