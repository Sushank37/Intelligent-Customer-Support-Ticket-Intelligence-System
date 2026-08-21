"""Multi-feature Queue classifier using text and structured metadata."""

from typing import Any, List

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class QueueMultiFeatureClassifier:
    """Queue classifier with separate text and metadata feature spaces."""

    def __init__(
        self,
        word_max_features: int = 10000,
        char_max_features: int = 10000,
    ):
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features

        self.pipeline = None

    def _build_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare text and structured metadata columns."""

        data = pd.DataFrame(index=df.index)

        data["ticket_text"] = (
            df["ticket_text"]
            .fillna("")
            .astype(str)
        )

        # Combine tags into one categorical text field.
        tag_columns = [
            "tag_1",
            "tag_2",
            "tag_3",
            "tag_4",
            "tag_5",
            "tag_6",
            "tag_7",
            "tag_8",
        ]

        data["tags"] = (
            df[tag_columns]
            .fillna("")
            .astype(str)
            .apply(
                lambda row: " ".join(
                    value
                    for value in row
                    if value.strip()
                ),
                axis=1,
            )
        )

        # Keep structured columns as strings so the vectorizer
        # treats their values as categorical features.
        for column in [
            "type",
            "priority",
            "language",
            "business_type",
        ]:
            data[column] = (
                df[column]
                .fillna("missing")
                .astype(str)
            )

        return data

    def fit(
        self,
        df: pd.DataFrame,
        y: List[str],
    ) -> "QueueMultiFeatureClassifier":
        """Train the multi-feature Queue classifier."""

        data = self._build_dataframe(df)

        self.pipeline = Pipeline([
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                max_features=self.word_max_features,
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                            ),
                            "ticket_text",
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char",
                                max_features=self.char_max_features,
                                min_df=2,
                                ngram_range=(3, 5),
                                sublinear_tf=True,
                            ),
                            "ticket_text",
                        ),
                        (
                            "tags_tfidf",
                            TfidfVectorizer(
                                token_pattern=r"(?u)\b\w+\b",
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                            ),
                            "tags",
                        ),
                        (
                            "type",
                            TfidfVectorizer(
                                token_pattern=r"(?u)\b\w+\b",
                            ),
                            "type",
                        ),
                        (
                            "priority",
                            TfidfVectorizer(
                                token_pattern=r"(?u)\b\w+\b",
                            ),
                            "priority",
                        ),
                        (
                            "language",
                            TfidfVectorizer(
                                token_pattern=r"(?u)\b\w+\b",
                            ),
                            "language",
                        ),
                        (
                            "business_type",
                            TfidfVectorizer(
                                token_pattern=r"(?u)\b\w+\b",
                            ),
                            "business_type",
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ])

        self.pipeline.fit(data, y)

        return self

    def predict(
        self,
        df: pd.DataFrame,
    ) -> List[str]:
        """Predict queues."""

        if self.pipeline is None:
            raise RuntimeError(
                "Model has not been trained. Call fit() first."
            )

        data = self._build_dataframe(df)

        return self.pipeline.predict(data).tolist()

    def predict_proba(
        self,
        df: pd.DataFrame,
    ) -> Any:
        """Return Queue prediction probabilities."""

        if self.pipeline is None:
            raise RuntimeError(
                "Model has not been trained. Call fit() first."
            )

        data = self._build_dataframe(df)

        return self.pipeline.predict_proba(data)

    def get_classes(self) -> List[str]:
        """Return Queue classes learned during training."""

        if self.pipeline is None:
            raise RuntimeError(
                "Model has not been trained yet."
            )

        return (
            self.pipeline
            .named_steps["clf"]
            .classes_
            .tolist()
        )