"""Ticket classification model using combined word and character TF-IDF."""

from typing import List, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion


class TicketClassifier:
    """Combined TF-IDF + Logistic Regression classifier."""

    def __init__(
        self,
        max_features: int = 10000,
        C: float = 1.0,
    ):
        features = FeatureUnion(
            [
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=max_features,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )

        self.pipeline = Pipeline(
            [
                ("features", features),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, X: List[str], y: List[str]) -> "TicketClassifier":
        """Train the classifier."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: List[str]) -> List[str]:
        """Predict class labels."""
        return self.pipeline.predict(X).tolist()

    def predict_proba(self, X: List[str]) -> Any:
        """Predict class probabilities."""
        return self.pipeline.predict_proba(X)

    def get_feature_names(self) -> List[str]:
        """Return combined feature names."""
        features = self.pipeline.named_steps["features"]

        feature_names = []

        for name, transformer in features.transformer_list:
            names = transformer.get_feature_names_out()
            feature_names.extend(
                [f"{name}__{feature}" for feature in names]
            )

        return feature_names