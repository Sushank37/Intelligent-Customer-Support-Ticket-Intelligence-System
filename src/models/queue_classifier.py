"""Enhanced Queue classifier using ticket text and structured metadata."""

from typing import List, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class QueueClassifier:
    """TF-IDF + Logistic Regression classifier for ticket queues."""

    def __init__(
        self,
        max_features: int = 15000,
        ngram_range: tuple = (1, 2),
    ):
        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
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

    def fit(
        self,
        X: List[str],
        y: List[str],
    ) -> "QueueClassifier":
        """Train the Queue classifier."""
        self.pipeline.fit(X, y)
        return self

    def predict(
        self,
        X: List[str],
    ) -> List[str]:
        """Predict ticket queues."""
        return self.pipeline.predict(X).tolist()

    def predict_proba(
        self,
        X: List[str],
    ) -> Any:
        """Return probability for each queue."""
        return self.pipeline.predict_proba(X)

    def get_classes(self) -> List[str]:
        """Return queue classes learned during training."""
        return self.pipeline.named_steps["clf"].classes_.tolist()