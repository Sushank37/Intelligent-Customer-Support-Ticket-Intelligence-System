"""Ticket Classification Model Wrapper."""

from typing import List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


class TicketClassifier:
    """Wrapper pipeline for ticket category classification and priority assessment."""

    def __init__(self, max_features: int = 5000, n_estimators: int = 100):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, stop_words="english")),
            ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=42))
        ])

    def fit(self, X: List[str], y: List[str]) -> "TicketClassifier":
        """Train the classifier pipeline."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: List[str]) -> List[str]:
        """Predict ticket categories."""
        return self.pipeline.predict(X).tolist()

    def predict_proba(self, X: List[str]) -> Any:
        """Predict category probabilities."""
        return self.pipeline.predict_proba(X)
