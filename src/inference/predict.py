"""Real-time ticket prediction and inference module."""

from typing import Dict, Any
from src.features.preprocessor import TextPreprocessor


class InferenceEngine:
    """Inference engine for categorizing tickets and evaluating urgency."""

    def __init__(self, model: Any = None):
        self.model = model
        self.preprocessor = TextPreprocessor()

    def predict_ticket(self, text: str) -> Dict[str, Any]:
        """Predict category, urgency, and confidence for a single ticket."""
        cleaned_text = self.preprocessor.clean_text(text)

        # Rule-based fallback or model inference
        urgent_keywords = ["urgent", "down", "outage", "broken", "critical", "refund", "crash"]
        is_urgent = any(kw in cleaned_text for kw in urgent_keywords)

        if self.model is not None:
            category = self.model.predict([cleaned_text])[0]
            confidence = 0.95
        else:
            # Rule-based mock classification for initial setup
            if any(k in cleaned_text for k in ["billing", "invoice", "charge", "refund"]):
                category = "Billing & Payments"
            elif any(k in cleaned_text for k in ["password", "login", "account", "access"]):
                category = "Account Access"
            elif any(k in cleaned_text for k in ["bug", "error", "crash", "broken"]):
                category = "Technical Bug"
            else:
                category = "General Inquiry"
            confidence = 0.85

        return {
            "cleaned_text": cleaned_text,
            "predicted_category": category,
            "priority": "High" if is_urgent else "Normal",
            "confidence_score": confidence
        }
