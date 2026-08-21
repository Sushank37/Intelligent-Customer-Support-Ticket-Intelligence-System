"""Unit tests for text preprocessing, classification model, and inference engine."""

import pytest
from src.features.preprocessor import TextPreprocessor
from src.inference.predict import InferenceEngine


def test_text_preprocessor():
    preprocessor = TextPreprocessor(lower_case=True)
    raw_text = "<h1>Urgent!</h1> Payment failed at http://example.com/pay!!!"
    cleaned = preprocessor.clean_text(raw_text)
    
    assert "http" not in cleaned
    assert "<h1>" not in cleaned
    assert "urgent payment failed at" in cleaned


def test_inference_engine_billing():
    engine = InferenceEngine()
    result = engine.predict_ticket("Invoice issue: double charged on billing cycle")
    
    assert result["predicted_category"] == "Billing & Payments"
    assert result["confidence_score"] > 0.0
    assert "cleaned_text" in result


def test_inference_engine_urgency():
    engine = InferenceEngine()
    result = engine.predict_ticket("Urgent system crash and critical database outage")
    
    assert result["priority"] == "High"
