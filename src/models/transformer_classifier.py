"""Multilingual transformer classifier for customer support tickets."""

from typing import List, Dict, Any

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class TransformerTicketClassifier:
    """Multilingual DistilBERT classifier for ticket classification."""

    MODEL_NAME = "distilbert-base-multilingual-cased"

    def __init__(
        self,
        num_labels: int,
        label_to_id: Dict[str, int],
        max_length: int = 256,
    ):
        self.num_labels = num_labels
        self.label_to_id = label_to_id
        self.id_to_label = {
            value: key for key, value in label_to_id.items()
        }
        self.max_length = max_length

        # Use Apple's Metal Performance Shaders on supported Macs.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=num_labels,
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        )

        self.model.to(self.device)

    def encode_labels(self, labels: List[str]) -> List[int]:
        """Convert string labels to integer IDs."""
        return [self.label_to_id[label] for label in labels]

    def decode_labels(self, label_ids: List[int]) -> List[str]:
        """Convert integer IDs back to string labels."""
        return [self.id_to_label[label_id] for label_id in label_ids]

    def tokenize(self, texts: List[str]) -> Dict[str, Any]:
        """Tokenize ticket text."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def predict(self, texts: List[str]) -> List[str]:
        """Predict class labels."""
        self.model.eval()

        encoded = self.tokenize(texts)
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = self.model(**encoded)

        predictions = torch.argmax(
            outputs.logits,
            dim=1,
        ).cpu().tolist()

        return self.decode_labels(predictions)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return class probabilities."""
        self.model.eval()

        encoded = self.tokenize(texts)
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = self.model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )

        return probabilities.cpu().numpy()

    def predict_with_confidence(
        self,
        texts: List[str],
    ) -> List[Dict[str, Any]]:
        """Return predictions together with confidence scores."""

        probabilities = self.predict_proba(texts)

        results = []

        for probability in probabilities:
            predicted_id = int(np.argmax(probability))
            confidence = float(probability[predicted_id])

            results.append(
                {
                    "label": self.id_to_label[predicted_id],
                    "confidence": confidence,
                }
            )

        return results