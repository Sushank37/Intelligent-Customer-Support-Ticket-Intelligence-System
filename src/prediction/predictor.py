"""Unified prediction service for customer support tickets."""

import joblib
from typing import Any, Dict

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.queue_multifeature_classifier import QueueMultiFeatureClassifier


TRANSFORMER_MODEL_PATH = "models/transformer_type"
QUEUE_MODEL_PATH = "models/queue/queue_model.joblib"
PRIORITY_MODEL_PATH = "models/priority/priority_model.joblib"


class TicketPredictor:
    """Generate Type, Queue, and Priority predictions for a ticket."""

    def __init__(
        self,
        transformer_model_path: str = TRANSFORMER_MODEL_PATH,
    ):
        print("Initializing TicketPredictor...")

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Device: {self.device}")

        # ---------------------------------------------------------
        # Type Transformer
        # ---------------------------------------------------------

        print("Loading Type transformer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model_path
        )

        self.type_model = AutoModelForSequenceClassification.from_pretrained(
            transformer_model_path
        )

        self.type_model.to(self.device)
        self.type_model.eval()

        print("Type transformer loaded.")

        # ---------------------------------------------------------
        # Queue Model
        # ---------------------------------------------------------

        print("Loading Queue model...")

        self.queue_model = joblib.load(
            QUEUE_MODEL_PATH
        )

        self.queue_model_ready = True

        print("Queue model loaded.")

        # ---------------------------------------------------------
        # Priority Model
        # ---------------------------------------------------------

        print("Loading Priority model...")

        self.priority_model = joblib.load(
            PRIORITY_MODEL_PATH
        )

        self.priority_model_ready = True

        print("Priority model loaded.")

    # =============================================================
    # Type Prediction
    # =============================================================

    def predict_type(
        self,
        ticket_text: str,
    ) -> Dict[str, Any]:
        """Predict ticket Type using the fine-tuned transformer."""

        if not isinstance(ticket_text, str):
            ticket_text = ""

        encoded = self.tokenizer(
            ticket_text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = self.type_model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )[0]

        predicted_id = int(
            torch.argmax(probabilities).item()
        )

        predicted_label = self.type_model.config.id2label[
            predicted_id
        ]

        confidence = float(
            probabilities[predicted_id].item()
        )

        return {
            "label": str(predicted_label),
            "confidence": confidence,
        }

    # =============================================================
    # Priority Prediction
    # =============================================================

    def predict_priority(
        self,
        ticket_text: str,
    ) -> Dict[str, Any]:
        """Predict ticket Priority."""

        if not self.priority_model_ready:
            raise RuntimeError(
                "Priority model has not been trained or loaded yet."
            )

        if not isinstance(ticket_text, str):
            ticket_text = ""

        probabilities = self.priority_model.predict_proba(
            [ticket_text]
        )[0]

        classes = self.priority_model.pipeline.named_steps[
            "clf"
        ].classes_

        predicted_index = int(
            probabilities.argmax()
        )

        return {
            "label": str(
                classes[predicted_index]
            ),
            "confidence": float(
                probabilities[predicted_index]
            ),
        }

    # =============================================================
    # Queue Prediction
    # =============================================================

    def predict_queue(
        self,
        ticket: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Predict ticket Queue using structured ticket features."""

        if not self.queue_model_ready:
            raise RuntimeError(
                "Queue model has not been trained or loaded yet."
            )

        # Convert ticket dictionary into the DataFrame expected
        # by the trained Queue model.
        df = pd.DataFrame([ticket])

        probabilities = self.queue_model.predict_proba(
            df
        )[0]

        predicted_index = int(
            probabilities.argmax()
        )

        classes = self.queue_model.get_classes()

        predicted_label = classes[predicted_index]

        confidence = float(
            probabilities[predicted_index]
        )

        return {
            "label": str(predicted_label),
            "confidence": confidence,
        }

    # =============================================================
    # Unified Prediction
    # =============================================================

    def predict(
        self,
        ticket: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate Type, Priority, and Queue predictions.

        Prediction order:

            1. Type
            2. Priority
            3. Queue

        The predicted Type and Priority are inserted into the
        ticket before the Queue model is called.
        """

        # Never modify the original dictionary supplied by the caller.
        ticket = ticket.copy()

        ticket_text = ticket.get(
            "ticket_text",
            "",
        )

        if not isinstance(ticket_text, str):
            ticket_text = ""

        # ---------------------------------------------------------
        # 1. Predict Type
        # ---------------------------------------------------------

        type_prediction = self.predict_type(
            ticket_text
        )

        predicted_type = type_prediction["label"]

        # ---------------------------------------------------------
        # 2. Predict Priority
        # ---------------------------------------------------------

        priority_prediction = self.predict_priority(
            ticket_text
        )

        predicted_priority = priority_prediction["label"]

        # ---------------------------------------------------------
        # 3. Insert predicted values into Queue features
        # ---------------------------------------------------------

        ticket["type"] = predicted_type
        ticket["priority"] = predicted_priority

        # ---------------------------------------------------------
        # 4. Predict Queue
        # ---------------------------------------------------------

        queue_prediction = self.predict_queue(
            ticket
        )

        # ---------------------------------------------------------
        # 5. Final response
        # ---------------------------------------------------------

        return {
            "type": type_prediction,
            "queue": queue_prediction,
            "priority": priority_prediction,
        }