"""Evaluate the fine-tuned transformer on the untouched test set."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_PATH = "models/transformer_type"
TEST_PATH = "data/processed/test.csv"
MAX_LENGTH = 256
BATCH_SIZE = 8


def main():
    """Evaluate the saved transformer model."""

    print("=" * 70)
    print("TRANSFORMER TEST-SET EVALUATION")
    print("=" * 70)

    # Device

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Check model

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    # Load test data

    print("\nLoading test dataset...")

    test_df = pd.read_csv(TEST_PATH)

    print(f"Test rows: {len(test_df)}")

    # Load model and tokenizer

    print("\nLoading fine-tuned model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # Label mapping

    label_to_id = {
        label: int(index)
        for label, index in model.config.label2id.items()
    }

    id_to_label = {
        int(index): label
        for index, label in model.config.id2label.items()
    }

    print("\nLabels:")
    print(label_to_id)

    # Prediction

    texts = test_df["ticket_text"].fillna("").tolist()

    true_labels = [
        label_to_id[label]
        for label in test_df["type"]
    ]

    predictions = []

    print("\nGenerating predictions...")

    for start in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = model(**encoded)

        batch_predictions = (
            torch.argmax(
                outputs.logits,
                dim=1,
            )
            .cpu()
            .tolist()
        )

        predictions.extend(batch_predictions)

    # Metrics

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # Results

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    # Classification report

    target_names = [
        id_to_label[index]
        for index in sorted(id_to_label)
    ]

    print("\nClassification Report:\n")

    print(
        classification_report(
            true_labels,
            predictions,
            labels=sorted(id_to_label),
            target_names=target_names,
            zero_division=0,
        )
    )

    
    # Confusion matrix

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=sorted(id_to_label),
    )

    print("Confusion Matrix:")
    print("Labels:", target_names)
    print(matrix)

    # Save predictions

    results_df = test_df.copy()

    results_df["true_type"] = [
        id_to_label[index]
        for index in true_labels
    ]

    results_df["predicted_type"] = [
        id_to_label[index]
        for index in predictions
    ]

    results_df["correct"] = (
        results_df["true_type"]
        == results_df["predicted_type"]
    )

    output_path = (
        "data/processed/"
        "transformer_type_test_predictions.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nPredictions saved to: {output_path}"
    )


if __name__ == "__main__":
    main()