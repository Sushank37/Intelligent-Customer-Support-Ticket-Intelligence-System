"""Train and evaluate the enhanced Queue classifier."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.features.queue_features import build_queue_features
from src.models.queue_classifier import QueueClassifier


TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"


def main():
    """Train and evaluate the enhanced Queue model."""

    print("=" * 70)
    print("ENHANCED QUEUE CLASSIFIER")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------------

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    # ---------------------------------------------------------------
    # Build features
    # ---------------------------------------------------------------

    print("\nBuilding training features...")

    X_train = build_queue_features(train_df)
    X_validation = build_queue_features(validation_df)

    y_train = train_df["queue"].tolist()
    y_validation = validation_df["queue"].tolist()

    print(f"Training features: {len(X_train)}")
    print(f"Validation features: {len(X_validation)}")

    # ---------------------------------------------------------------
    # Initialize classifier
    # ---------------------------------------------------------------

    print("\nInitializing Queue classifier...")

    classifier = QueueClassifier(
        max_features=15000,
        ngram_range=(1, 2),
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    print("\nTraining model...")

    classifier.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # ---------------------------------------------------------------
    # Predict
    # ---------------------------------------------------------------

    print("\nGenerating validation predictions...")

    predictions = classifier.predict(
        X_validation
    )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    # ---------------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------------

    classes = classifier.get_classes()

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_validation,
            predictions,
            labels=classes,
            target_names=classes,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=classes,
    )

    print("Labels:")
    print(classes)

    print("\nConfusion Matrix:")
    print(matrix)

    # ---------------------------------------------------------------
    # Save validation predictions
    # ---------------------------------------------------------------

    output_dir = Path("data/processed")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = validation_df.copy()

    results_df["true_queue"] = y_validation
    results_df["predicted_queue"] = predictions
    results_df["correct"] = (
        results_df["true_queue"]
        == results_df["predicted_queue"]
    )

    output_path = (
        output_dir
        / "enhanced_queue_validation_predictions.csv"
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