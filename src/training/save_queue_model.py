"""Train and save the final Queue classification model."""

from pathlib import Path

import joblib
import pandas as pd

from src.models.queue_multifeature_classifier import (
    QueueMultiFeatureClassifier,
)


TRAIN_PATH = "data/processed/train.csv"
MODEL_DIR = Path("models/queue")


def main():
    """Train the final Queue model and save it."""

    print("=" * 70)
    print("SAVING FINAL QUEUE MODEL")
    print("=" * 70)

    print("\nLoading training data...")

    train_df = pd.read_csv(TRAIN_PATH)

    print(f"Training rows: {len(train_df)}")

    # Train on the complete training split

    print("\nTraining Queue model...")

    model = QueueMultiFeatureClassifier(
        word_max_features=10000,
        char_max_features=10000,
    )

    model.fit(
        train_df,
        train_df["queue"].tolist(),
    )

    print("Training complete.")

    # Create model directory

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save model

    model_path = MODEL_DIR / "queue_model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    print(f"\nQueue model saved to: {model_path}")

    # Verify saved model
    file_size_mb = (
        model_path.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"Model size: {file_size_mb:.2f} MB"
    )

    print("\nQueue classes:")

    for label in model.get_classes():
        print(f"  - {label}")

    print("\nQueue model saved successfully.")


if __name__ == "__main__":
    main()