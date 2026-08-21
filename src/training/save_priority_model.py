"""Train and save the final Priority classification model."""

from pathlib import Path

import joblib
import pandas as pd

from src.models.classifier import TicketClassifier


TRAIN_PATH = "data/processed/train.csv"
MODEL_DIR = Path("models/priority")


def main():
    print("=" * 70)
    print("SAVING FINAL PRIORITY MODEL")
    print("=" * 70)

    print("\nLoading training data...")

    train_df = pd.read_csv(TRAIN_PATH)

    print(f"Training rows: {len(train_df)}")

    print("\nTraining Priority model...")

    model = TicketClassifier(
        max_features=10000,
        C=1.0,
    )

    model.fit(
        train_df["ticket_text"]
        .fillna("")
        .astype(str)
        .tolist(),
        train_df["priority"].tolist(),
    )

    print("Training complete.")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "priority_model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    print(f"\nPriority model saved to: {model_path}")

    file_size_mb = (
        model_path.stat().st_size
        / (1024 * 1024)
    )

    print(f"Model size: {file_size_mb:.2f} MB")

    print("\nPriority classes:")

    for label in model.pipeline.named_steps["clf"].classes_:
        print(f"  - {label}")

    print("\nPriority model saved successfully.")


if __name__ == "__main__":
    main()