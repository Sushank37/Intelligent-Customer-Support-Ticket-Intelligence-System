"""Fine-tune a multilingual transformer for ticket classification."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "distilbert-base-multilingual-cased"

TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"

OUTPUT_DIR = "models/transformer_type"

MAX_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TicketDataset(Dataset):
    """PyTorch dataset for support tickets."""

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
        )

        encoding["labels"] = self.labels[index]

        return encoding


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_prediction):
    """Calculate classification metrics."""

    logits, labels = eval_prediction

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main():
    """Train and evaluate the transformer."""

    set_seed(SEED)

    print("=" * 70)
    print("MULTILINGUAL TRANSFORMER TRAINING")
    print("=" * 70)

    
    # Device

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device: {device}")

    # Load data

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    
    # Labels

    labels = sorted(train_df["type"].unique())

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    print("\nLabels:")
    print(label_to_id)

    train_labels = [
        label_to_id[label]
        for label in train_df["type"]
    ]

    validation_labels = [
        label_to_id[label]
        for label in validation_df["type"]
    ]

    # Tokenizer
    

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # Datasets

    print("Tokenizing datasets...")

    train_dataset = TicketDataset(
        texts=train_df["ticket_text"].tolist(),
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    validation_dataset = TicketDataset(
        texts=validation_df["ticket_text"].tolist(),
        labels=validation_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    # Model

    print("\nLoading pretrained model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # Data collator

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # Training arguments

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        logging_strategy="steps",
        logging_steps=25,

        save_total_limit=2,

        report_to="none",

        seed=SEED,

        use_cpu=(device == "cpu"),
    )

    # Trainer

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    # Training

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    trainer.train()

    # Final validation

    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULTS")
    print("=" * 70)

    metrics = trainer.evaluate()

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Save best model

    print("\nSaving best model...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()