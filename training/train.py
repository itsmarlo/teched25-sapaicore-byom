"""
Train Hugging Face Transformer for Invoice Classification
Optimized for SAP AI Core deployment
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path


# ======================================================
# Utility: Load label map
# ======================================================
def load_label_map(path=None):
    """Load label mapping from JSON file"""
    if path is None:
        data_path = os.getenv("DATA_PATH", "/mnt/data")
        path = os.path.join(data_path, "label_map.json")

    if os.path.exists(path):
        print(f"Loading label map from: {path}")
        with open(path) as f:
            m = json.load(f)
        # handle both string→int and int→string mappings
        id2label = (
            {int(v): k for k, v in m.items()}
            if all(isinstance(v, int) for v in m.values())
            else {int(k): v for k, v in m.items()}
        )
        label2id = {v: k for k, v in id2label.items()}
        print(f"Loaded {len(id2label)} labels: {id2label}")
        return id2label, label2id

    print(f"Warning: label_map.json not found at {path}")
    return None, None


# ======================================================
# Trainer class
# ======================================================
class InvoiceClassificationTrainer:
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=5):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.id2label, self.label2id = load_label_map()

    def load_data(self):
        """Load train/validation CSVs from mounted S3 dataset"""
        data_path = os.getenv("DATA_PATH", "/mnt/data")
        print(f"\n=== Loading Data ===")
        print(f"Data path: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        print(f"Contents of {data_path}: {os.listdir(data_path)}")

        train_path = os.path.join(data_path, "train.csv")
        val_path = os.path.join(data_path, "validation.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data missing: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data missing: {val_path}")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        print(f"Loaded {len(train_df)} train / {len(val_df)} validation samples")

        # map text labels → ints if label_map available
        if self.label2id and train_df["label"].dtype == object:
            train_df["label"] = train_df["label"].map(self.label2id)
            val_df["label"] = val_df["label"].map(self.label2id)

        train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
        val_dataset = Dataset.from_pandas(val_df[["text", "label"]])
        return train_dataset, val_dataset

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=256
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train(self, output_dir="/mnt/models"):
        """Train and save model for AI Core artifact upload"""
        print("\n=== Training Initialization ===")
        print(f"Model: {self.model_name}")
        print(f"Num labels: {self.num_labels}")
        print(f"Output dir: {output_dir}")

        # Ensure output directory exists so Argo can collect it
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("\nLoading tokenizer/model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        print("\nPreparing datasets...")
        train_ds, val_ds = self.load_data()
        train_ds = train_ds.map(self.tokenize_function, batched=True)
        val_ds = val_ds.map(self.tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            push_to_hub=False,
            logging_dir=f"{output_dir}/logs",
            logging_steps=20,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        print("\n=== Training Started ===")
        train_result = trainer.train()
        print("\n=== Training Finished ===")

        print("\nEvaluating model...")
        eval_results = trainer.evaluate()
        print(eval_results)

        print("\nSaving model to artifact path...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training metadata
        metadata = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "training_loss": float(train_result.training_loss),
            "eval_results": eval_results,
            "id2label": self.id2label,
        }
        meta_path = Path(output_dir) / "training_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {meta_path}")

        print("\n=== Training Completed Successfully ===")
        return trainer


# ======================================================
# Main entry point
# ======================================================
def main():
    print("=" * 60)
    print("Invoice Classification Training - SAP AI Core")
    print("=" * 60)

    model_name = os.getenv("MODEL_NAME", "microsoft/deberta-v3-base")
    num_labels = int(os.getenv("NUM_LABELS", "5"))
    output_dir = os.getenv("OUTPUT_DIR", "/mnt/models")

    print(f"\nConfiguration:")
    print(f"  MODEL_NAME: {model_name}")
    print(f"  NUM_LABELS: {num_labels}")
    print(f"  OUTPUT_DIR: {output_dir}")
    print(f"  DATA_PATH: {os.getenv('DATA_PATH', '/mnt/data')}")

    try:
        trainer = InvoiceClassificationTrainer(model_name=model_name, num_labels=num_labels)
        trainer.train(output_dir=output_dir)
        print("\n✓ Training finished successfully.")
    except Exception as e:
        print(f"\n✗ Training failed: {type(e).__name__} - {e}")
        raise


if __name__ == "__main__":
    main()
