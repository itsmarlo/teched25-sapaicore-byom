"""
Train Hugging Face Transformer for Invoice Classification
Optimized for SAP AI Core deployment
"""

# ===== Missing and required imports =====
import os
import json
import torch
import pandas as pd
import numpy as np                      # <-- add missing import
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path                 # <-- helpful for safe path handling
# ========================================


def load_label_map(path=None):
    """Load label mapping from JSON file"""
    if path is None:
        data_path = os.getenv("DATA_PATH", "/app/data")
        path = os.path.join(data_path, "label_map.json")
    
    if os.path.exists(path):
        print(f"Loading label map from: {path}")
        with open(path) as f:
            m = json.load(f)
        # keep both directions
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


class InvoiceClassificationTrainer:
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=5):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.id2label, self.label2id = load_label_map()

    def load_data(self):
        """Load training and validation data from artifact mount point"""
        data_path = os.getenv("DATA_PATH", "/app/data")
        print(f"\n=== Loading Data ===")
        print(f"Data path: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        print(f"Directory contents: {os.listdir(data_path)}")

        train_path = os.path.join(data_path, "train.csv")
        val_path = os.path.join(data_path, "validation.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found at {val_path}")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        print(f"Loaded {len(train_df)} training and {len(val_df)} validation samples")

        # Map string labels → ids if label_map exists
        if self.label2id and train_df["label"].dtype == object:
            train_df["label"] = train_df["label"].map(self.label2id)
            val_df["label"] = val_df["label"].map(self.label2id)

        train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
        val_dataset = Dataset.from_pandas(val_df[["text", "label"]])
        return train_dataset, val_dataset

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
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
        """Train the model and save to output directory"""
        print("\n=== Starting Training Process ===")
        print(f"Model: {self.model_name}")
        print(f"Number of labels: {self.num_labels}")
        print(f"Output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        print("\nLoading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        print("\nLoading and tokenizing datasets...")
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
        print("\n=== Training Completed ===")

        print("\n=== Evaluating Model ===")
        eval_results = trainer.evaluate()
        print(eval_results)

        print("\n=== Saving Model to Artifact Directory ===")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        metadata = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "training_loss": float(train_result.training_loss),
            "eval_results": eval_results,
            "id2label": self.id2label,
        }
        metadata_path = os.path.join(output_dir, "training_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

        print("\n=== Training Pipeline Completed Successfully ===")
        return trainer


def main():
    print("=" * 60)
    print("Invoice Classification Training - SAP AI Core")
    print("=" * 60)

    model_name = os.getenv("MODEL_NAME", "microsoft/deberta-v3-base")
    num_labels = int(os.getenv("NUM_LABELS", "5"))
    output_dir = os.getenv("OUTPUT_DIR", "/mnt/models")

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Number of labels: {num_labels}")
    print(f"  Output directory: {output_dir}")
    print(f"  Data path: {os.getenv('DATA_PATH', '/app/data')}")

    try:
        trainer = InvoiceClassificationTrainer(model_name=model_name, num_labels=num_labels)
        trainer.train(output_dir=output_dir)
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed with error:")
        print(f"  {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
