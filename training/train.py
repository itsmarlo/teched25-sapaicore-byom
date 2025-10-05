"""
Train Hugging Face Transformer for Invoice Classification
Optimized for SAP AI Core deployment
"""

import os
import json
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_label_map(path="label_map.json"):
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        # keep both directions
        id2label = {int(v): k for k, v in m.items()} if all(isinstance(v, int) for v in m.values()) else {int(k): v for k, v in m.items()}
        label2id = {v: k for k, v in id2label.items()}
        return id2label, label2id
    return None, None

class InvoiceClassificationTrainer:
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=5):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.id2label, self.label2id = load_label_map()

    def load_data(self):
        train_df = pd.read_csv('data/train.csv')
        val_df   = pd.read_csv('data/validation.csv')
        # If labels are strings and label_map exists, map them to ids
        if self.label2id and train_df['label'].dtype == object:
            train_df['label'] = train_df['label'].map(self.label2id)
            val_df['label']   = val_df['label'].map(self.label2id)
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
        val_dataset   = Dataset.from_pandas(val_df[['text', 'label']])
        return train_dataset, val_dataset

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=256
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    def train(self, output_dir='./invoice_classifier'):
        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )

        print("Loading and tokenizing datasets...")
        train_ds, val_ds = self.load_data()
        train_ds = train_ds.map(self.tokenize_function, batched=True)
        val_ds   = val_ds.map(self.tokenize_function, batched=True)

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
            logging_dir='./logs',
            logging_steps=20
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        print("Starting training...")
        train_result = trainer.train()
        print("\nTraining completed!")
        print(f"Training loss: {train_result.training_loss:.4f}")

        print("\nEvaluating model...")
        eval_results = trainer.evaluate()
        for k, v in eval_results.items():
            try:
                print(f"  {k}: {float(v):.4f}")
            except Exception:
                print(f"  {k}: {v}")

        print(f"\nSaving model to {output_dir}...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        metadata = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "training_loss": float(train_result.training_loss),
            "eval_results": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in eval_results.items()},
            "id2label": self.id2label
        }
        with open(f'{output_dir}/training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return trainer

def main():
    trainer = InvoiceClassificationTrainer(
        model_name="microsoft/deberta-v3-base",
        num_labels=int(os.getenv("NUM_LABELS", "5"))
    )
    trainer.train(output_dir=os.getenv("OUTPUT_DIR", "./invoice_classifier"))

if __name__ == "__main__":
    main()
