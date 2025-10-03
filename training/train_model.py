"""
Train Hugging Face Transformer for Invoice Classification
Optimized for SAP AI Core deployment
"""
import os
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
import json

class InvoiceClassificationTrainer:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=5):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        
    def load_data(self):
        """Load prepared datasets"""
        train_df = pd.read_csv('data/train.csv')
        val_df = pd.read_csv('data/validation.csv')
        
        # Convert to HuggingFace Dataset
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
        
        return train_dataset, val_dataset
    
    def tokenize_function(self, examples):
        """Tokenize text data"""
        return self.tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True,
            max_length=128
        )
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, output_dir='./model_output'):
        """Train the classification model"""
        
        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        print("Loading and tokenizing datasets...")
        train_dataset, val_dataset = self.load_data()
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Training arguments
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
            logging_steps=10,
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        print("Starting training...")
        train_result = trainer.train()
        
        print("\nTraining completed!")
        print(f"Training loss: {train_result.training_loss:.4f}")
        
        # Evaluate on validation set
        print("\nEvaluating model...")
        eval_results = trainer.evaluate()
        print(f"Validation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        # Save model and tokenizer
        print(f"\nSaving model to {output_dir}...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metadata
        metadata = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "training_loss": train_result.training_loss,
            "eval_results": eval_results,
        }
        
        with open(f'{output_dir}/training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return trainer

def main():
    """Main training pipeline"""
    
    # Initialize trainer
    trainer = InvoiceClassificationTrainer(
        model_name="distilbert-base-uncased",
        num_labels=5
    )
    
    # Train model
    trained_model = trainer.train(output_dir='./invoice_classifier')
    
    print("\n✅ Model training completed successfully!")
    print("📁 Model saved to: ./invoice_classifier")

if __name__ == "__main__":
    main()