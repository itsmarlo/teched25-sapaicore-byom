"""
Prepare blocked invoice dataset for training
Simulates extraction from SAP S/4HANA
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

# Simulate blocked invoice data from SAP S/4HANA
def generate_sample_data():
    """Generate synthetic blocked invoice data"""
    
    categories = {
        0: "Price Variance",
        1: "Quantity Mismatch",
        2: "Missing PO Reference",
        3: "Three-Way Match Failure",
        4: "Supplier Issues"
    }
    
    # Sample invoice descriptions and blocking reasons
    samples = [
        ("Invoice amount 5000 EUR exceeds PO amount 4500 EUR by 11%", 0),
        ("Price difference detected: invoiced 100 EUR per unit vs PO 90 EUR", 0),
        ("Unit price variance of 15% requires approval", 0),
        ("Received quantity 500 units but invoice shows 550 units", 1),
        ("Quantity on invoice does not match goods receipt", 1),
        ("Delivery note shows 100 items, invoice has 120 items", 1),
        ("Purchase order reference missing on invoice document", 2),
        ("Cannot match invoice to any existing PO in system", 2),
        ("PO number not found in SAP system", 2),
        ("Goods receipt not posted for this purchase order", 3),
        ("Invoice received before goods receipt posting", 3),
        ("Three-way match failed: PO-GR-Invoice discrepancy", 3),
        ("Supplier blocked due to compliance issues", 4),
        ("Vendor on credit hold cannot process payment", 4),
        ("Supplier bank details pending verification", 4),
    ]
    
    # Expand dataset with variations
    data = []
    for text, label in samples * 20:  # Duplicate for larger dataset
        # Add slight variations
        variations = [
            text,
            text.replace("EUR", "USD"),
            f"URGENT: {text}",
            f"ALERT: {text.lower()}",
        ]
        for var in variations:
            data.append({
                "text": var,
                "label": label,
                "label_name": categories[label]
            })
    
    return pd.DataFrame(data)

def prepare_dataset():
    """Prepare and split dataset"""
    
    df = generate_sample_data()
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # Save datasets
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/validation.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    # Save label mapping
    label_map = {
        0: "Price Variance",
        1: "Quantity Mismatch",
        2: "Missing PO Reference",
        3: "Three-Way Match Failure",
        4: "Supplier Issues"
    }
    
    with open('data/label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"Dataset prepared:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"\nClass distribution:")
    print(train_df['label_name'].value_counts())
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    prepare_dataset()