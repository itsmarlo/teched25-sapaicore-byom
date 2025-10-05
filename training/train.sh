#!/usr/bin/env bash
set -e  # Exit immediately if a command fails

echo "=== Invoice Classifier Training Start ==="

# --- Step 1: Prepare data path ---
# Your training code expects /app/data/, but AI Core mounts datasets to /mnt/data
if [ -d "/mnt/data" ]; then
  echo "Linking /mnt/data -> /app/data ..."
  rm -rf /app/data
  ln -s /mnt/data /app/data
else
  echo "Warning: /mnt/data not found, continuing without mount"
fi

# --- Step 2: Prepare model output path ---
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/models}
mkdir -p "$OUTPUT_DIR"

echo "Using OUTPUT_DIR: $OUTPUT_DIR"
echo "Using NUM_LABELS: ${NUM_LABELS:-5}"

# --- Step 3: Launch training ---
python /app/train.py

echo "=== Training complete. Artifacts saved to $OUTPUT_DIR ==="