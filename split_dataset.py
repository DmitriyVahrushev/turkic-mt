#!/usr/bin/env python3
"""
Dataset Splitting Script
Splits the Bashkir-Russian parallel corpus into train/validation sets (95/5).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
SOURCE_PARQUET = "/home/ubuntu/dmitrii_projects/turkic-mt/hf_data/hub/datasets--AigizK--bashkir-russian-parallel-corpora/snapshots/0cddd5ffe7fffa8e23fd64a94a52429668513bbc/data/train-00000-of-00001.parquet"
OUTPUT_DIR = "data"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.parquet")
VALID_FILE = os.path.join(OUTPUT_DIR, "valid.parquet")

# Split configuration
TRAIN_RATIO = 0.95
VALID_RATIO = 0.05
RANDOM_SEED = 42


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print(f"Loading data from {SOURCE_PARQUET}")
    df = pd.read_parquet(SOURCE_PARQUET)
    print(f"Total samples: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    # Shuffle and split
    print(f"\nSplitting data: {TRAIN_RATIO*100:.0f}% train, {VALID_RATIO*100:.0f}% validation")
    train_df, valid_df = train_test_split(
        df,
        test_size=VALID_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print(f"Train samples: {len(train_df):,}")
    print(f"Validation samples: {len(valid_df):,}")

    # Save splits
    print(f"\nSaving train set to {TRAIN_FILE}")
    train_df.to_parquet(TRAIN_FILE, index=False)

    print(f"Saving validation set to {VALID_FILE}")
    valid_df.to_parquet(VALID_FILE, index=False)

    print("\nDone!")
    print(f"  Train: {TRAIN_FILE} ({len(train_df):,} samples)")
    print(f"  Valid: {VALID_FILE} ({len(valid_df):,} samples)")


if __name__ == "__main__":
    main()
