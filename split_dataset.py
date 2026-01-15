#!/usr/bin/env python3
"""
Dataset Splitting Script
Splits the Bashkir-Russian parallel corpus into train/validation sets (95/5).
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


TRAIN_RATIO = 0.95
VALID_RATIO = 0.05
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/validation sets (95/5)."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input parquet file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data",
        help="Output directory for train/valid parquet files (default: data)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_file = os.path.join(args.output_dir, "train.parquet")
    valid_file = os.path.join(args.output_dir, "valid.parquet")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
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
    print(f"\nSaving train set to {train_file}")
    train_df.to_parquet(train_file, index=False)

    print(f"Saving validation set to {valid_file}")
    valid_df.to_parquet(valid_file, index=False)

    print("\nDone!")
    print(f"  Train: {train_file} ({len(train_df):,} samples)")
    print(f"  Valid: {valid_file} ({len(valid_df):,} samples)")


if __name__ == "__main__":
    main()
