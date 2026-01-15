#!/usr/bin/env python3
"""
Dataset Filtering Script
Removes outliers, misaligned samples, and samples with mismatched numbers.
"""

import argparse
import re
import os
import pandas as pd

# Filtering thresholds
MIN_LENGTH = 2
MAX_LENGTH = 2000
MAX_LENGTH_RATIO = 2.0


def extract_numbers(text: str) -> list[str]:
    """Extract all numbers from text."""
    return re.findall(r'\d+', text)


def filter_by_length(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Remove samples where any text is too short or too long."""
    # Calculate lengths
    ru_len = df["ru"].str.len()
    ba_len = df["ba"].str.len()

    # Filter: both texts must be within bounds
    mask = (
        (ru_len >= MIN_LENGTH) & (ru_len <= MAX_LENGTH) &
        (ba_len >= MIN_LENGTH) & (ba_len <= MAX_LENGTH)
    )

    filtered_df = df[mask].copy()

    too_short = ((ru_len < MIN_LENGTH) | (ba_len < MIN_LENGTH)).sum()
    too_long = ((ru_len > MAX_LENGTH) | (ba_len > MAX_LENGTH)).sum()

    return filtered_df, too_short, too_long


def filter_by_alignment(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove samples where length ratio between languages exceeds threshold."""
    initial_count = len(df)

    ru_len = df["ru"].str.len()
    ba_len = df["ba"].str.len()

    # Calculate ratio (always >= 1)
    ratio = ru_len / ba_len
    ratio = ratio.where(ratio >= 1, 1 / ratio)

    mask = ratio <= MAX_LENGTH_RATIO
    filtered_df = df[mask].copy()

    removed = initial_count - len(filtered_df)
    return filtered_df, removed


def filter_by_numbers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove samples where numbers don't match between languages."""
    initial_count = len(df)

    def numbers_match(row) -> bool:
        ru_numbers = extract_numbers(str(row["ru"]))
        ba_numbers = extract_numbers(str(row["ba"]))
        return sorted(ru_numbers) == sorted(ba_numbers)

    mask = df.apply(numbers_match, axis=1)
    filtered_df = df[mask].copy()

    removed = initial_count - len(filtered_df)
    return filtered_df, removed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter dataset: remove outliers, misaligned samples, and number mismatches."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input parquet file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output parquet file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    initial_count = len(df)
    print(f"Initial samples: {initial_count:,}")

    # Step 1: Filter by length
    print(f"\n[1/3] Filtering by length (min={MIN_LENGTH}, max={MAX_LENGTH})...")
    df, too_short, too_long = filter_by_length(df)
    print(f"  Removed {too_short:,} samples (too short, <{MIN_LENGTH} chars)")
    print(f"  Removed {too_long:,} samples (too long, >{MAX_LENGTH} chars)")
    print(f"  Remaining: {len(df):,}")

    # Step 2: Filter by alignment
    print(f"\n[2/3] Filtering misaligned samples (ratio > {MAX_LENGTH_RATIO}x)...")
    df, misaligned = filter_by_alignment(df)
    print(f"  Removed {misaligned:,} samples (length ratio > {MAX_LENGTH_RATIO}x)")
    print(f"  Remaining: {len(df):,}")

    # Step 3: Filter by number mismatch
    print("\n[3/3] Filtering samples with mismatched numbers...")
    df, number_mismatch = filter_by_numbers(df)
    print(f"  Removed {number_mismatch:,} samples (numbers don't match)")
    print(f"  Remaining: {len(df):,}")

    # Summary
    total_removed = initial_count - len(df)
    print("\n" + "=" * 50)
    print("FILTERING SUMMARY")
    print("=" * 50)
    print(f"Initial samples:      {initial_count:,}")
    print(f"Too short (<{MIN_LENGTH}):       {too_short:,}")
    print(f"Too long (>{MAX_LENGTH}):     {too_long:,}")
    print(f"Misaligned (>{MAX_LENGTH_RATIO}x):      {misaligned:,}")
    print(f"Number mismatch:      {number_mismatch:,}")
    print(f"Total removed:        {total_removed:,} ({100*total_removed/initial_count:.2f}%)")
    print(f"Final samples:        {len(df):,} ({100*len(df)/initial_count:.2f}%)")

    # Save filtered data
    print(f"\nSaving filtered data to {args.output}")
    df.to_parquet(args.output, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
