#!/usr/bin/env python3
"""
NLLB Evaluation Script
Evaluates NLLB model on validation samples and computes ChrF++ metric.
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sacrebleu

# Model configuration
BASE_MODEL = "facebook/nllb-200-3.3B"
SRC_LANG = "rus_Cyrl"  # Russian
TGT_LANG = "bak_Cyrl"  # Bashkir

# File paths
VALID_FILE = "data/valid.parquet"

# Evaluation settings
NUM_SAMPLES = 1000
BATCH_SIZE = 8
MAX_LENGTH = 1024
NUM_BEAMS = 5
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NLLB model on validation set")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to finetuned model checkpoint. If not specified, uses base model."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help=f"Number of samples to evaluate (default: {NUM_SAMPLES})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for inference (default: {BATCH_SIZE})"
    )
    return parser.parse_args()


def load_model(model_path: str):
    """Load the NLLB model and tokenizer."""
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Model loaded successfully on device: {model.device}")
    return model, tokenizer


def translate_batch(texts, model, tokenizer):
    """Translate a batch of texts from Russian to Bashkir."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    # Move inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get the target language token id
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True
        )

    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations


def main():
    args = parse_args()

    # Determine model path
    model_path = args.model_path if args.model_path else BASE_MODEL

    # Load validation data
    print(f"Loading validation data from {VALID_FILE}")
    df = pd.read_parquet(VALID_FILE)
    print(f"Total validation samples: {len(df):,}")

    # Select first N samples (same as training script subset)
    df_subset = df.head(args.num_samples)
    print(f"Evaluating on {len(df_subset):,} samples")

    # Load model
    model, tokenizer = load_model(model_path)

    # Translate in batches
    predictions = []
    source_texts = df_subset["ru"].tolist()
    reference_texts = df_subset["ba"].tolist()

    print("Starting translation...")
    for i in tqdm(range(0, len(source_texts), args.batch_size), desc="Translating"):
        batch = source_texts[i:i + args.batch_size]
        batch_translations = translate_batch(batch, model, tokenizer)
        predictions.extend(batch_translations)

    # Compute ChrF++
    chrf = sacrebleu.corpus_chrf(predictions, [reference_texts], word_order=2)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Samples: {len(predictions)}")
    print(f"ChrF++: {chrf.score:.2f}")
    print("=" * 50)

    # Show some examples
    print("\nSample translations:")
    for i in range(min(5, len(predictions))):
        print(f"\n[{i+1}]")
        print(f"  Source (RU): {source_texts[i][:100]}...")
        print(f"  Reference (BA): {reference_texts[i][:100]}...")
        print(f"  Prediction: {predictions[i][:100]}...")


if __name__ == "__main__":
    main()
