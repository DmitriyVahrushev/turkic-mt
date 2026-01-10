#!/usr/bin/env python3
"""
MADLAD-400-10B Inference Script
Translates Russian text to Bashkir using google/madlad400-10b-mt model.
"""

import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set HF cache directory
os.environ["HF_HOME"] = "/home/ubuntu/dmitrii_projects/turkic-mt/hf_data"

# Model configuration
MODEL_NAME = "google/madlad400-10b-mt"
TGT_LANG = "ba"  # Bashkir language code for MADLAD (ISO 639-1)

# File paths
INPUT_FILE = "test.csv"
OUTPUT_FILE = "submission-madlad400-10b-mt-base.csv"

# Batch size for inference (adjust based on GPU memory)
BATCH_SIZE = 8
MAX_LENGTH = 512


def load_model():
    """Load the MADLAD model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Model loaded successfully on device: {model.device}")
    return model, tokenizer


def translate_batch(texts, model, tokenizer):
    """Translate a batch of texts from Russian to Bashkir."""
    # MADLAD uses <2xx> prefix for target language
    prefixed_texts = [f"<2{TGT_LANG}> {text}" for text in texts]

    inputs = tokenizer(
        prefixed_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    # Move inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=5,
            early_stopping=True
        )

    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations


def main():
    # Load the data
    print(f"Loading data from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows")

    # Load model
    model, tokenizer = load_model()

    # Translate in batches
    translations = []
    texts = df["source_ru"].tolist()

    print("Starting translation...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Translating"):
        batch = texts[i:i + BATCH_SIZE]
        batch_translations = translate_batch(batch, model, tokenizer)
        translations.extend(batch_translations)

    # Create output dataframe
    output_df = pd.DataFrame({
        "id": df["id"],
        "submission": translations
    })

    # Save results
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"Total translations: {len(output_df)}")


if __name__ == "__main__":
    main()
