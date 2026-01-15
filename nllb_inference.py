#!/usr/bin/env python3
"""
NLLB-200-3.3B Inference Script
Translates Russian text to Bashkir using facebook/nllb-200-3.3B model.
Supports both base model and finetuned checkpoints.
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set HF cache directory
os.environ["HF_HOME"] = "/home/ubuntu/dmitrii_projects/turkic-mt/hf_data"

# Model configuration
BASE_MODEL = "facebook/nllb-200-3.3B"
SRC_LANG = "rus_Cyrl"  # Russian
TGT_LANG = "bak_Cyrl"  # Bashkir

# File paths
INPUT_FILE = "test.csv"
OUTPUT_FILE = "submission.csv"

# Batch size for inference (adjust based on GPU memory)
BATCH_SIZE = 8
MAX_LENGTH = 4096


def parse_args():
    parser = argparse.ArgumentParser(description="NLLB Russian to Bashkir translation")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to finetuned model checkpoint. If not specified, uses base model."
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
            max_length=MAX_LENGTH,
            num_beams=5,
            early_stopping=True
        )

    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations


def main():
    args = parse_args()
    model_path = args.model_path if args.model_path else BASE_MODEL

    df = pd.read_csv(INPUT_FILE)
    model, tokenizer = load_model(model_path)
    translations = []
    texts = df["source_ru"].tolist()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Translating"):
        batch = texts[i:i + BATCH_SIZE]
        batch_translations = translate_batch(batch, model, tokenizer)
        translations.extend(batch_translations)

    output_df = pd.DataFrame({
        "id": df["id"],
        "submission": translations
    })

    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"Total translations: {len(output_df)}")


if __name__ == "__main__":
    main()
