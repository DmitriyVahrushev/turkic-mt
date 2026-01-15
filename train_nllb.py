#!/usr/bin/env python3
"""
NLLB-200-3.3B Full Finetuning Script
Russian → Bashkir translation with DDP training.

Launch with:
    torchrun --nproc_per_node=8 train_nllb.py
"""

import os
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import sacrebleu


os.environ["NCCL_TIMEOUT"] = "3600"

MODEL_NAME = "facebook/nllb-200-3.3B"
SRC_LANG = "rus_Cyrl"  # Russian
TGT_LANG = "bak_Cyrl"  # Bashkir

# Data
TRAIN_FILE = "data/train.parquet"
VALID_FILE = "data/valid.parquet"

# Training hyperparameters
BATCH_SIZE = 4 * 8  # Per GPU
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 4 * 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 4 * 1000
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 512
GRADIENT_CLIP = 1.0

# Evaluation
EVAL_STEPS = 500  # Evaluate every N steps
EVAL_SUBSET_SIZE = 1000  # Subset for step-wise evaluation
NUM_BEAMS = 5

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "runs"

# Random seed
SEED = 42


class TranslationDataset(Dataset):
    """Dataset for Russian-Bashkir translation."""

    def __init__(self, parquet_path: str, tokenizer, max_length: int = 256):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source_text = str(row["ru"])
        target_text = str(row["ba"])

        # Tokenize source
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(
                target_text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )

        return {
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
            "labels": target_encoding["input_ids"],
        }


class DataCollator:
    """Collate function with dynamic padding."""

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.label_pad_token_id = -100

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Pad sequences
        max_input_len = min(max(len(x) for x in input_ids), self.max_length)
        max_label_len = min(max(len(x) for x in labels), self.max_length)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for inp, att, lab in zip(input_ids, attention_mask, labels):
            # Pad input
            pad_len = max_input_len - len(inp)
            padded_input_ids.append(inp + [self.pad_token_id] * pad_len)
            padded_attention_mask.append(att + [0] * pad_len)

            # Pad labels
            lab_pad_len = max_label_len - len(lab)
            padded_labels.append(lab + [self.label_pad_token_id] * lab_pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def evaluate_chrf(
    model,
    tokenizer,
    dataloader,
    device,
    max_samples: int = None,
    num_beams: int = 5,
):
    """Evaluate model and compute ChrF++ score."""
    model.eval()

    predictions = []
    references = []
    total_loss = 0.0
    num_batches = 0

    # Get target language token id for generation
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if max_samples and batch_idx * dataloader.batch_size >= max_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Compute loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            num_batches += 1

            # Generate translations
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=MAX_SEQ_LENGTH,
                num_beams=num_beams,
                early_stopping=True,
            )

            # Decode predictions
            pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(pred_texts)

            # Decode references (replace -100 with pad token for decoding)
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            ref_texts = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            references.extend(ref_texts)

    # Compute ChrF++
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    model.train()

    return {
        "chrf++": chrf.score,
        "loss": avg_loss,
        "num_samples": len(predictions),
    }


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return dist.get_rank() == 0


def train():
    """Main training function."""
    # Setup DDP
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    set_seed(SEED + local_rank)

    if is_main_process():
        print("=" * 60)
        print("NLLB-200-3.3B Finetuning")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {BATCH_SIZE}")
        print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"Effective batch size: {BATCH_SIZE * world_size * GRADIENT_ACCUMULATION_STEPS}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Epochs: {NUM_EPOCHS}")
        print("=" * 60)

    # Create directories
    if is_main_process():
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

    # Load tokenizer
    if is_main_process():
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    # Load model
    if is_main_process():
        print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # Load datasets
    if is_main_process():
        print("Loading datasets...")
    train_dataset = TranslationDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LENGTH)
    valid_dataset = TranslationDataset(VALID_FILE, tokenizer, MAX_SEQ_LENGTH)

    if is_main_process():
        print(f"Train samples: {len(train_dataset):,}")
        print(f"Valid samples: {len(valid_dataset):,}")

    # Create data loaders
    collator = DataCollator(tokenizer, MAX_SEQ_LENGTH)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=SEED,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler
    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    # TensorBoard
    writer = None
    if is_main_process():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(os.path.join(LOG_DIR, f"nllb_finetune_{timestamp}"))

    # Training loop
    global_step = 0
    best_chrf = 0.0

    if is_main_process():
        print("\nStarting training...")

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            disable=not is_main_process(),
        )

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            # Backward pass
            loss.backward()

            # Update weights
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log to TensorBoard
                if is_main_process() and writer:
                    writer.add_scalar("train/loss", loss.item() * GRADIENT_ACCUMULATION_STEPS, global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

                # Step-wise evaluation
                if global_step % EVAL_STEPS == 0 and is_main_process():
                    print(f"\nStep {global_step}: Running evaluation on subset...")

                    # Create subset loader for quick eval
                    subset_indices = list(range(min(EVAL_SUBSET_SIZE, len(valid_dataset))))
                    subset = torch.utils.data.Subset(valid_dataset, subset_indices)
                    subset_loader = DataLoader(
                        subset,
                        batch_size=BATCH_SIZE,
                        collate_fn=collator,
                        num_workers=2,
                    )

                    eval_results = evaluate_chrf(
                        model.module,
                        tokenizer,
                        subset_loader,
                        device,
                        num_beams=NUM_BEAMS,
                    )

                    print(f"  Subset ChrF++: {eval_results['chrf++']:.2f}")
                    print(f"  Subset Loss: {eval_results['loss']:.4f}")

                    if writer:
                        writer.add_scalar("eval_subset/chrf++", eval_results["chrf++"], global_step)
                        writer.add_scalar("eval_subset/loss", eval_results["loss"], global_step)

        # End of epoch evaluation
        dist.barrier()

        if is_main_process():
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"\nEpoch {epoch + 1} completed. Avg train loss: {avg_epoch_loss:.4f}")
            print(f"Running validation on {EVAL_SUBSET_SIZE} samples...")

            # Create subset loader for evaluation
            subset_indices = list(range(min(EVAL_SUBSET_SIZE, len(valid_dataset))))
            subset = torch.utils.data.Subset(valid_dataset, subset_indices)
            eval_loader = DataLoader(
                subset,
                batch_size=BATCH_SIZE,
                collate_fn=collator,
                num_workers=2,
            )

            eval_results = evaluate_chrf(
                model.module,
                tokenizer,
                eval_loader,
                device,
                num_beams=NUM_BEAMS,
            )

            print(f"  Valid ChrF++: {eval_results['chrf++']:.2f}")
            print(f"  Valid Loss: {eval_results['loss']:.4f}")

            if writer:
                writer.add_scalar("eval/chrf++", eval_results["chrf++"], global_step)
                writer.add_scalar("eval/loss", eval_results["loss"], global_step)
                writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch + 1)

            # Save checkpoint
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch + 1}")
            print(f"Saving checkpoint to {checkpoint_path}")
            model.module.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

            # Save best model
            if eval_results["chrf++"] > best_chrf:
                best_chrf = eval_results["chrf++"]
                best_path = os.path.join(CHECKPOINT_DIR, "best_model")
                print(f"New best ChrF++: {best_chrf:.2f}. Saving to {best_path}")
                model.module.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)

        dist.barrier()

    # Cleanup
    if is_main_process() and writer:
        writer.close()
        print("\nTraining completed!")
        print(f"Best ChrF++: {best_chrf:.2f}")

    cleanup_ddp()


if __name__ == "__main__":
    train()
