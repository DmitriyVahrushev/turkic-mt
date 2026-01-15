# turkic-mt

Fine-tuning multilingual MT models for Russian → Bashkir translation.

## Models

- **NLLB-200-3.3B**  - Full training pipeline with DDP
- **MADLAD-400-10B** - Inference and evaluation only

## Usage

```bash
# 1. Filter dataset to remove misaligned samples (where length of one of the texts is signficantly longer the the other), samples with different numbers (e.g. «Уфа, книги, 2021» <-> «Өфө, китап, 2013»)
python filter_dataset.py -i raw_data.parquet -o data/filtered.parquet

# 2. Split into train/valid
python split_dataset.py -i data/filtered.parquet -o data

# 3. Train 
torchrun --nproc_per_node=8 train_nllb.py

# 4. Evaluate model on validation dataset
python evaluate_nllb.py --model-path checkpoints/best_model

# 5. Inference
python nllb_inference.py --model-path checkpoints/best_model
```

## Data

Uses [AigizK/bashkir-russian-parallel-corpora](https://huggingface.co/datasets/AigizK/bashkir-russian-parallel-corpora) .

## Evaluation

ChrF++ metric (with sacrebleu)
