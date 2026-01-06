#!/bin/bash
#SBATCH --job-name=test-llama-3.1
#SBATCH --gres=gpu:A100_40:1
#SBATCH --mem=70gb
#SBATCH --output=logs/llama-3.1%A.out
#SBATCH --error=logs/llama-3.1%A.err

source ~/miniconda3/bin/activate neutralization_env

# Set the model directly
MODEL="llama-3.1-8b"
OUTPUT_DIR="outputs/${MODEL}-neutral-rewriter"

python  /italian-hyperpartisan-neutralization/experiments/scripts/FT_other_unsloth.py \
  --model $MODEL \
  --train_file data/sft_train_original.jsonl \
  --dev_file data/sft_dev_original.jsonl \
  --output_dir "$OUTPUT_DIR" \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum 8