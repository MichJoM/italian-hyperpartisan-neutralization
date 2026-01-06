#!/bin/bash
#SBATCH --job-name=gemma2-2b-it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170gb
#SBATCH --gres=gpu:V100S:1
#SBATCH --output=/italian-hyperpartisan-neutralization/logs/train_%j.out
#SBATCH --error=/italian-hyperpartisan-neutralization/logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=michelejoshua.maggini@usc.es

# Activate environment
source /italian-hyperpartisan-neutralization/miniconda3/bin/activate neutralization_env

# V100S Compatibility + Memory optimizations
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="7.0"
export NO_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/italian-hyperpartisan-neutralization/miniconda3/envs/neutralization_env/lib/

# Execute with memory-optimized settings
python /italian-hyperpartisan-neutralization/experiments/scripts/FT_LLM.py \
  --model_name "google/gemma-2-2b-it" \
  --train_file "/italian-hyperpartisan-neutralization/data/sft_train_original.jsonl" \
  --dev_file "/italian-hyperpartisan-neutralization/data/sft_dev_original.jsonl" \
  --output_dir "/italian-hyperpartisan-neutralization/outputs/gemma2-v100" \
  --batch_size 1 \
  --grad_accum 32 \
  --max_seq_len 512 \
  --epochs 3 \
  --lora_r 16 \
  --lora_alpha 32 \
  --learning_rate 2e-4 \
  --cpu_offload \
  --no_eval