#!/bin/bash
#SBATCH --job-name=gemma3-unsloth-4b-v100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180gb                    # Slightly more than before for safety
#SBATCH --gres=gpu:V100S:1
#SBATCH --output= /italian-hyperpartisan-neutralization/logs/unsloth_train_%j.out
#SBATCH --error= /italian-hyperpartisan-neutralization/logs/unsloth_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=michelejoshua.maggini@usc.es

# Activate your Unsloth environment
source  /miniconda3/bin/activate neutralization_env

# Optional: V100S is fine with modern kernels, but these help stability if needed
export CUDA_LAUNCH_BLOCKING=0          # Usually not needed, but safe
export TORCH_CUDA_ARCH_LIST="7.0"      # Matches V100S
# No need to disable flash attention — V100S works with SDPA (Unsloth uses it by default)

# Optional: Force FP16 explicitly (Unsloth handles it well on V100)
export TORCH_DTYPE=fp16

# Run your Unsloth script
python  /italian-hyperpartisan-neutralization/experiments/scripts/FT_unsloth.py

echo "Job finished at $(date)"