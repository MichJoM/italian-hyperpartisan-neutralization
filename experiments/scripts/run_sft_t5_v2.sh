#!/bin/bash
#SBATCH --job-name=SFT_bart-base-NOCLASS
#SBATCH --output=slurm_logs/sft_bart-base-NOCLASS_v2_%j.out
#SBATCH --error=slurm_logs/sft_bart-base-NOCLASS_v2_%j.err
#SBATCH --partition=defaultPartition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:V100S:1

# Load modules
module load CUDA/12.4

# Activate environment - use t5_env which has sentencepiece
source  /miniconda3/etc/profile.d/conda.sh
conda activate t5_env

# Navigate to project
cd  /italian-hyperpartisan-neutralization

# Create logs directory if needed
mkdir -p slurm_logs

# Print GPU info
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "================"

# Run training
# Options:
# 1. Debug mode (10 samples, fast test):
#    python experiments/scripts/sft_train_2.py  # with DEBUG=True in script
#
# 2. FLAN-T5-base with classifier:
#    Edit MODEL_NAME="google/flan-t5-base" and USE_NEUTRALITY_GUIDANCE=True
#
# 3. FLAN-T5-large without classifier (OOM test):
#    Edit MODEL_NAME="google/flan-t5-large" and USE_NEUTRALITY_GUIDANCE=False

echo "Starting SFT Training v2 (T5 Fixed)..."
python experiments/scripts/sft_train_2.py

echo "Done!"
