#!/bin/bash
#SBATCH --job-name=llama-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --gres=gpu:A100_40:1
#SBATCH --output=/italian-hyperpartisan-neutralization/logs/eval_%j.out
#SBATCH --error= /italian-hyperpartisan-neutralization/logs/eval_%j.err

set -euo pipefail

# Activate environment
source  /miniconda3/bin/activate unsloth_env

export HF_TOKEN="***REMOVED***"

PROJECT_ROOT=" /italian-hyperpartisan-neutralization"
PYTHON_SCRIPT="$PROJECT_ROOT/experiments/scripts/evaluate_llama.py"

echo "========================================"
echo "Starting evaluation of fine-tuned Llama"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

python "$PYTHON_SCRIPT"

echo "========================================"
echo "Evaluation completed at: $(date)"
echo "========================================"
