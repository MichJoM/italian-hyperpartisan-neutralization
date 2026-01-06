#!/bin/bash
#SBATCH --job-name=SFT_IT5_large-NOCLASS
#SBATCH --output=slurm_logs/sft_it5_largeNOCLASS_%j.out
#SBATCH --error=slurm_logs/sft_it5_largeNOCLASS_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:V100S:1

# Load modules
module load CUDA/12.4

# Activate environment
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

# Run IT5 training
echo "Starting IT5 Fine-tuning for Italian Text Neutralization..."
python experiments/scripts/sft_it5.py

echo "Done!"
