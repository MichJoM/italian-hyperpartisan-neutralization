#!/bin/bash
#SBATCH --job-name=qlora-flan-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200gb
#SBATCH --gres=gpu:A100_80:1
#SBATCH --qos=regular
#SBATCH --output= /italian-hyperpartisan-neutralization/slurm_logs/qlora-flan-large-%j.log
#SBATCH --error= /italian-hyperpartisan-neutralization/slurm_logs/qlora-flan-large-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJECT_ROOT"

SCRIPT_PATH="experiments/scripts/sft_flan_large.py"

if command -v module &>/dev/null; then
  module purge >/dev/null 2>&1 || true
fi

source  /miniconda3/bin/activate t5_env

echo "Using environment: $(conda env list | grep '*' | awk '{print $1}')"
python -c "import transformers; print('transformers version:', transformers.__version__)"

TRAIN_FILE=${TRAIN_FILE:-data/sft_train_original.jsonl}
DEV_FILE=${DEV_FILE:-data/sft_dev_original.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/models/sft_flan_large}
LOG_DIR=${LOG_DIR:-outputs/logs}

PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-4}
PER_DEVICE_EVAL_BS=${PER_DEVICE_EVAL_BS:-8}
GRAD_ACC=${GRAD_ACC:-4}
LR=${LR:-1.5e-4}
EPOCHS=${EPOCHS:-2}
MAX_INPUT=${MAX_INPUT:-512}
MAX_TARGET=${MAX_TARGET:-256}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" slurm_logs

python "$SCRIPT_PATH" \
  --train_file "$TRAIN_FILE" \
  --dev_file "$DEV_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BS" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BS" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --learning_rate "$LR" \
  --num_train_epochs "$EPOCHS" \
  --max_input_length "$MAX_INPUT" \
  --max_target_length "$MAX_TARGET" \
  --gradient_checkpointing \
  --group_by_length \
  --bf16 \
  --use_wandb
