#!/bin/bash
#SBATCH --job-name=dpo-bart-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80gb
#SBATCH --qos=regular
#SBATCH --output= /italian-hyperpartisan-neutralization/slurm_logs/dpo-bart-%j.log
#SBATCH --gres=gpu:V100S:1

# =============================================================================
# DPO Training Script for BART models
# Uses unsloth_env (transformers 5.0.0.dev0)
# =============================================================================

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJECT_ROOT"

SCRIPT_DIR="$PROJECT_ROOT/experiments/scripts"
PYTHON_SCRIPT="$SCRIPT_DIR/train_dpo_2.py"

# === Environment bootstrap ===
module purge
module load cuda/12.1 || true
module load anaconda || true

source  /miniconda3/bin/activate t5_env

echo "Using environment: unsloth_env"
python -c "import transformers; print(f'transformers version: {transformers.__version__}')"

export WANDB_PROJECT=${WANDB_PROJECT:-italian-hyperpartisan-neutralization}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-dpo}

mkdir -p outputs/logs outputs/models/dpo slurm_logs

MODEL_FAMILY=${MODEL_FAMILY:-bart}
MODEL_SIZE=${MODEL_SIZE:-large}
MODEL_NAME=${MODEL_NAME:-}
CONFIG_PATH=${CONFIG_PATH:-experiments/configs/dpo_${MODEL_FAMILY}_${MODEL_SIZE}.yaml}
TRAIN_FILE=${TRAIN_FILE:-data/dpo_train.jsonl}
DEV_FILE=${DEV_FILE:-data/dpo_dev.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/models/dpo}
LOG_DIR=${LOG_DIR:-outputs/logs}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [[ -z "$MODEL_NAME" ]]; then
  case "$MODEL_SIZE" in
    base)
      MODEL_NAME="facebook/bart-base" ;;
    large)
      MODEL_NAME="facebook/bart-large" ;;
    *)
      echo "Unknown MODEL_SIZE=$MODEL_SIZE for BART" >&2
      exit 1 ;;
  esac
fi

# Build SFT model path if not provided (assume SFT was run first)
if [[ -z "$SFT_MODEL_PATH" ]]; then
  SAFE_MODEL_NAME="${MODEL_NAME//\//_}"
  SFT_MODEL_PATH="outputs/models/sft/${SAFE_MODEL_NAME}_${MODEL_SIZE}"
fi

echo "=== DPO Training Configuration ==="
echo "Model: $MODEL_NAME"
echo "Config: $CONFIG_PATH"
echo "Train file: $TRAIN_FILE"
echo "SFT model: $SFT_MODEL_PATH"
echo "=================================="

# Check if SFT model exists
if [[ -d "$SFT_MODEL_PATH" ]]; then
  SFT_ARG="--sft_model_path $SFT_MODEL_PATH"
  echo "Using SFT model as starting point"
else
  SFT_ARG=""
  echo "WARNING: SFT model not found, training DPO from scratch"
fi

python "$PYTHON_SCRIPT" \
  --model_family "$MODEL_FAMILY" \
  --model_size "$MODEL_SIZE" \
  --model_name "$MODEL_NAME" \
  --config "$CONFIG_PATH" \
  --train_file "$TRAIN_FILE" \
  --dev_file "$DEV_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  $SFT_ARG \
  ${EXTRA_ARGS}
