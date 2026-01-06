#!/bin/bash
#SBATCH --job-name=dpo-it5-base-NOCLASS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80gb
#SBATCH --qos=regular
#SBATCH --output= /italian-hyperpartisan-neutralization/slurm_logs/dpo-bart-large-%j.log
#SBATCH --gres=gpu:L4:1

# =============================================================================
# DPO Training Script for IT5 and mT5 models
# Uses t5_env with transformers 4.44.2 (stable version for IT5/mT5)
# =============================================================================

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJECT_ROOT"

SCRIPT_DIR="$PROJECT_ROOT/experiments/scripts"
PYTHON_SCRIPT="$SCRIPT_DIR/train_dpo_full_eval.py"

# === Environment bootstrap ===
# Use t5_env for IT5/mT5 compatibility
module purge
module load cuda/12.1 || true
module load anaconda || true

source  /miniconda3/bin/activate t5_env

echo "Using environment: t5_env"
python -c "import transformers; print(f'transformers version: {transformers.__version__}')"

export WANDB_PROJECT=${WANDB_PROJECT:-italian-hyperpartisan-neutralization}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-dpo}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}

mkdir -p outputs/logs outputs/logs/dpo_eval outputs/models/dpo slurm_logs

# Default to IT5, can be overridden with MODEL_FAMILY=mt5
MODEL_FAMILY=${MODEL_FAMILY:-it5}
# Canonicalize model family (allow flan-t5 input but use underscores everywhere else)
MODEL_FAMILY=${MODEL_FAMILY//-/_}
MODEL_SIZE=${MODEL_SIZE:-base}
MODEL_NAME=${MODEL_NAME:-}
CONFIG_PATH=${CONFIG_PATH:-experiments/configs/dpo_${MODEL_FAMILY}_${MODEL_SIZE}.yaml}
TRAIN_FILE=${TRAIN_FILE:-data/dpo_train.jsonl}
DEV_FILE=${DEV_FILE:-data/dpo_dev.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/models/dpo}
LOG_DIR=${LOG_DIR:-outputs/logs}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [[ -z "$MODEL_NAME" ]]; then
  case "$MODEL_FAMILY" in
    it5)
      MODEL_NAME="gsarti/it5-${MODEL_SIZE}" ;;
    mt5)
      MODEL_NAME="google/mt5-${MODEL_SIZE}" ;;
    t5|flan_t5)
      MODEL_NAME="google/flan-t5-${MODEL_SIZE}" ;;
    bart)
      MODEL_NAME="facebook/bart-${MODEL_SIZE}" ;;
    *)
      if [[ -z "$MODEL_NAME" ]]; then
        echo "Unknown model family and no MODEL_NAME provided" >&2
        exit 1
      fi
      ;;  # Allow override via MODEL_NAME
  esac
fi

if [[ ! -f "$DEV_FILE" ]]; then
  echo "WARNING: Dev file $DEV_FILE not found; seq2seq metrics will be skipped" >&2
fi

# Build SFT model path if not provided (assume SFT was run first)
if [[ -z "$SFT_MODEL_PATH" ]]; then
  SAFE_MODEL_NAME="${MODEL_NAME//\//_}"
  SFT_MODEL_PATH="outputs/models/sft/${SAFE_MODEL_NAME}_${MODEL_SIZE}_base_baseline_lora"
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

WANDB_ENTITY_ARG=()
if [[ -n "$WANDB_ENTITY" ]]; then
  WANDB_ENTITY_ARG=(--wandb_entity "$WANDB_ENTITY")
fi
WANDB_RUN_NAME_ARG=()
if [[ -n "$WANDB_RUN_NAME" ]]; then
  WANDB_RUN_NAME_ARG=(--wandb_run_name "$WANDB_RUN_NAME")
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
  "${WANDB_ENTITY_ARG[@]}" \
  "${WANDB_RUN_NAME_ARG[@]}" \
  $SFT_ARG \
  ${EXTRA_ARGS}