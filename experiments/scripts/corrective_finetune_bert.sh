#!/bin/bash
#SBATCH --job-name=corrective-bert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:V100S:1
#SBATCH --output=/italian-hyperpartisan-neutralization/slurm_logs/corrective-bert-%j.log

# =============================================================================
# Corrective fine-tuning job for the Italian BERT classifier
# Trains on the full corrective dataset and logs evaluation metrics/predictions
# =============================================================================

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${PROJECT_ROOT}"

SCRIPT_PATH="${PROJECT_ROOT}/experiments/scripts/corrective_finetune_bert.py"

module purge
module load cuda/12.1 || true
module load anaconda || true
source /miniconda3/bin/activate base

python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import torch; print(f'Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

mkdir -p outputs/logs slurm_logs

TRAIN_FILE=${TRAIN_FILE:-data/corrective_train.csv}
VAL_FILE=${VAL_FILE:-data/corrective_val.csv}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/models/bert_corrected}
BEST_DIR=${BEST_DIR:-${OUTPUT_DIR}/best}
LOG_FILE=${LOG_FILE:-${OUTPUT_DIR}/training_logs.json}
REPORT_FILE=${REPORT_FILE:-${OUTPUT_DIR}/before_after_comparison.txt}
METRICS_FILE=${METRICS_FILE:-${OUTPUT_DIR}/final_metrics.json}
PREDICTIONS_FILE=${PREDICTIONS_FILE:-${OUTPUT_DIR}/val_predictions.csv}
BASE_MODEL=${BASE_MODEL:-/italian-hyperpartisan-neutralization/XAI_HIPP/models/FT/sent-ita-xxl}
MAX_LENGTH=${MAX_LENGTH:-512}
EPOCHS=${EPOCHS:-5}
TRAIN_BS=${TRAIN_BS:-16}
EVAL_BS=${EVAL_BS:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
CLASS_WEIGHT_HYPER=${CLASS_WEIGHT_HYPER:-1.0}
CLASS_WEIGHT_NEUTRAL=${CLASS_WEIGHT_NEUTRAL:-2.0}
EXTRA_ARGS=${EXTRA_ARGS:-}

cat <<EOCONFIG
=== Corrective fine-tuning configuration ===
Train file:        ${TRAIN_FILE}
Validation file:   ${VAL_FILE}
Base model:        ${BASE_MODEL}
Output dir:        ${OUTPUT_DIR}
Best model dir:    ${BEST_DIR}
Log file:          ${LOG_FILE}
Metrics file:      ${METRICS_FILE}
Predictions file:  ${PREDICTIONS_FILE}
Epochs:            ${EPOCHS}
Train batch size:  ${TRAIN_BS}
Eval batch size:   ${EVAL_BS}
Learning rate:     ${LEARNING_RATE}
Weight decay:      ${WEIGHT_DECAY}
Warmup ratio:      ${WARMUP_RATIO}
Class weight (H):  ${CLASS_WEIGHT_HYPER}
Class weight (N):  ${CLASS_WEIGHT_NEUTRAL}
=============================================
EOCONFIG

python "${SCRIPT_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --base_model "${BASE_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --best_model_dir "${BEST_DIR}" \
  --log_file "${LOG_FILE}" \
  --report_file "${REPORT_FILE}" \
  --metrics_file "${METRICS_FILE}" \
  --predictions_file "${PREDICTIONS_FILE}" \
  --max_length "${MAX_LENGTH}" \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size "${TRAIN_BS}" \
  --per_device_eval_batch_size "${EVAL_BS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --class_weight_hyper "${CLASS_WEIGHT_HYPER}" \
  --class_weight_neutral "${CLASS_WEIGHT_NEUTRAL}" \
  ${EXTRA_ARGS}
