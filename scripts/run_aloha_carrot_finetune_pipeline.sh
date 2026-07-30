#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_DIR="${SOURCE_DIR:-data/aloha_carrot_easy}"
DATA_DIR="${DATA_DIR:-outputs/derived/aloha_carrot_easy_rlds}"
DATASET_NAME="${DATASET_NAME:-aloha_carrot_easy_rlds}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_VERSION="${DATASET_VERSION:-1.0.0}"
DATASET_BUILDER_DIR="${DATA_DIR}/${DATASET_NAME}/${DATASET_VERSION}"
DATASET_STATS="${DATASET_STATS:-${DATASET_BUILDER_DIR}/octo_dataset_statistics.json}"

PRETRAINED_PATH="${PRETRAINED_PATH:-hf://rail-berkeley/octo-small-1.5}"
SAVE_DIR="${SAVE_DIR:-checkpoints/octo/aloha_carrot_finetune_seed42}"
METRICS_DIR="${METRICS_DIR:-${SAVE_DIR}/metrics}"
EVAL_DIR="${EVAL_DIR:-outputs/eval/aloha_carrot_finetune_seed42}"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
NUM_STEPS="${NUM_STEPS:-60000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
PLOT_INTERVAL="${PLOT_INTERVAL:-250}"
WINDOW_SIZE="${WINDOW_SIZE:-1}"
ACTION_HORIZON="${ACTION_HORIZON:-20}"
FREEZE_POLICY="${FREEZE_POLICY:-stage_b}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
NEW_MODULE_LR="${NEW_MODULE_LR:-0.0001}"
BACKBONE_LR="${BACKBONE_LR:-0.00001}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
DATASET_SUBSAMPLE_LENGTH="${DATASET_SUBSAMPLE_LENGTH:-32}"
DATASET_SHUFFLE_BUFFER="${DATASET_SHUFFLE_BUFFER:-64}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-200}"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_STEP="${RESUME_STEP:-}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export ALOHA_CARROT_SOURCE_DIR="${SOURCE_DIR}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [[ ! -f "${DATASET_BUILDER_DIR}/dataset_info.json" || ! -f "${DATASET_STATS}" ]]; then
  echo "[pipeline] RLDS not found, converting ${SOURCE_DIR} -> ${DATA_DIR}"
  "${PYTHON_BIN}" scripts/convert_aloha_carrot_to_rlds.py \
    --source-dir "${SOURCE_DIR}" \
    --output-dir "${DATA_DIR}"
else
  echo "[pipeline] RLDS exists: ${DATASET_BUILDER_DIR}"
fi

mkdir -p "${SAVE_DIR}" "${METRICS_DIR}" "${EVAL_DIR}"

echo "[pipeline] finetune: ${NUM_STEPS} steps, checkpoint every ${SAVE_INTERVAL} steps"
FINETUNE_CMD=(
  "${PYTHON_BIN}" examples/02_pt_finetune_new_observation_action.py
  --pretrained_path="${PRETRAINED_PATH}"
  --data_dir="${DATA_DIR}"
  --dataset_name="${DATASET_NAME}"
  --dataset_split="${DATASET_SPLIT}"
  --dataset_statistics="${DATASET_STATS}"
  --primary_image_key=top
  --wrist_image_key=wrist
  --proprio_key=state
  --language_key=language_instruction
  --window_size="${WINDOW_SIZE}"
  --action_horizon="${ACTION_HORIZON}"
  --freeze_policy="${FREEZE_POLICY}"
  --learning_rate="${LEARNING_RATE}"
  --new_module_learning_rate="${NEW_MODULE_LR}"
  --backbone_learning_rate="${BACKBONE_LR}"
  --warmup_steps="${WARMUP_STEPS}"
  --weight_decay="${WEIGHT_DECAY}"
  --max_grad_norm="${MAX_GRAD_NORM}"
  --batch_size="${BATCH_SIZE}"
  --gradient_accumulation_steps="${GRAD_ACCUM}"
  --num_steps="${NUM_STEPS}"
  --save_interval="${SAVE_INTERVAL}"
  --seed="${SEED}"
  --device="${DEVICE}"
  --save_dir="${SAVE_DIR}"
  --metrics_dir="${METRICS_DIR}"
  --dataset_shuffle_buffer="${DATASET_SHUFFLE_BUFFER}"
  --dataset_subsample_length="${DATASET_SUBSAMPLE_LENGTH}"
  --log_interval="${LOG_INTERVAL}"
  --plot_interval="${PLOT_INTERVAL}"
  --run_name="aloha_carrot_finetune_seed${SEED}"
  --wandb_mode=disabled
)
if [[ -n "${RESUME_FROM}" ]]; then
  FINETUNE_CMD+=(--resume_from="${RESUME_FROM}")
fi
if [[ -n "${RESUME_STEP}" ]]; then
  FINETUNE_CMD+=(--resume_step="${RESUME_STEP}")
fi
"${FINETUNE_CMD[@]}"

echo "[pipeline] plot train metrics"
"${PYTHON_BIN}" scripts/plot_training_metrics.py \
  --metrics-csv "${METRICS_DIR}/training_metrics.csv" \
  --output-dir "${METRICS_DIR}"

echo "[pipeline] offline checkpoint metrics"
"${PYTHON_BIN}" scripts/evaluate_octo_checkpoint_metrics.py \
  --checkpoint-dir "${SAVE_DIR}" \
  --data-dir "${DATA_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-split "${DATASET_SPLIT}" \
  --dataset-statistics "${DATASET_STATS}" \
  --window-size "${WINDOW_SIZE}" \
  --action-horizon "${ACTION_HORIZON}" \
  --batch-size "${BATCH_SIZE}" \
  --num-batches "${EVAL_NUM_BATCHES}" \
  --device "${DEVICE}" \
  --output-dir "${EVAL_DIR}"

echo "[pipeline] done"
echo "[pipeline] checkpoints: ${SAVE_DIR}"
echo "[pipeline] training metrics: ${METRICS_DIR}/training_metrics.csv"
echo "[pipeline] eval metrics: ${EVAL_DIR}/checkpoint_metrics.csv"
