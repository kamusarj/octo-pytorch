#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/linh/anaconda3/envs/octo_pt/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/open_loop_two_models_original_data}"
SPLITS="${SPLITS:-train,validation}"
TRAJ_IDS="${TRAJ_IDS:-0}"
STEPS="${STEPS:-200}"
DEVICE="${DEVICE:-cuda:0}"
DATA_DIR="${DATA_DIR:-outputs/derived/aloha_carrot_easy_rlds}"
DATASET_NAME="${DATASET_NAME:-aloha_carrot_easy_rlds}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"

export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false

run_open_loop() {
  local checkpoint_dir="$1"
  local checkpoint_step="$2"
  local output_name="$3"
  echo "[open-loop] checkpoint=${checkpoint_dir}/${checkpoint_step}"
  "${PYTHON_BIN}" scripts/evaluate_octo_open_loop.py \
    --checkpoint-dir "${checkpoint_dir}" \
    --checkpoint-step "${checkpoint_step}" \
    --splits "${SPLITS}" \
    --traj-ids "${TRAJ_IDS}" \
    --steps "${STEPS}" \
    --execution-horizon "${EXECUTION_HORIZON}" \
    --device "${DEVICE}" \
    --data-dir "${DATA_DIR}" \
    --dataset-name "${DATASET_NAME}" \
    --output-dir "${OUTPUT_ROOT}/${output_name}"
}

# Run sequentially so only one Octo model occupies host/GPU memory at a time.
run_open_loop \
  checkpoints/octo/aloha_carrot_w2_h8_stageb_seed42 \
  20000 \
  aloha_carrot_w2_h8_stageb_step20000

run_open_loop \
  checkpoints/octo/one_episode_ep0_jitter1cm_command_v4_adapt \
  4500 \
  one_episode_ep0_real_ep0_step4500

echo "[open-loop] results=${OUTPUT_ROOT}"
