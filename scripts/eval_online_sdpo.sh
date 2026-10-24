#!/usr/bin/env bash
set -euo pipefail

# Run online SDPO evaluation: interleaved training + evaluation loop.
#
# Usage:
#   uv run bash ./scripts/eval_online_sdpo.sh [--dry-run]
#
# Common overrides:
#   MODEL, STYLE, TRAIN_N... 等參數都請在 .env 檔案設定
#   若沒有設定，才會使用下面的 default 值，盡量不要在執行時才在命令列設環境變數

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift # 把參數往左推一格，第一個參數捨棄(也就是 --dry-run)
  echo "Dry run mode enabled. Commands will be printed but not executed."
fi

run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '%q ' "$@" # 印出所有參數
    printf '\n'
  else
    "$@" # 直接執行
  fi
}

# 取得腳本所在目錄的上一層，也就是 scripts/ 上一層的 sdpo_feedback/
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load parameters from .env
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

RUN_ID="${RUN_ID:-eval-online-sdpo-$(date +%Y%m%d-%H%M%S)}"
CHECKPT_DIR="/shareData/shihyu/outputs/$RUN_ID"
EVAL_DIR="$REPO_ROOT/eval/$RUN_ID"
mkdir -p "$EVAL_DIR"

# HF_HOME 寫在 ~/.bashrc 裡的環境變數
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Summary
echo "=== EVAL ONLINE SDPO ==="
echo "GPUs:       $(nvidia-smi -L 2>/dev/null | wc -l || echo 'N/A')"
echo "Date:       $(date)"
echo "Model:      $MODEL"
echo "User model: $USER_MODEL"
echo "User API:   $USER_API_PROVIDER"
echo "TRAIN_N:    $TRAIN_N"
echo "EVAL_N:     $EVAL_N"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LR:         $LR"
echo "Loss mode:  $LOSS_MODE"
echo "DATA_DIR:   $DATA_DIR ($LANGUAGE)"
echo "STYLE_MODE: ${STYLE_MODE:-false}"
echo "CHECKPT_DIR: $CHECKPT_DIR"
echo "EVAL_DIR: $EVAL_DIR"
echo ""

# Run
cd "$REPO_ROOT"

python_cmd=(
  python -m online_sdpo.eval_online_sdpo
  --model "$MODEL"
  --user_model "$USER_MODEL"
  --user_api_provider "$USER_API_PROVIDER"
  --judge_model "$USER_MODEL"

  --lr "$LR"
  --loss_mode "$LOSS_MODE"
  --train_n "$TRAIN_N"
  --eval_n "$EVAL_N"
  --eval_every "$EVAL_EVERY"
  --batch_size "$BATCH_SIZE"
  --seed "$SEED"
  --use_lora

  --max_input_tokens "$MAX_INPUT_TOKENS"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"

  --user_max_input_tokens "$USER_MAX_INPUT_TOKENS"
  --user_max_new_tokens "$USER_MAX_NEW_TOKENS"

  --train_jsonl "$DATA_DIR/train.jsonl"
  --val_jsonl "$DATA_DIR/validation.jsonl"
  --language "$LANGUAGE"

  --model_gpu "$MODEL_GPU"
  --user_gpu "$USER_GPU"
  --vllm_gpu "$VLLM_GPU"
  --vllm_gpu_mem "$VLLM_GPU_MEM"

  --checkpt_dir "$CHECKPT_DIR"
  --eval_dir "$EVAL_DIR"
  --debug_first_example
)

if [[ "$MODEL_VLLM" == true ]]; then
  python_cmd+=(--model_vllm)
fi

if [[ "$USER_VLLM" == true ]]; then
  python_cmd+=(--user_vllm)
fi

if [[ "${STYLE_MODE:-false}" == true ]]; then
  python_cmd+=(--style_mode)
fi

if [[ "$SAVE_CHECKPOINTS" == "true" ]]; then
  python_cmd+=(--save_checkpoints)
  mkdir -p "$CHECKPT_DIR"
fi

python_cmd+=("$@")

run "${python_cmd[@]}"
