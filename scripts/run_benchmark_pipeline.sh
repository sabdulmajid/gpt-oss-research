#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_20B_PATH="${GPT_OSS_20B_PATH:-openai/gpt-oss-20b}"

TASKS_ROOT="$ROOT/eval/ml_research_eval/tasks"
LOG_DIR="$ROOT/artifacts/logs"
EVAL_DIR="$ROOT/artifacts/eval"
REPORT_DIR="$ROOT/reports/experiments"

BASE_REPORT="$EVAL_DIR/internal_eval_base_51_v1.json"
SFT_REPORT="$EVAL_DIR/internal_eval_sft_benchmark_v1.json"
GRPO_REPORT="$EVAL_DIR/internal_eval_grpo_benchmark_v1.json"
SUMMARY_REPORT="$REPORT_DIR/benchmark_pipeline_v1.md"

SFT_OUTPUT_DIR="$ROOT/artifacts/checkpoints/sft/gpt_oss_20b_sft_benchmark_v1"
GRPO_OUTPUT_DIR="$ROOT/artifacts/checkpoints/grpo/gpt_oss_20b_grpo_benchmark_v1"
SFT_TRAIN_REPORT="$SFT_OUTPUT_DIR/train_report.json"
GRPO_TRAIN_REPORT="$GRPO_OUTPUT_DIR/train_report.json"

mkdir -p "$LOG_DIR" "$EVAL_DIR" "$REPORT_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

cuda_ready() {
  python scripts/gpu_diagnostics.py --ready-check >/dev/null 2>&1
}

wait_for_cuda() {
  until cuda_ready; do
    log "$(python scripts/gpu_diagnostics.py --summary-line)"
    log "CUDA not ready; waiting 60s"
    sleep 60
  done
  log "$(python scripts/gpu_diagnostics.py --summary-line)"
  log "CUDA is ready"
}

run_if_missing_file() {
  local target="$1"
  shift
  if [[ -f "$target" ]]; then
    log "Skipping existing artifact: $target"
    return 0
  fi
  "$@"
}

wait_for_cuda

run_if_missing_file \
  "$BASE_REPORT" \
  python scripts/run_model_eval.py \
    --model-path "$MODEL_20B_PATH" \
    --tasks-root "$TASKS_ROOT" \
    --output "$BASE_REPORT" \
    --max-new-tokens 768

if [[ ! -f "$SFT_OUTPUT_DIR/adapter_model.safetensors" ]]; then
  python scripts/run_sft.py configs/training/sft_gpt_oss_20b_lora_benchmark_v1.yaml
else
  log "Skipping existing SFT adapter: $SFT_OUTPUT_DIR/adapter_model.safetensors"
fi

run_if_missing_file \
  "$SFT_REPORT" \
  python scripts/run_model_eval.py \
    --model-path "$MODEL_20B_PATH" \
    --adapter-path "$SFT_OUTPUT_DIR" \
    --tasks-root "$TASKS_ROOT" \
    --output "$SFT_REPORT" \
    --max-new-tokens 768

if [[ ! -f "$GRPO_OUTPUT_DIR/adapter_model.safetensors" ]]; then
  python scripts/run_grpo.py configs/training/grpo_gpt_oss_20b_lora_benchmark_v1.yaml
else
  log "Skipping existing GRPO adapter: $GRPO_OUTPUT_DIR/adapter_model.safetensors"
fi

run_if_missing_file \
  "$GRPO_REPORT" \
  python scripts/run_model_eval.py \
    --model-path "$MODEL_20B_PATH" \
    --adapter-path "$GRPO_OUTPUT_DIR" \
    --tasks-root "$TASKS_ROOT" \
    --output "$GRPO_REPORT" \
    --max-new-tokens 768

python scripts/summarize_benchmark_pipeline.py \
  --base-report "$BASE_REPORT" \
  --sft-report "$SFT_REPORT" \
  --grpo-report "$GRPO_REPORT" \
  --sft-train-report "$SFT_TRAIN_REPORT" \
  --grpo-train-report "$GRPO_TRAIN_REPORT" \
  --output "$SUMMARY_REPORT"

log "Benchmark pipeline complete"
