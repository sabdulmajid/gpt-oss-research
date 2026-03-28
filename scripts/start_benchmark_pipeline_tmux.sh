#!/usr/bin/env bash
set -euo pipefail

ROOT="/pub7/neel2/gpt-oss-research"
SESSION="benchmark-pipeline"
LOG_PATH="$ROOT/artifacts/logs/benchmark_pipeline_v1.log"
PIPELINE="$ROOT/scripts/run_benchmark_pipeline.sh"

restart=0
if [[ "${1:-}" == "--restart" ]]; then
  restart=1
fi

mkdir -p "$ROOT/artifacts/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$restart" -eq 0 ]]; then
    echo "benchmark pipeline session already exists: $SESSION"
    exit 0
  fi
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" "bash $PIPELINE > $LOG_PATH 2>&1"
echo "started benchmark pipeline in tmux session '$SESSION'"
echo "log: $LOG_PATH"
