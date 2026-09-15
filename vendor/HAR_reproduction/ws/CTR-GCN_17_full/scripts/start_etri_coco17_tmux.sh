#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/home/youhan/ws/CTR-GCN_17_full"
readonly SESSION="etri-ctrgcn17"
readonly LOG="${ROOT}/logs/etri_coco17_train.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi

mkdir -p -- "${ROOT}/logs"
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && ./scripts/train_etri_coco17_guarded.sh 2>&1 | tee -a '$LOG'"
echo "Started tmux session: $SESSION"
