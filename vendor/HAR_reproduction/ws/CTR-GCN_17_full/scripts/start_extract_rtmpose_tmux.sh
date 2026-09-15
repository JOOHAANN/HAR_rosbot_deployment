#!/usr/bin/env bash
set -euo pipefail

session=etri-rtmpose
runner=/home/youhan/ws/CTR-GCN_17_full/scripts/run_extract_rtmpose_etri.sh

if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)
if (( gpu_processes > 0 )); then
  echo "GPU already has $gpu_processes compute process(es); refusing to start." >&2
  exit 1
fi

tmux new-session -d -s "$session" "$runner"
echo "Started tmux session: $session"
echo "Attach: tmux attach -t $session"
echo "Log: /home/youhan/ws/CTR-GCN_17_full/logs/rtmpose_extract.log"
