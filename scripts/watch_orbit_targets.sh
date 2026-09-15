#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${HAR_CONTAINER_NAME:-har-rosbot-deployment}"

command -v docker >/dev/null 2>&1 || {
  echo "docker is required" >&2
  exit 2
}

running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)"
[[ "$running" == "true" ]] || {
  echo "container is not running: $CONTAINER_NAME" >&2
  echo "start it with ./scripts/start_orbit_system.sh --dry-run --continuous" >&2
  exit 2
}

docker_exec_flags=(-i)
if [[ -t 0 && -t 1 && "${HAR_NONINTERACTIVE:-0}" != "1" ]]; then
  docker_exec_flags=(-it)
fi

exec docker exec "${docker_exec_flags[@]}" "$CONTAINER_NAME" bash -lc \
  'source /workspace/CLIPGCN/scripts/runtime_env.sh &&
   exec python3 /workspace/CLIPGCN/scripts/watch_orbit_targets.py "$@"' \
  -- "$@"
