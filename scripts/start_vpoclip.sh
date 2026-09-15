#!/usr/bin/env bash
set -euo pipefail

# User-facing entry point for the two VPOCLIP recognition windows. It can be
# run on the host or from inside har-rosbot-deployment.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${HAR_CONTAINER_NAME:-har-rosbot-deployment}"
ROBOT_NAMESPACES="${HAR_VPOCLIP_ROBOT_NAMESPACES:-rosbot_1,rosbot_2}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start_vpoclip.sh

Starts one VPOCLIP recognition window per namespace in
HAR_VPOCLIP_ROBOT_NAMESPACES (default: rosbot_1,rosbot_2).
Run it from a desktop terminal so OpenCV can connect to DISPLAY.
Press Ctrl-C in this terminal to stop both recognition processes; press q or
Esc in a window to stop the VPOCLIP stack.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f /.dockerenv && "${HAR_START_IN_CONTAINER:-0}" != "1" ]]; then
  command -v docker >/dev/null 2>&1 || {
    echo "docker is required on the host" >&2
    exit 2
  }
  export DISPLAY="${DISPLAY:-:0}"
  if ! command -v xhost >/dev/null 2>&1 || ! xhost +si:localuser:root >/dev/null; then
    echo "X11 access failed; run 'xhost +si:localuser:root' in the desktop terminal" >&2
    exit 2
  fi
  cd "$ROOT"
  HAR_DISPLAY=1 DISPLAY="$DISPLAY" docker compose up -d --no-build
  docker_exec_flags=(-i)
  if [[ -t 0 && -t 1 && "${HAR_NONINTERACTIVE:-0}" != "1" ]]; then
    docker_exec_flags=(-it)
  fi
  exec docker exec "${docker_exec_flags[@]}" \
    -e HAR_START_IN_CONTAINER=1 \
    -e HAR_DISPLAY=1 \
    -e DISPLAY="$DISPLAY" \
    -e HAR_VPOCLIP_ROBOT_NAMESPACES="$ROBOT_NAMESPACES" \
    -e HAR_VPOCLIP_STABILIZE_SECONDS="${HAR_VPOCLIP_STABILIZE_SECONDS:-8}" \
    -e HAR_VPOCLIP_RTMPOSE_DEVICE="${HAR_VPOCLIP_RTMPOSE_DEVICE-}" \
    -e HAR_VPOCLIP_CONFIG="${HAR_VPOCLIP_CONFIG-}" \
    "$CONTAINER_NAME" bash -lc \
    'cd /workspace/CLIPGCN && exec bash scripts/start_vpoclip.sh'
fi

source "$ROOT/scripts/runtime_env.sh"
export HAR_DISPLAY=1
export DISPLAY="${DISPLAY:-:0}"
# The default RTMPose device is stored in vpoclip_config.csv. Set
# HAR_VPOCLIP_RTMPOSE_DEVICE to override it for a one-off experiment.
export HAR_RECOGNIZER_RTMPOSE_DEVICE="${HAR_VPOCLIP_RTMPOSE_DEVICE-}"

mkdir -p "$ROOT/logs"

find_pids() {
  local needle="$1"
  ps -eo pid=,args= | awk -v needle="$needle" 'index($0, needle) {print $1}'
}

stop_pids() {
  local pids="$1"
  local pid
  for pid in $pids; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ "$pid" == "$$" ]] && continue
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
}

# Remove an older combined-stack supervisor and its VPOCLIP children. This
# leaves the separate Nav2 container untouched.
old_stack_pids="$(find_pids "run_rosbot_stack.sh" || true)"
stop_pids "$old_stack_pids"
old_recognizer_pids="$(find_pids "$ROOT/ros_realtime.py" || true)"
stop_pids "$old_recognizer_pids"

children=()
cleanup() {
  trap - EXIT INT TERM
  local pid
  for pid in "${children[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${children[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_for_ready() {
  local pid="$1"
  local log_file="$2"
  local namespace="$3"
  local deadline=$((SECONDS + ${HAR_VPOCLIP_START_TIMEOUT_SEC:-120}))
  while (( SECONDS < deadline )); do
    if grep -q "Realtime CLIPGCN ROS 2 camera inference" "$log_file" 2>/dev/null; then
      echo "VPOCLIP ready: $namespace"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "VPOCLIP exited during startup: $namespace (see $log_file)" >&2
      return 1
    fi
    sleep 1
  done
  echo "VPOCLIP startup timeout: $namespace (see $log_file)" >&2
  return 1
}

stabilize_vpoclip() {
  local pid="$1"
  local namespace="$2"
  local seconds="${HAR_VPOCLIP_STABILIZE_SECONDS:-8}"

  [[ "$seconds" =~ ^[0-9]+$ ]] || seconds=8
  if (( seconds == 0 )); then
    return 0
  fi

  echo "VPOCLIP stable-check: $namespace (${seconds}s before next recognizer)"
  sleep "$seconds"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "VPOCLIP exited after startup: $namespace" >&2
    return 1
  fi
}

IFS=',' read -r -a namespaces <<<"$ROBOT_NAMESPACES"
started=0
for raw_namespace in "${namespaces[@]}"; do
  namespace="${raw_namespace//[[:space:]]/}"
  [[ -n "$namespace" ]] || continue
  log_file="$ROOT/logs/vpoclip_${namespace}.log"
  : >"$log_file"
  HAR_ROBOT_NAMESPACE="$namespace" \
  HAR_RECOGNIZER_NODE_NAME="har_vpoclip_recognizer_${namespace}" \
    "$ROOT/scripts/run_rosbot_har.sh" >>"$log_file" 2>&1 &
  pid="$!"
  children+=("$pid")
  started=$((started + 1))
  # Initialize the two recognizers serially and let the first one settle. This
  # is important on the local 6 GB GPU: starting the second CUDA session while
  # the first one is still warming up can trigger a PyTorch clock assertion.
  wait_for_ready "$pid" "$log_file" "$namespace" || exit 1
  stabilize_vpoclip "$pid" "$namespace" || exit 1
done

(( started > 0 )) || {
  echo "HAR_VPOCLIP_ROBOT_NAMESPACES contains no robot namespace" >&2
  exit 2
}

echo "VPOCLIP GUI started in $ROOT"
echo "Windows: CLIPGCN rosbot_1 / CLIPGCN rosbot_2"
echo "Logs:    $ROOT/logs/vpoclip_<namespace>.log"

while :; do
  alive=0
  for pid in "${children[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
    fi
  done
  (( alive == 1 )) || break
  sleep 2
done
