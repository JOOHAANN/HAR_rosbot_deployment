#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
mkdir -p "$ROOT/logs"
: >"$ROOT/logs/active_view.log"

ROBOT_NAMESPACES="${HAR_VPOCLIP_ROBOT_NAMESPACES:-${HAR_ROBOT_NAMESPACE:-rosbot_1}}"
IFS=',' read -r -a robot_namespaces <<<"$ROBOT_NAMESPACES"
recognizer_logs=()

children=()
wait_for_recognizer_ready() {
  local pid="$1"
  local log_file="$2"
  local namespace="$3"
  local timeout_sec="${HAR_VPOCLIP_START_TIMEOUT_SEC:-120}"
  local deadline=$((SECONDS + timeout_sec))

  while (( SECONDS < deadline )); do
    if grep -q "Realtime CLIPGCN ROS 2 camera inference" "$log_file" 2>/dev/null; then
      echo "  VPOCLIP ready: $namespace"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "  VPOCLIP exited during startup: $namespace (see $log_file)" >&2
      return 1
    fi
    sleep 1
  done

  echo "  VPOCLIP startup timeout: $namespace (see $log_file)" >&2
  return 1
}

cleanup() {
  trap - EXIT INT TERM
  for pid in "${children[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${children[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

if [[ "${HAR_ENABLE_VIEW_POLICY:-1}" == "1" ]]; then
  "$ROOT/scripts/run_active_view_ros.sh" >>"$ROOT/logs/active_view.log" 2>&1 &
  children+=("$!")
fi

for raw_namespace in "${robot_namespaces[@]}"; do
  namespace="${raw_namespace//[[:space:]]/}"
  [[ -n "$namespace" ]] || continue
  log_file="$ROOT/logs/recognizer_${namespace}.log"
  : >"$log_file"
  recognizer_logs+=("$log_file")
  HAR_ROBOT_NAMESPACE="$namespace" \
  HAR_RECOGNIZER_NODE_NAME="${HAR_RECOGNIZER_NODE_NAME:-har_vpoclip_recognizer}_${namespace}" \
    "$ROOT/scripts/run_rosbot_har.sh" >>"$log_file" 2>&1 &
  recognizer_pid="$!"
  children+=("$recognizer_pid")
  # A second CUDA RTMPose/YOLO process can briefly exceed the 6 GB GPU during
  # model initialization. Serialize startup after the first node has built all
  # of its sessions, while keeping both nodes running afterward.
  wait_for_recognizer_ready "$recognizer_pid" "$log_file" "$namespace" || true
done

if [[ "${#recognizer_logs[@]}" == "0" ]]; then
  echo "HAR_VPOCLIP_ROBOT_NAMESPACES did not contain a robot namespace" >&2
  exit 2
fi

echo "HAR ROSbot stack started in $ROOT"
for log_file in "${recognizer_logs[@]}"; do
  echo "  recognizer log: $log_file"
done
if [[ "${HAR_ENABLE_VIEW_POLICY:-1}" == "1" ]]; then
  echo "  policy log:     $ROOT/logs/active_view.log"
  echo "  policy topic:   ${HAR_POLICY_OUTPUT_TOPIC:-/har/selected_view_pair}"
fi

while :; do
  alive=0
  for pid in "${children[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
    fi
  done
  if [[ "$alive" == "0" ]]; then
    break
  fi
  sleep 2
done

echo "HAR ROSbot stack stopped; see logs under $ROOT/logs" >&2
