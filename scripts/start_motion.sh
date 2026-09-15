#!/usr/bin/env bash
set -euo pipefail

# The exported RL artifact is a four-view selector, not a Nav2/base-motion
# policy. This entry point exposes the three protocols from har_export:
# exported DDQN (rl), random_pair (random), and cyclic_pair (cyclic).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${HAR_CONTAINER_NAME:-har-rosbot-deployment}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start_motion.sh [rl|random|cyclic]

If no mode is supplied, selection_mode is read from
config/orbit_system_config.csv. A mode on the command line overrides the CSV.

Modes:
  rl       Exported full_depth19 two-stage DDQN viewpoint selector
  random   har_export random_pair baseline
  cyclic   har_export cyclic_pair baseline (0-2, 1-3, ...)

The selected pair is published on /har/selected_view_pair. The archive does
not contain a map-specific base-motion/Nav2 policy, so this command does not
publish cmd_vel or send Nav2 goals.

Environment:
  HAR_VIEW_RANDOM_SEED             Random baseline seed (default: 20260915)
  HAR_VIEW_BASELINE_HOLD_SECONDS   Hold time for random/cyclic pair (default: 3)
EOF
}

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  if [[ -n "${HAR_VIEW_SELECTION_MODE:-}" ]]; then
    MODE="$HAR_VIEW_SELECTION_MODE"
  else
    ORBIT_CONFIG="${HAR_ORBIT_CONFIG:-$ROOT/config/orbit_system_config.csv}"
    if [[ "$ORBIT_CONFIG" != /* ]]; then
      ORBIT_CONFIG="$ROOT/$ORBIT_CONFIG"
    fi
    [[ -f "$ORBIT_CONFIG" ]] || { echo "orbit movement config table not found: $ORBIT_CONFIG" >&2; exit 2; }
    MODE="$(awk -F',' '
      NR == 1 { next }
      /^[[:space:]]*#/ { next }
      $1 == "selection_mode" {
        value = $2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        print value
        exit
      }
    ' "$ORBIT_CONFIG")"
    MODE="${MODE:-rl}"
  fi
fi
case "$MODE" in
  rl|random|cyclic)
    [[ $# -eq 0 ]] || shift
    ;;
  -h|--help|"")
    usage
    [[ -n "$MODE" ]] || exit 2
    exit 0
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

[[ $# -eq 0 ]] || { echo "unexpected arguments: $*" >&2; exit 2; }

if [[ ! -f /.dockerenv && "${HAR_START_IN_CONTAINER:-0}" != "1" ]]; then
  command -v docker >/dev/null 2>&1 || {
    echo "docker is required on the host" >&2
    exit 2
  }
  cd "$ROOT"
  # Keep the compose environment stable so replacing the view selector does
  # not recreate the container and terminate VPOCLIP/fusion nodes.
  HAR_DISPLAY=1 docker compose up -d --no-build
  docker_exec_flags=(-i)
  if [[ -t 0 && -t 1 && "${HAR_NONINTERACTIVE:-0}" != "1" ]]; then
    docker_exec_flags=(-it)
  fi
  exec docker exec "${docker_exec_flags[@]}" \
    -e HAR_START_IN_CONTAINER=1 \
    -e HAR_VIEW_SELECTION_MODE="$MODE" \
    -e HAR_ORBIT_CONFIG="${HAR_ORBIT_CONFIG-}" \
    -e HAR_VIEW_RANDOM_SEED="${HAR_VIEW_RANDOM_SEED-}" \
    -e HAR_VIEW_BASELINE_HOLD_SECONDS="${HAR_VIEW_BASELINE_HOLD_SECONDS-}" \
    "$CONTAINER_NAME" bash -lc \
    'cd /workspace/CLIPGCN && exec bash scripts/start_motion.sh "$HAR_VIEW_SELECTION_MODE"'
fi

source "$ROOT/scripts/runtime_env.sh"
export HAR_VIEW_SELECTION_MODE="$MODE"
if [[ -n "${HAR_VIEW_RANDOM_SEED:-}" ]]; then
  export HAR_VIEW_RANDOM_SEED
else
  unset HAR_VIEW_RANDOM_SEED
fi
if [[ -n "${HAR_VIEW_BASELINE_HOLD_SECONDS:-}" ]]; then
  export HAR_VIEW_BASELINE_HOLD_SECONDS
else
  unset HAR_VIEW_BASELINE_HOLD_SECONDS
fi

find_pids() {
  ps -eo pid=,args= | awk -v needle="$ROOT/active_view_ros.py" 'index($0, needle) {print $1}'
}

# Only one view-selection protocol may own the output topic at a time.
old_pids="$(find_pids || true)"
for pid in $old_pids; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  [[ "$pid" == "$$" ]] && continue
  kill -TERM "$pid" 2>/dev/null || true
done
sleep 2

echo "Starting viewpoint motion protocol: $MODE"
echo "Output: /har/selected_view_pair"
echo "Movement parameters: ${HAR_ORBIT_CONFIG:-$ROOT/config/orbit_system_config.csv}"
echo "Physical Nav2 dispatch: owned by orbit_coordinator; this node only selects the next pair"
exec bash "$ROOT/scripts/run_active_view_ros.sh"
