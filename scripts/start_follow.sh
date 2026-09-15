#!/usr/bin/env bash
set -euo pipefail

# Start the existing person-follow framework in its safe default state. The
# controller/arbiter are namespaced per robot and enabled_on_start=false, so
# merely starting this script cannot command a moving base. The orbit
# coordinator enables view-only follow whenever both robots are stationary.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${HAR_CONTAINER_NAME:-har-rosbot-deployment}"
ROBOTS="${HAR_FOLLOW_ROBOTS:-rosbot_1,rosbot_2}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start_follow.sh [--dry-run]

Starts person_follow_demo controller + cmd_vel_arbiter for both ROSbots.
The controller starts disabled; orbit_coordinator enables it whenever neither
robot is moving. Nav2 keeps priority through cmd_vel_arbiter.
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# A dry-run is deliberately host-local: it must not recreate or restart the
# ML container merely to print the safe follow configuration.
if [[ "$DRY_RUN" == "1" ]]; then
  echo "container=$CONTAINER_NAME robots=$ROBOTS"
  echo "enabled_on_start=false"
  exit 0
fi

if [[ ! -f /.dockerenv && "${HAR_START_IN_CONTAINER:-0}" != "1" ]]; then
  command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 2; }
  cd "$ROOT"
  # Match the GUI launcher/complete-stack compose environment so starting
  # follow nodes does not recreate the container and kill VPOCLIP.
  HAR_DISPLAY=1 docker compose up -d --no-build
  docker_exec_flags=(-i)
  if [[ -t 0 && -t 1 && "${HAR_NONINTERACTIVE:-0}" != "1" ]]; then
    docker_exec_flags=(-it)
  fi
  exec docker exec "${docker_exec_flags[@]}" \
    -e HAR_START_IN_CONTAINER=1 \
    -e HAR_FOLLOW_ROBOTS="$ROBOTS" \
    -e HAR_FOLLOW_DRY_RUN="$DRY_RUN" \
    "$CONTAINER_NAME" bash -lc \
    'cd /workspace/CLIPGCN && exec bash scripts/start_follow.sh'
fi

source "$ROOT/scripts/runtime_env.sh"
mkdir -p "$ROOT/logs"

IFS=',' read -r -a namespaces <<<"$ROBOTS"
started=0
for raw_namespace in "${namespaces[@]}"; do
  namespace="${raw_namespace//[[:space:]]/}"
  [[ -n "$namespace" ]] || continue
  node_name="/$namespace/person_follow_controller"
  if ros2 node list 2>/dev/null | grep -Fxq "$node_name"; then
    echo "follow already running: $namespace"
    continue
  fi
  log_file="$ROOT/logs/follow_${namespace}.log"
  nohup ros2 launch person_follow_demo person_follow_demo.launch.py \
    "namespace:=$namespace" \
    >"$log_file" 2>&1 </dev/null &
  echo "follow started disabled: $namespace (log: $log_file)"
  started=$((started + 1))
done

(( started > 0 )) || echo "all requested follow nodes were already running"
echo "Follow framework starts disabled; orbit_coordinator enables view-follow in stationary states and disables it during Nav2 movement."
