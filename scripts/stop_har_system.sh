#!/usr/bin/env bash
set -euo pipefail

# Stop the complete HAR deployment without deleting containers or volumes.
# Nav2 and localization live in jazzy-rosbot; all HAR/VPOCLIP/follow nodes
# live in the compose-managed har-rosbot-deployment container.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HAR_CONTAINER="${HAR_CONTAINER_NAME:-har-rosbot-deployment}"
NAV_CONTAINER="${HAR_NAV_CONTAINER_NAME:-jazzy-rosbot}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/stop_har_system.sh [--dry-run]

Stops:
  - HAR/VPOCLIP recognizers, fusion, active-view selector and follow nodes
  - both robots' localization and Nav2 processes
  - the multi-robot RViz process launched by this deployment

It does not remove containers, images or volumes. The low-level robot driver
inside jazzy-rosbot is left running.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null 2>&1 || {
  echo "docker is required" >&2
  exit 2
}

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'would stop HAR container=%s and Nav2 container=%s\n' \
    "$HAR_CONTAINER" "$NAV_CONTAINER"
  exit 0
fi

cd "$ROOT"

nav_running="$(docker inspect -f '{{.State.Running}}' "$NAV_CONTAINER" 2>/dev/null || true)"
if [[ "$nav_running" == "true" ]]; then
  echo "[1/2] Stopping localization/Nav2 and deployment RViz in $NAV_CONTAINER."
  docker exec -i "$NAV_CONTAINER" bash -lc '
set +u
source /opt/ros/jazzy/setup.bash
set -u
cd /root/rosbot2-jazzy-image/host/offboard
./rosbot-offboard stop || true

# RViz is not tracked by rosbot-offboard pids, so close only the profile used
# by this deployment. The bracket expression avoids matching this shell.
for pid in $(pgrep -f "[r]viz2.*rosbot2_multi_robot.rviz" || true); do
  kill -TERM "$pid" 2>/dev/null || true
done
'
else
  echo "[1/2] Navigation container $NAV_CONTAINER is not running."
fi

har_running="$(docker inspect -f '{{.State.Running}}' "$HAR_CONTAINER" 2>/dev/null || true)"
if [[ "$har_running" == "true" ]]; then
  echo "[2/2] Stopping HAR/VPOCLIP/follow container $HAR_CONTAINER."
  docker compose stop clipgcn >/dev/null
else
  echo "[2/2] HAR container $HAR_CONTAINER is not running."
fi

echo "All deployment processes stopped. Containers and images were preserved."
