#!/usr/bin/env bash
set -euo pipefail

# Bring up the complete coordinated stack. The default is dry-run: Nav2,
# mapping/localization, RViz, follow nodes, recognizers, fusion and the active
# view selector may run, but orbit_coordinator never sends a navigation goal.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${HAR_CONTAINER_NAME:-har-rosbot-deployment}"
NAV_CONTAINER="${HAR_NAV_CONTAINER_NAME:-jazzy-rosbot}"
ORBIT_CONFIG="${HAR_ORBIT_CONFIG:-$ROOT/config/orbit_system_config.csv}"
if [[ "$ORBIT_CONFIG" != /* ]]; then
  ORBIT_CONFIG="$ROOT/$ORBIT_CONFIG"
fi
if [[ ! -f "$ORBIT_CONFIG" ]]; then
  echo "orbit movement config table not found: $ORBIT_CONFIG" >&2
  exit 2
fi
if [[ "$ORBIT_CONFIG" == "$ROOT/"* ]]; then
  ORBIT_CONFIG_CONTAINER="${ORBIT_CONFIG#$ROOT/}"
else
  echo "HAR_ORBIT_CONFIG must be inside the project so Docker can mount it: $ORBIT_CONFIG" >&2
  exit 2
fi

orbit_config_value() {
  local wanted="$1"
  local fallback="${2:-}"
  local value
  value="$(awk -F',' -v wanted="$wanted" '
    NR == 1 { next }
    /^[[:space:]]*#/ { next }
    {
      key = $1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      if (key == wanted) {
        value = $2
        sub(/\r$/, "", value)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        print value
        exit
      }
    }
  ' "$ORBIT_CONFIG")"
  if [[ -n "$value" ]]; then printf '%s' "$value"; else printf '%s' "$fallback"; fi
}

orbit_bool() {
  local key="$1"
  local value="${2:-}"
  case "${value,,}" in
    1|true|yes|y|on|enable|enabled) printf '1' ;;
    0|false|no|n|off|disable|disabled) printf '0' ;;
    *) echo "Invalid boolean for $key in $ORBIT_CONFIG: $value" >&2; exit 2 ;;
  esac
}

ROBOT_NAMESPACES="$(orbit_config_value robot_namespaces 'rosbot_1;rosbot_2')"
ROBOT_NAMESPACES="${ROBOT_NAMESPACES//;/,}"
MAP_NAME="${HAR_NAV_MAP:-$(orbit_config_value map_name home)}"
INITIAL_POSE_CONFIG="${HAR_INITIAL_POSE_CONFIG:-$(orbit_config_value initial_pose_config config/robot_initial_poses.csv)}"
if [[ "$INITIAL_POSE_CONFIG" != /* ]]; then
  INITIAL_POSE_CONFIG="$ROOT/$INITIAL_POSE_CONFIG"
fi
if [[ ! -f "$INITIAL_POSE_CONFIG" ]]; then
  echo "initial pose config table not found: $INITIAL_POSE_CONFIG" >&2
  exit 2
fi
SELECTION_MODE="${HAR_VIEW_SELECTION_MODE:-$(orbit_config_value selection_mode rl)}"
if [[ "$(orbit_bool dry_run "$(orbit_config_value dry_run true)")" == "1" ]]; then
  COORD_MODE="--dry-run"
else
  COORD_MODE="--live"
fi
WITH_VPOCLIP=1
WITH_RVIZ=1
RESTART_NAV=0
CONTINUOUS=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start_orbit_system.sh [options]

Options:
  --dry-run              Print/plan ring motion only (default, safe)
  --live                 Enable serialized Nav2 orbit motion (explicit)
  --selection MODE       rl, random, or cyclic (default: rl)
  --no-vpoclip           Do not start the two GUI recognizers
  --no-rviz              Do not open the existing multi-robot RViz profile
  --restart-nav          Restart localization/Nav2 before starting
  --continuous           Keep dry-run coordinator alive after its first cycle
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) COORD_MODE="--dry-run"; shift ;;
    --live) COORD_MODE="--live"; shift ;;
    --selection)
      [[ $# -ge 2 ]] || { echo "--selection needs rl|random|cyclic" >&2; exit 2; }
      SELECTION_MODE="$2"; shift 2 ;;
    --no-vpoclip) WITH_VPOCLIP=0; shift ;;
    --no-rviz) WITH_RVIZ=0; shift ;;
    --restart-nav) RESTART_NAV=1; shift ;;
    --continuous) CONTINUOUS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SELECTION_MODE" in
  rl|random|cyclic) ;;
  *) echo "invalid --selection: $SELECTION_MODE" >&2; exit 2 ;;
esac

command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 2; }
cd "$ROOT"
mkdir -p "$ROOT/logs"

echo "[1/6] Starting existing jazzy-rosbot localization/Nav2 (no goal is sent)."
nav_args=(--robots "$ROBOT_NAMESPACES" --map "$MAP_NAME" --initial-pose-config "$INITIAL_POSE_CONFIG")
[[ "$RESTART_NAV" == "1" ]] && nav_args+=(--restart)
"$ROOT/scripts/start_nav.sh" "${nav_args[@]}"

if [[ "$WITH_RVIZ" == "1" ]]; then
  echo "[2/6] Opening multi-robot RViz."
  docker exec -d -e DISPLAY="${DISPLAY:-:0}" "$NAV_CONTAINER" bash -lc \
    'source /opt/ros/jazzy/setup.bash; cd /root/rosbot2-jazzy-image/host/offboard; exec ./rosbot-offboard rviz' \
    >/dev/null
else
  echo "[2/6] RViz disabled by option."
fi

echo "[3/6] Starting existing follow-person controller/arbiter disabled."
HAR_FOLLOW_ROBOTS="$ROBOT_NAMESPACES" "$ROOT/scripts/start_follow.sh"

echo "[4/6] Starting VPOCLIP fusion and serialized orbit coordinator."
# Keep the compose environment identical to the GUI launcher. Without this
# explicit value Docker sees HAR_DISPLAY as changed and recreates the
# container, which would terminate already-running recognizers and ROS nodes.
HAR_DISPLAY=1 docker compose up -d --no-build >/dev/null
coord_flags="$COORD_MODE"
[[ "$CONTINUOUS" == "1" ]] && coord_flags="$coord_flags --continuous"
docker exec -d -e HAR_ORBIT_CONFIG="$ORBIT_CONFIG_CONTAINER" "$CONTAINER_NAME" bash -lc \
  "source /workspace/CLIPGCN/scripts/runtime_env.sh; cd /workspace/CLIPGCN; \
   if ! ros2 node list 2>/dev/null | grep -Fxq /har_vpoclip_fusion; then \
     nohup bash scripts/run_vpoclip_fusion.sh >logs/vpoclip_fusion.log 2>&1 < /dev/null & \
   fi; \
   if ! ros2 node list 2>/dev/null | grep -Fxq /har_orbit_coordinator; then \
     nohup bash scripts/run_orbit_coordinator.sh $coord_flags >logs/orbit_coordinator.log 2>&1 < /dev/null & \
   fi" \
  >/dev/null

if [[ "$WITH_VPOCLIP" == "1" ]]; then
  echo "[5/6] Starting the two VPOCLIP GUI windows."
  command -v xhost >/dev/null 2>&1 || {
    echo "xhost is required to open the VPOCLIP GUI" >&2
    exit 2
  }
  xhost +si:localuser:root >/dev/null || {
    echo "X11 access failed; run 'xhost +si:localuser:root' in the desktop terminal" >&2
    exit 2
  }
  docker exec -d \
    -e HAR_START_IN_CONTAINER=1 \
    -e HAR_DISPLAY=1 \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e HAR_VPOCLIP_ROBOT_NAMESPACES="${HAR_VPOCLIP_ROBOT_NAMESPACES:-$ROBOT_NAMESPACES}" \
    -e HAR_VPOCLIP_STABILIZE_SECONDS="${HAR_VPOCLIP_STABILIZE_SECONDS:-8}" \
    -e HAR_VPOCLIP_RTMPOSE_DEVICE="${HAR_VPOCLIP_RTMPOSE_DEVICE-}" \
    -e HAR_VPOCLIP_CONFIG="${HAR_VPOCLIP_CONFIG-}" \
    "$CONTAINER_NAME" bash -lc \
    'cd /workspace/CLIPGCN && exec bash scripts/start_vpoclip.sh >logs/start_vpoclip.log 2>&1' \
    >/dev/null
  echo "VPOCLIP launcher is supervised inside $CONTAINER_NAME; see logs/start_vpoclip.log and vpoclip_<robot>.log"
else
  echo "[5/6] VPOCLIP GUI disabled by option."
fi

echo "[6/6] Starting ${SELECTION_MODE} active-view selector."
docker exec -d \
  -e HAR_START_IN_CONTAINER=1 \
  -e HAR_VIEW_SELECTION_MODE="$SELECTION_MODE" \
  -e HAR_ORBIT_CONFIG="$ORBIT_CONFIG_CONTAINER" \
  "$CONTAINER_NAME" bash -lc \
  'cd /workspace/CLIPGCN && exec bash scripts/start_motion.sh "$HAR_VIEW_SELECTION_MODE" >logs/active_view_launcher.log 2>&1' \
  >/dev/null
echo "Active-view selector is supervised inside $CONTAINER_NAME; see logs/active_view_launcher.log and logs/active_view.log"

echo
if [[ "$COORD_MODE" == "--dry-run" ]]; then
  echo "Orbit system started in DRY-RUN mode."
else
  echo "Orbit system started in LIVE mode."
fi
echo "Safety: no rosbot-offboard goal command was run; dry-run coordinator sends no Nav2 action."
echo "State:   /har/orbit/state"
echo "Plan:    /har/orbit/plan"
echo "Targets: /har/orbit/target_poses"
echo "Fusion:  /har/final_action"
echo "Monitor: ./scripts/watch_orbit_targets.sh"
