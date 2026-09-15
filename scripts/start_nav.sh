#!/usr/bin/env bash
set -euo pipefail

# User-facing entry point for the existing, separate jazzy-rosbot Nav2
# container. No Nav2 process is started in the HAR/VPOCLIP container.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${HAR_NAV_CONTAINER_NAME:-jazzy-rosbot}"
ROBOT_NAMESPACES="${HAR_NAV_ROBOTS:-rosbot_1,rosbot_2}"
MAP_NAME="${HAR_NAV_MAP:-home}"
INITIAL_POSE_CONFIG="${HAR_INITIAL_POSE_CONFIG:-$ROOT/config/robot_initial_poses.csv}"
RESTART=0
RVIZ=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start_nav.sh [options]

Options:
  --robots LIST   Comma-separated namespaces (default: rosbot_1,rosbot_2)
  --map NAME      Saved map name (default: home)
  --initial-pose-config FILE
                  CSV with saved map-frame initial poses (default: config/robot_initial_poses.csv)
  --restart       Stop and start localization/Nav2 for the selected robots
  --rviz          Also open the existing multi-robot RViz profile
  --dry-run       Print the commands without starting anything
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robots)
      [[ $# -ge 2 ]] || { echo "--robots needs a value" >&2; exit 2; }
      ROBOT_NAMESPACES="$2"
      shift 2
      ;;
    --map)
      [[ $# -ge 2 ]] || { echo "--map needs a value" >&2; exit 2; }
      MAP_NAME="$2"
      shift 2
      ;;
    --initial-pose-config)
      [[ $# -ge 2 ]] || { echo "--initial-pose-config needs a file" >&2; exit 2; }
      INITIAL_POSE_CONFIG="$2"
      shift 2
      ;;
    --restart) RESTART=1; shift ;;
    --rviz) RVIZ=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$INITIAL_POSE_CONFIG" != /* ]]; then
  INITIAL_POSE_CONFIG="$ROOT/$INITIAL_POSE_CONFIG"
fi
[[ -f "$INITIAL_POSE_CONFIG" ]] || {
  echo "initial pose config table not found: $INITIAL_POSE_CONFIG" >&2
  exit 2
}

# Read enabled rows on the host. The navigation container does not mount this
# project, so pass only the selected rows into docker exec. Yaw is stored in
# degrees for easy editing and converted to the radians expected by ROS.
pose_for_robot() {
  local wanted_robot="$1"
  awk -F',' -v wanted_map="$MAP_NAME" -v wanted_robot="$wanted_robot" '
    function trim(value) {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      sub(/\r$/, "", value)
      return value
    }
    function numeric(value) {
      return value ~ /^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/
    }
    NR == 1 { next }
    /^[[:space:]]*#/ || NF == 0 { next }
    {
      map_name = trim($1)
      robot = trim($2)
      enabled = tolower(trim($3))
      if (map_name == wanted_map && robot == wanted_robot &&
          enabled ~ /^(1|true|yes|y|on|enabled)$/) {
        x = trim($4)
        y = trim($5)
        yaw_deg = trim($6)
        if (!numeric(x) || !numeric(y) || !numeric(yaw_deg)) {
          printf("invalid numeric pose for %s on map %s in %s\\n", wanted_robot, wanted_map, FILENAME) > "/dev/stderr"
          invalid = 1
          exit
        }
        printf("%s|%s|%.12f", x, y, (yaw_deg + 0) * 3.141592653589793 / 180.0)
        found = 1
        exit
      }
    }
    END {
      if (invalid) exit 2
      if (!found) exit 1
    }
  ' "$INITIAL_POSE_CONFIG"
}

INITIAL_POSE_ENTRIES=""
IFS="," read -r -a pose_namespaces <<< "$ROBOT_NAMESPACES"
for raw_namespace in "${pose_namespaces[@]}"; do
  ns="${raw_namespace//[[:space:]]/}"
  [[ -n "$ns" ]] || continue
  pose=""
  if pose="$(pose_for_robot "$ns")"; then
    [[ -z "$INITIAL_POSE_ENTRIES" ]] || INITIAL_POSE_ENTRIES+=","
    INITIAL_POSE_ENTRIES+="$ns|$pose"
  else
    pose_status=$?
    [[ "$pose_status" == "1" ]] || exit "$pose_status"
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'container=%s robots=%s map=%s restart=%s rviz=%s\n' \
    "$CONTAINER_NAME" "$ROBOT_NAMESPACES" "$MAP_NAME" "$RESTART" "$RVIZ"
  printf 'initial_pose_config=%s\n' "$INITIAL_POSE_CONFIG"
  if [[ -n "$INITIAL_POSE_ENTRIES" ]]; then
    echo "enabled_saved_initial_poses=$INITIAL_POSE_ENTRIES"
  else
    echo "enabled_saved_initial_poses=none (localization will use 0,0,0 unless RViz is used)"
  fi
  exit 0
fi

command -v docker >/dev/null 2>&1 || {
  echo "docker is required" >&2
  exit 2
}

docker start "$CONTAINER_NAME" >/dev/null 2>&1 || {
  echo "could not start navigation container: $CONTAINER_NAME" >&2
  exit 2
}

# The offboard helper deliberately backgrounds ros2 launch processes.  Keep
# this exec detached from a controlling TTY so those processes survive after
# this wrapper returns (the RViz command also works without a pseudo-terminal).
docker exec -i \
  -e HAR_NAV_ROBOTS="$ROBOT_NAMESPACES" \
  -e HAR_NAV_MAP="$MAP_NAME" \
  -e HAR_NAV_RESTART="$RESTART" \
  -e HAR_NAV_RVIZ="$RVIZ" \
  -e HAR_NAV_INITIAL_POSES="$INITIAL_POSE_ENTRIES" \
  "$CONTAINER_NAME" bash -lc '
set -euo pipefail
set +u
source /opt/ros/jazzy/setup.bash
set -u
cd /root/rosbot2-jazzy-image/host/offboard

pid_alive() {
  local file="$1"
  [[ -s "$file" ]] && kill -0 "$(cat "$file")" 2>/dev/null
}

pose_for_namespace() {
  local wanted="$1"
  local entry entry_ns entry_x entry_y entry_yaw
  IFS="," read -r -a pose_entries <<< "${HAR_NAV_INITIAL_POSES:-}"
  for entry in "${pose_entries[@]}"; do
    [[ -n "$entry" ]] || continue
    IFS="|" read -r entry_ns entry_x entry_y entry_yaw <<< "$entry"
    if [[ "$entry_ns" == "$wanted" ]]; then
      printf "%s %s %s" "$entry_x" "$entry_y" "$entry_yaw"
      return 0
    fi
  done
  return 1
}

IFS="," read -r -a namespaces <<<"$HAR_NAV_ROBOTS"
started=0
for raw_namespace in "${namespaces[@]}"; do
  ns="${raw_namespace//[[:space:]]/}"
  [[ -n "$ns" ]] || continue
  if [[ "$HAR_NAV_RESTART" == "1" ]]; then
    ./rosbot-offboard stop "$ns" || true
  fi

  slam_pid_file="$HOME/.rosbot_offboard/pids/slam-$ns.pid"
  nav_pid_file="$HOME/.rosbot_offboard/pids/nav-$ns.pid"
  if pid_alive "$slam_pid_file"; then
    echo "localization already running: $ns"
  else
    saved_pose="$(pose_for_namespace "$ns" || true)"
    if [[ -n "$saved_pose" ]]; then
      read -r pose_x pose_y pose_yaw <<< "$saved_pose"
      echo "using saved initial pose: $ns map=$HAR_NAV_MAP x=$pose_x y=$pose_y yaw_rad=$pose_yaw"
      ./rosbot-offboard localize "$ns" "$HAR_NAV_MAP" "$pose_x" "$pose_y" "$pose_yaw"
    else
      echo "no enabled saved initial pose for $ns on map $HAR_NAV_MAP; using localization default 0,0,0"
      ./rosbot-offboard localize "$ns" "$HAR_NAV_MAP"
    fi
    sleep "${HAR_NAV_LOCALIZE_WAIT_SEC:-2}"
  fi

  if pid_alive "$nav_pid_file"; then
    echo "Nav2 already running: $ns"
  else
    ./rosbot-offboard nav "$ns"
  fi
  started=$((started + 1))
done

(( started > 0 )) || { echo "HAR_NAV_ROBOTS contains no namespace" >&2; exit 2; }

if [[ "$HAR_NAV_RVIZ" == "1" ]]; then
  ./rosbot-offboard rviz
fi

echo "Nav2 started/verified for: $HAR_NAV_ROBOTS"
echo "Map: $HAR_NAV_MAP"
for raw_namespace in "${namespaces[@]}"; do
  ns="${raw_namespace//[[:space:]]/}"
  [[ -n "$ns" ]] || continue
  ros2 lifecycle get "/$ns/slam_toolbox" || true
  ros2 lifecycle get "/$ns/controller_server" || true
  ros2 action info "/$ns/navigate_to_pose" || true
done
'
