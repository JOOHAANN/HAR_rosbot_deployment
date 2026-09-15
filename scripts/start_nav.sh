#!/usr/bin/env bash
set -euo pipefail

# User-facing entry point for the existing, separate jazzy-rosbot Nav2
# container. No Nav2 process is started in the HAR/VPOCLIP container.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${HAR_NAV_CONTAINER_NAME:-jazzy-rosbot}"
ORBIT_CONFIG="${HAR_ORBIT_CONFIG:-$ROOT/config/orbit_system_config.csv}"
ROBOT_NAMESPACES="${HAR_NAV_ROBOTS:-}"
MAP_NAME="${HAR_NAV_MAP:-}"
INITIAL_POSE_CONFIG="${HAR_INITIAL_POSE_CONFIG:-}"
RESTART=0
RVIZ=0
DRY_RUN=0

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

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start_nav.sh [options]

Options:
  --robots LIST   Comma-separated namespaces (default: rosbot_1,rosbot_2)
  --map NAME      Saved map name (default: home)
  --orbit-config FILE
                  Movement/human CSV; relative mode derives robot initial poses
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
    --orbit-config)
      [[ $# -ge 2 ]] || { echo "--orbit-config needs a file" >&2; exit 2; }
      ORBIT_CONFIG="$2"
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

if [[ "$ORBIT_CONFIG" != /* ]]; then
  ORBIT_CONFIG="$ROOT/$ORBIT_CONFIG"
fi
[[ -f "$ORBIT_CONFIG" ]] || {
  echo "orbit movement config table not found: $ORBIT_CONFIG" >&2
  exit 2
}

[[ -n "$ROBOT_NAMESPACES" ]] || {
  ROBOT_NAMESPACES="$(orbit_config_value robot_namespaces 'rosbot_1;rosbot_2')"
}
ROBOT_NAMESPACES="${ROBOT_NAMESPACES//;/,}"
[[ -n "$MAP_NAME" ]] || MAP_NAME="$(orbit_config_value map_name home)"
[[ -n "$INITIAL_POSE_CONFIG" ]] || {
  INITIAL_POSE_CONFIG="$(orbit_config_value initial_pose_config config/robot_initial_poses.csv)"
}
INITIAL_POSE_MODE="$(orbit_config_value initial_robot_pose_mode relative_human_slots)"
case "$INITIAL_POSE_MODE" in
  relative_human_slots|manual_csv) ;;
  *)
    echo "initial_robot_pose_mode must be relative_human_slots or manual_csv: $INITIAL_POSE_MODE" >&2
    exit 2
    ;;
esac

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

# Derive the startup pose from the fixed/online human pose.  The configured
# ring angles are slot offsets; human_front_slot is the zero-degree reference.
# The resulting yaw always points from the robot back to the human.
relative_pose_for_robot() {
  local target_slot="$1"
  local orbit_map radius human_x human_y human_yaw front_slot angles
  [[ "$INITIAL_POSE_MODE" == "relative_human_slots" ]] || return 1
  orbit_map="$(orbit_config_value map_name home)"
  [[ "$orbit_map" == "$MAP_NAME" ]] || return 1
  radius="$(orbit_config_value ring_radius_m 1.5)"
  human_x="$(orbit_config_value human_x_m 0.0)"
  human_y="$(orbit_config_value human_y_m 0.0)"
  human_yaw="$(orbit_config_value human_yaw_deg 0.0)"
  front_slot="$(orbit_config_value human_front_slot 1)"
  angles="$(orbit_config_value ring_angles_deg '270;0;90;180')"
  awk -v slot="$target_slot" -v radius="$radius" -v human_x="$human_x" \
    -v human_y="$human_y" -v human_yaw="$human_yaw" -v front_slot="$front_slot" \
    -v angles="$angles" '
    BEGIN {
      pi = 3.141592653589793
      gsub(/,/, ";", angles)
      count = split(angles, angle, /;/)
      if (count != 4 || slot < 0 || slot >= 4 || front_slot < 0 || front_slot >= 4) exit 2
      for (i = 1; i <= count; i++) {
        if (angle[i] !~ /^[[:space:]]*[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?[[:space:]]*$/) exit 2
      }
      front = angle[front_slot + 1] + 0
      theta = (human_yaw + (angle[slot + 1] + 0) - front) * pi / 180.0
      x = human_x + radius * cos(theta)
      y = human_y + radius * sin(theta)
      yaw = atan2(human_y - y, human_x - x)
      printf "%.12f|%.12f|%.12f", x, y, yaw
    }
  '
}

INITIAL_POSE_ENTRIES=""
IFS="," read -r -a pose_namespaces <<< "$ROBOT_NAMESPACES"
initial_slots_text="$(orbit_config_value initial_robot_slots '0;1')"
initial_slots_text="${initial_slots_text//,/;}"
IFS=";" read -r -a initial_slots <<< "$initial_slots_text"
if [[ "${#initial_slots[@]}" != "2" || "${initial_slots[0]:-}" == "${initial_slots[1]:-}" ]]; then
  echo "initial_robot_slots must contain two distinct values" >&2
  exit 2
fi
for initial_slot in "${initial_slots[@]}"; do
  [[ "$initial_slot" =~ ^[0-3]$ ]] || {
    echo "initial_robot_slots values must be integers in 0..3" >&2
    exit 2
  }
done
derived_initial_poses=""
robot_index=0
for raw_namespace in "${pose_namespaces[@]}"; do
  ns="${raw_namespace//[[:space:]]/}"
  [[ -n "$ns" ]] || continue
  pose=""
  initial_slot="${initial_slots[$robot_index]:-}"
  used_relative=0
  if [[ "$INITIAL_POSE_MODE" == "relative_human_slots" && -n "$initial_slot" ]]; then
    if pose="$(relative_pose_for_robot "$initial_slot")"; then
      [[ -z "$INITIAL_POSE_ENTRIES" ]] || INITIAL_POSE_ENTRIES+=","
      INITIAL_POSE_ENTRIES+="$ns|$pose"
      [[ -z "$derived_initial_poses" ]] || derived_initial_poses+=","
      derived_initial_poses+="$ns|slot$initial_slot|$pose"
      used_relative=1
    else
      pose_status=$?
      [[ "$pose_status" == "1" ]] || exit "$pose_status"
    fi
  fi
  if [[ "$used_relative" == "0" ]]; then
    if pose="$(pose_for_robot "$ns")"; then
      [[ -z "$INITIAL_POSE_ENTRIES" ]] || INITIAL_POSE_ENTRIES+=","
      INITIAL_POSE_ENTRIES+="$ns|$pose"
    else
      pose_status=$?
      [[ "$pose_status" == "1" ]] || exit "$pose_status"
    fi
  fi
  robot_index=$((robot_index + 1))
done

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'container=%s robots=%s map=%s restart=%s rviz=%s\n' \
    "$CONTAINER_NAME" "$ROBOT_NAMESPACES" "$MAP_NAME" "$RESTART" "$RVIZ"
  printf 'orbit_config=%s initial_robot_pose_mode=%s\n' "$ORBIT_CONFIG" "$INITIAL_POSE_MODE"
  printf 'initial_pose_config=%s\n' "$INITIAL_POSE_CONFIG"
  if [[ -n "$derived_initial_poses" ]]; then
    echo "derived_relative_initial_poses=$derived_initial_poses"
  fi
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
