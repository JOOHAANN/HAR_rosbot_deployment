#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
ORBIT_CONFIG="${HAR_ORBIT_CONFIG:-$ROOT/config/orbit_system_config.csv}"
if [[ "$ORBIT_CONFIG" != /* ]]; then
  ORBIT_CONFIG="$ROOT/$ORBIT_CONFIG"
fi
if [[ ! -f "$ORBIT_CONFIG" ]]; then
  echo "orbit movement config table not found: $ORBIT_CONFIG" >&2
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
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
  else
    printf '%s' "$fallback"
  fi
}

orbit_config_or_env() {
  local env_name="$1"
  local key="$2"
  local fallback="${3:-}"
  local env_value="${!env_name-}"
  if [[ -n "$env_value" ]]; then
    printf '%s' "$env_value"
  else
    orbit_config_value "$key" "$fallback"
  fi
}

orbit_bool() {
  local key="$1"
  local value="${2:-}"
  case "${value,,}" in
    1|true|yes|y|on|enable|enabled) printf '1' ;;
    0|false|no|n|off|disable|disabled) printf '0' ;;
    *)
      echo "Invalid boolean for $key in $ORBIT_CONFIG: $value" >&2
      exit 2
      ;;
  esac
}

resolve_project_path() {
  local value="${1:-}"
  if [[ -z "$value" || "${value,,}" == "none" ]]; then
    printf ''
  elif [[ "$value" == /* ]]; then
    printf '%s' "$value"
  else
    printf '%s/%s' "$ROOT" "$value"
  fi
}

csv_semicolon_to_comma() {
  local value="$1"
  printf '%s' "${value//;/,}"
}

VIEW_TOPICS="$(csv_semicolon_to_comma "$(orbit_config_or_env HAR_VIEW_TOPICS view_topics '/rosbot_1/camera/rgb/image_raw/compressed;/rosbot_2/camera/rgb/image_raw/compressed;/har/view/slot_2/compressed;/har/view/slot_3/compressed')")"
ROBOT_TOPICS="$(csv_semicolon_to_comma "$(orbit_config_or_env HAR_VIEW_ROBOT_TOPICS robot_rgb_topics '/rosbot_1/camera/rgb/image_raw/compressed;/rosbot_2/camera/rgb/image_raw/compressed')")"
ROBOT_NAMESPACES="$(csv_semicolon_to_comma "$(orbit_config_or_env HAR_VIEW_ROBOT_NAMESPACES robot_namespaces 'rosbot_1;rosbot_2')")"
RING_ANGLES="$(csv_semicolon_to_comma "$(orbit_config_or_env HAR_VIEW_ANGLES_DEG ring_angles_deg '0;90;180;270')")"
INITIAL_SLOTS="$(orbit_config_or_env HAR_VIEW_INITIAL_SLOTS initial_robot_slots '0;1')"
INITIAL_SLOTS="${INITIAL_SLOTS//,/;}"
IFS=';' read -r START_A START_B _ <<<"$INITIAL_SLOTS"
START_A="${START_A//[[:space:]]/}"
START_B="${START_B//[[:space:]]/}"
[[ -n "$START_A" && -n "$START_B" ]] || { echo "initial_robot_slots must contain two values" >&2; exit 2; }
SELECTION_MODE="$(orbit_config_or_env HAR_VIEW_SELECTION_MODE selection_mode rl)"
VIEW_OUTPUT_TOPIC="$(orbit_config_or_env HAR_POLICY_OUTPUT_TOPIC selected_view_topic /har/selected_view_pair)"
IMAGE_TRANSPORT="$(orbit_config_or_env HAR_IMAGE_TRANSPORT view_image_transport compressed)"
RATE_HZ="$(orbit_config_or_env HAR_VIEW_POLICY_RATE_HZ active_view_rate_hz 1.0)"
INPUT_STALE_SECONDS="$(orbit_config_or_env HAR_VIEW_INPUT_STALE_SECONDS active_view_input_stale_seconds 3.0)"
VARIANT="$(orbit_config_or_env HAR_POLICY_VARIANT active_view_variant full_depth19)"
POLICY_DEVICE="$(orbit_config_or_env HAR_POLICY_DEVICE active_view_policy_device cuda)"
VIEW_RTMPOSE_DEVICE="$(orbit_config_or_env HAR_VIEW_RTMPOSE_DEVICE active_view_rtmpose_device cpu)"
RTMPOSE_MODE="$(orbit_config_or_env HAR_VIEW_RTMPOSE_MODE active_view_rtmpose_mode balanced)"
RTMPOSE_BACKEND="$(orbit_config_or_env HAR_VIEW_RTMPOSE_BACKEND active_view_rtmpose_backend onnxruntime)"
RANDOM_SEED="$(orbit_config_or_env HAR_VIEW_RANDOM_SEED active_view_random_seed 20260915)"
BASELINE_HOLD="$(orbit_config_or_env HAR_VIEW_BASELINE_HOLD_SECONDS baseline_hold_seconds 3.0)"
SLOT_MAP_TOPIC="$(orbit_config_or_env HAR_ORBIT_SLOT_MAP_TOPIC slot_map_topic /har/orbit/robot_slots)"
TRIGGER_TOPIC="$(orbit_config_or_env HAR_ORBIT_SELECT_NEXT_TOPIC select_next_topic /har/orbit/select_next)"
POLICY_CHECKPOINT="$(resolve_project_path "$(orbit_config_or_env HAR_POLICY_CHECKPOINT active_view_policy_checkpoint)")"
ORIENTATION_CHECKPOINT="$(resolve_project_path "$(orbit_config_or_env HAR_ORIENTATION_CHECKPOINT active_view_orientation_checkpoint)")"
RTMPOSE_ROOT="$ROOT/vendor/HAR_reproduction/.cache/rtmlib/hub/checkpoints"

# Use the --name=value form because the default angle value starts with -90
# and argparse otherwise treats it as another option.
# The active-view policy and its RTMPose feature extractor are independently
# configurable. The deployment table currently selects CUDA for both.
args=(
  "$ROOT/active_view_ros.py"
  --view-topics "$VIEW_TOPICS"
  --robot-topics "$ROBOT_TOPICS"
  --robot-namespaces "$ROBOT_NAMESPACES"
  --image-transport "$IMAGE_TRANSPORT"
  --ros-qos-reliability "${HAR_ROS_QOS_RELIABILITY:-best-effort}"
  --ros-node-name "${HAR_POLICY_NODE_NAME:-har_active_view_policy}"
  --output-topic "$VIEW_OUTPUT_TOPIC"
  --rate-hz "$RATE_HZ"
  --input-stale-seconds "$INPUT_STALE_SECONDS"
  "--view-angles-deg=$RING_ANGLES"
  --selection-mode "$SELECTION_MODE"
  --random-seed "$RANDOM_SEED"
  --baseline-hold-seconds "$BASELINE_HOLD"
  --start-a "$START_A"
  --start-b "$START_B"
  --slot-map-topic "$SLOT_MAP_TOPIC"
  --trigger-topic "$TRIGGER_TOPIC"
  --variant "$VARIANT"
  --policy-device "$POLICY_DEVICE"
  --rtmpose-mode "$RTMPOSE_MODE"
  --rtmpose-backend "$RTMPOSE_BACKEND"
  --rtmpose-device "$VIEW_RTMPOSE_DEVICE"
  --rtmpose-det "$RTMPOSE_ROOT/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
  --rtmpose-pose "$RTMPOSE_ROOT/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx"
)

if [[ -n "$POLICY_CHECKPOINT" ]]; then
  args+=(--policy-checkpoint "$POLICY_CHECKPOINT")
fi
if [[ -n "$ORIENTATION_CHECKPOINT" ]]; then
  args+=(--orientation-checkpoint "$ORIENTATION_CHECKPOINT")
fi
if [[ "$(orbit_bool active_view_allow_unobserved_candidates "$(orbit_config_value active_view_allow_unobserved_candidates true)")" == "1" ]]; then
  args+=(--allow-unobserved-candidates)
fi
if [[ "$(orbit_bool active_view_wait_for_trigger "$(orbit_config_value active_view_wait_for_trigger true)")" == "1" ]]; then
  args+=(--wait-for-trigger)
fi

exec python3 "${args[@]}"
