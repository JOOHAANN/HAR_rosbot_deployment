#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"

# Keep all VPOCLIP knobs in one editable CSV. Values are intentionally simple
# (no commas); the third CSV column is documentation and is ignored here.
VPOCLIP_CONFIG="${HAR_VPOCLIP_CONFIG:-$ROOT/vpoclip_config.csv}"
if [[ "$VPOCLIP_CONFIG" != /* ]]; then
  VPOCLIP_CONFIG="$ROOT/$VPOCLIP_CONFIG"
fi
if [[ ! -f "$VPOCLIP_CONFIG" ]]; then
  echo "VPOCLIP config table not found: $VPOCLIP_CONFIG" >&2
  exit 2
fi

config_value() {
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
  ' "$VPOCLIP_CONFIG")"
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
  else
    printf '%s' "$fallback"
  fi
}

config_or_env() {
  local env_name="$1"
  local key="$2"
  local fallback="${3:-}"
  local env_value="${!env_name-}"
  if [[ -n "$env_value" ]]; then
    printf '%s' "$env_value"
  else
    config_value "$key" "$fallback"
  fi
}

as_bool() {
  local key="$1"
  local value="${2:-}"
  case "${value,,}" in
    1|true|yes|y|on|enable|enabled|active) printf '1' ;;
    0|false|no|n|off|disable|disabled|inactive) printf '0' ;;
    *)
      echo "Invalid boolean for $key in $VPOCLIP_CONFIG: $value" >&2
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

add_option() {
  local flag="$1"
  local value="${2:-}"
  if [[ -n "$value" && "${value,,}" != "none" ]]; then
    args+=("$flag" "$value")
  fi
}

ROBOT_NAMESPACE="${HAR_ROBOT_NAMESPACE:-$(config_value robot_namespace rosbot_1)}"
IMAGE_TRANSPORT="$(config_or_env HAR_IMAGE_TRANSPORT image_transport compressed)"
ROS_QOS="$(config_or_env HAR_ROS_QOS_RELIABILITY ros_qos_reliability best-effort)"
ROS_WAIT_TIMEOUT="$(config_or_env HAR_ROS_WAIT_TIMEOUT ros_wait_timeout 3600)"
ROS_NODE_NAME="$(config_or_env HAR_RECOGNIZER_NODE_NAME ros_node_name har_vpoclip_recognizer)"
RUNTIME_DEVICE="$(config_or_env HAR_RUNTIME_DEVICE runtime_device config)"
RTMPOSE_DEVICE="$(config_or_env HAR_RECOGNIZER_RTMPOSE_DEVICE rtmpose_device cpu)"
DISPLAY_ENABLED="$(config_or_env HAR_DISPLAY display false)"
FOLLOW_OUTPUT="$(config_or_env HAR_ENABLE_FOLLOW enable_person_follow_output false)"
FUSION_DISPLAY_ENABLED="$(config_or_env HAR_FUSION_DISPLAY_ENABLED fusion_display_enabled false)"
FUSION_DISPLAY_ROBOT="$(config_value fusion_display_robot rosbot_1)"
FUSION_DISPLAY_TOPIC="$(config_value fusion_display_topic /har/final_action)"
FUSION_DISPLAY_STALE_SECONDS="$(config_value fusion_display_stale_seconds 15.0)"
WINDOW_NAME="$(config_or_env HAR_WINDOW_NAME window_name 'CLIPGCN {namespace}')"
WINDOW_NAME="${WINDOW_NAME//\{namespace\}/$ROBOT_NAMESPACE}"

CONFIG_PATH="$(resolve_project_path "$(config_value config config/config_runtime.yaml)")"
CLASS_SPLIT_DIR="$(resolve_project_path "$(config_value class_split_dir config/class_splits/strict45_10)")"
CLASS_CONFIG="$(resolve_project_path "$(config_value class_config realtime_class_config.csv)")"
CLIPGCN_CHECKPOINT="$(resolve_project_path "$(config_value clipgcn_checkpoint)")"
X3D_ROOT="$(resolve_project_path "$(config_value x3d_root X3D)")"
X3D_CONFIG="$(resolve_project_path "$(config_value x3d_config X3D/configs/x3d-s_clipgcn_tensor_cross_subject_70_10_20_182.yaml)")"
X3D_CHECKPOINT="$(resolve_project_path "$(config_value x3d_checkpoint)")"
CTRGCN_ROOT="$(resolve_project_path "$(config_value ctrgcn_root CTR-GCN_17)")"
CTRGCN_CONFIG="$(resolve_project_path "$(config_value ctrgcn_config CTR-GCN_17/config/etri-coco17/ctrgcn_joint_coco17_13.yaml)")"
CTRGCN_CHECKPOINT="$(resolve_project_path "$(config_value ctrgcn_weights)")"
YOLO_REPO="$(resolve_project_path "$(config_value yolo_repo yolov5)")"
YOLO_CHECKPOINT="$(resolve_project_path "$(config_value yolo_weights)")"
RTMPOSE_DET="$(resolve_project_path "$(config_value rtmpose_det)")"
RTMPOSE_POSE="$(resolve_project_path "$(config_value rtmpose_pose)")"

args=(
  "$ROOT/ros_realtime.py"
  --robot-namespace "$ROBOT_NAMESPACE"
  --image-transport "$IMAGE_TRANSPORT"
  --ros-qos-reliability "$ROS_QOS"
  --ros-node-name "$ROS_NODE_NAME"
  --ros-wait-timeout "$ROS_WAIT_TIMEOUT"
  --config "$CONFIG_PATH"
  --clipgcn-checkpoint "$CLIPGCN_CHECKPOINT"
  --class-split-dir "$CLASS_SPLIT_DIR"
  --candidate-scope "$(config_or_env HAR_CANDIDATE_SCOPE candidate_scope all)"
  --unseen-score-scale "$(config_or_env HAR_UNSEEN_SCORE_SCALE unseen_score_scale 1.03)"
  --pose-source "$(config_value pose_source rtmpose)"
  --rtmpose-mode "$(config_value rtmpose_mode balanced)"
  --rtmpose-backend "$(config_value rtmpose_backend onnxruntime)"
  --rtmpose-device "$RTMPOSE_DEVICE"
  --x3d-root "$X3D_ROOT"
  --x3d-config "$X3D_CONFIG"
  --x3d-checkpoint "$X3D_CHECKPOINT"
  --x3d-layer "$(config_value x3d_layer s5)"
  --ctrgcn-root "$CTRGCN_ROOT"
  --ctrgcn-config "$CTRGCN_CONFIG"
  --ctrgcn-weights "$CTRGCN_CHECKPOINT"
  --ctrgcn-hook-layer "$(config_value ctrgcn_hook_layer l4)"
  --yolo-repo "$YOLO_REPO"
  --yolo-weights "$YOLO_CHECKPOINT"
  --yolo-size "$(config_value yolo_size 640)"
  --yolo-conf "$(config_value yolo_conf 0.25)"
  --yolo-iou "$(config_value yolo_iou 0.45)"
  --yolo-detect-every "$(config_or_env HAR_YOLO_DETECT_EVERY yolo_detect_every 1)"
  --frames "$(config_value frames 13)"
  --temporal-strategy "$(config_value temporal_strategy uniform3s)"
  --uniform-window-seconds "$(config_or_env HAR_UNIFORM_WINDOW_SECONDS uniform_window_seconds 3)"
  --short-window-seconds "$(config_value short_window_seconds 2)"
  --long-window-seconds "$(config_value long_window_seconds 4)"
  --long-rerank-top-k "$(config_value long_rerank_top_k 5)"
  --predict-every "$(config_or_env HAR_PREDICT_EVERY predict_every 1)"
  --top-k "$(config_or_env HAR_TOP_K top_k 1)"
  --display-filter-window "$(config_or_env HAR_DISPLAY_FILTER_WINDOW display_filter_window 10)"
  --display-width "$(config_value display_width 1280)"
  --display-height "$(config_value display_height 0)"
  --window-name "$WINDOW_NAME"
)

# "config"/"yaml" leaves config_runtime.yaml in charge of PyTorch while
# preserving the independent rtmpose_device setting. Passing --runtime-device
# cuda/cpu/auto intentionally couples the whole stack to that device.
case "${RUNTIME_DEVICE,,}" in
  config|yaml|from_config|none) ;;
  *) args+=(--runtime-device "$RUNTIME_DEVICE") ;;
esac

add_option --class-config "$CLASS_CONFIG"
add_option --image-topic "$(config_value image_topic)"
add_option --decision-entropy-threshold "$(config_value decision_entropy_threshold)"
add_option --decision-temperature "$(config_value decision_temperature)"
add_option --rtmpose-det "$RTMPOSE_DET"
add_option --rtmpose-pose "$RTMPOSE_POSE"
add_option --person-detection-topic "$(config_value person_detection_topic)"
add_option --person-detection-frame-id "$(config_value person_detection_frame_id)"
add_option --prediction-topic "$(config_value prediction_topic)"
if [[ "$ROBOT_NAMESPACE" == "$FUSION_DISPLAY_ROBOT" ]] && \
   [[ "$(as_bool fusion_display_enabled "$FUSION_DISPLAY_ENABLED")" == "1" ]]; then
  add_option --fusion-topic "$FUSION_DISPLAY_TOPIC"
  add_option --fusion-stale-seconds "$FUSION_DISPLAY_STALE_SECONDS"
fi
add_option --person-confidence-threshold "$(config_value person_confidence_threshold 0.35)"
add_option --person-lock-iou-threshold "$(config_value person_lock_iou_threshold 0.15)"
add_option --person-max-center-jump "$(config_value person_max_center_jump 0.20)"
add_option --person-max-lost-frames "$(config_value person_max_lost_frames 5)"
add_option --object-grid-size "$(config_value object_grid_size 6)"
add_option --object-value "$(config_value object_value presence)"
add_option --object-max-distance-weight "$(config_value object_max_distance_weight 10.0)"

if [[ "$(as_bool cudnn_benchmark "$(config_value cudnn_benchmark false)")" == "1" ]]; then
  args+=(--cudnn-benchmark)
fi
if [[ "$(as_bool yolo_half "$(config_value yolo_half false)")" == "1" ]]; then
  args+=(--yolo-half)
fi
if [[ "$(as_bool no_yolo "$(config_value no_yolo false)")" == "1" ]]; then
  args+=(--no-yolo)
fi
if [[ "$(as_bool hide_person_boxes "$(config_value hide_person_boxes false)")" == "1" ]]; then
  args+=(--hide-person-boxes)
fi
if [[ "$(as_bool allow_cpu "$(config_value allow_cpu false)")" == "1" ]]; then
  args+=(--allow-cpu)
fi
if [[ "$(as_bool all_classes_seen "$(config_value all_classes_seen false)")" == "1" ]]; then
  args+=(--all-classes-seen)
fi

add_option --seen-classes "$(config_value seen_classes)"
add_option --unseen-classes "$(config_value unseen_classes)"
add_option --exclude-classes "$(config_value exclude_classes)"

if [[ "$(as_bool "$DISPLAY_ENABLED" "$DISPLAY_ENABLED")" != "1" ]]; then
  args+=(--headless)
fi

if [[ "$(as_bool enable_person_follow_output "$FOLLOW_OUTPUT")" == "1" ]]; then
  args+=(--enable-person-follow-output)
fi

if [[ "${HAR_PRINT_COMMAND:-0}" == "1" ]]; then
  printf 'VPOCLIP command from %s:\n' "$VPOCLIP_CONFIG"
  printf '  %q' python3
  printf ' %q' "${args[@]}"
  printf '\n'
  exit 0
fi

exec python3 "${args[@]}"
