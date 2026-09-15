#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
ORBIT_CONFIG="${HAR_ORBIT_CONFIG:-$ROOT/config/orbit_system_config.csv}"
if [[ "$ORBIT_CONFIG" != /* ]]; then
  ORBIT_CONFIG="$ROOT/$ORBIT_CONFIG"
fi
[[ -f "$ORBIT_CONFIG" ]] || { echo "orbit movement config table not found: $ORBIT_CONFIG" >&2; exit 2; }

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

orbit_config_or_env() {
  local env_name="$1"
  local key="$2"
  local fallback="${3:-}"
  local env_value="${!env_name-}"
  if [[ -n "$env_value" ]]; then printf '%s' "$env_value"; else orbit_config_value "$key" "$fallback"; fi
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

fusion_args=(
  "$ROOT/vpoclip_fusion_ros.py"
  --trigger-topic "$(orbit_config_or_env HAR_ORBIT_RECOGNITION_TOPIC recognition_cycle_topic /har/orbit/recognition_cycle)" \
  --prediction-topic-1 "$(orbit_config_or_env HAR_VPOCLIP_PREDICTION_TOPIC_1 prediction_topic_1 /rosbot_1/vpoclip/prediction)" \
  --prediction-topic-2 "$(orbit_config_or_env HAR_VPOCLIP_PREDICTION_TOPIC_2 prediction_topic_2 /rosbot_2/vpoclip/prediction)" \
  --output-topic "$(orbit_config_or_env HAR_ORBIT_FUSION_TOPIC fusion_topic /har/final_action)" \
  --timeout-sec "$(orbit_config_value fusion_timeout_sec 10.0)" \
  --robot1-weight "$(orbit_config_value fusion_robot1_weight 1.0)" \
  --robot2-weight "$(orbit_config_value fusion_robot2_weight 1.0)" \
  --top-k "$(orbit_config_value fusion_top_k 5)" \
  --score-field "$(orbit_config_or_env HAR_ORBIT_FUSION_SCORE_FIELD fusion_score_field candidate_ranking_scores)"
)

if [[ "$(orbit_bool fusion_continuous "$(orbit_config_value fusion_continuous true)")" == "1" ]]; then
  fusion_args+=(--continuous)
fi

exec python3 "${fusion_args[@]}"
