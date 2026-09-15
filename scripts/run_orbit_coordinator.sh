#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
exec python3 "$ROOT/orbit_coordinator.py" \
  --config "${HAR_ORBIT_CONFIG:-$ROOT/config/orbit_system_config.csv}" \
  "$@"
