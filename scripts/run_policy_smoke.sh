#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
DEVICE="${HAR_POLICY_DEVICE:-cuda}"
exec python3 "$ROOT/viewpoint_policy.py" --smoke --device "$DEVICE"
