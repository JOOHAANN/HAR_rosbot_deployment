#!/usr/bin/env bash
set -euo pipefail

ROOT="${HAR_DEPLOYMENT_ROOT:-/workspace/CLIPGCN}"
source "$ROOT/scripts/runtime_env.sh"

exec "$@"
