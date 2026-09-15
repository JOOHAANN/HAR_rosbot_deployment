#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/home/youhan/ws/CTR-GCN_17_full"
readonly CONDA="/opt/miniforge3/bin/conda"
readonly ENV_NAME="clipgcn"
readonly CONFIG="config/etri-coco17/ctrgcn_joint_coco17_13.yaml"
readonly DATASET="data/etri_coco17/ETRI_CS_45_25_10_20_rtmpose_coco17_13.npz"
readonly WORK_DIR="work_dir/etri_coco17/ctrgcn_joint_coco17_13_cs45_zsl50_5"

cd -- "$ROOT"

if pgrep -f 'build_etri_rgb_tensor.py|train_etri_x3d.py' >/dev/null; then
  echo "Refusing to start: the X3D pipeline is still using the machine." >&2
  exit 2
fi

read -r load_one _ < /proc/loadavg
cpu_count=$(nproc)
if ! awk -v load_value="$load_one" -v cpus="$cpu_count" \
  'BEGIN { exit !(load_value / cpus < 0.60) }'; then
  echo "Refusing to start: load average ${load_one} is too high for ${cpu_count} CPUs." >&2
  exit 2
fi

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if ((available_kib < 20 * 1024 * 1024)); then
  echo "Refusing to start: less than 20 GiB RAM is available." >&2
  exit 2
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset is not prepared: ${ROOT}/${DATASET}" >&2
  exit 3
fi

if [[ -e "$WORK_DIR" ]]; then
  echo "Refusing to overwrite existing work directory: ${ROOT}/${WORK_DIR}" >&2
  exit 3
fi

mkdir -p logs
exec "$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python -u main.py --config "$CONFIG"
