#!/usr/bin/env bash
# Shared environment setup for commands executed with docker exec.  The
# original VPOCLIP_rosbot container is intentionally kept as a long-running
# ROS 2 environment; these scripts are then launched inside it.

HAR_DEPLOYMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  # ROS setup files reference optional variables while being sourced.
  set +u
  source /opt/ros/jazzy/setup.bash
  if [[ -f /workspace/person_follow_ws/install/setup.bash ]]; then
    source /workspace/person_follow_ws/install/setup.bash
  fi
  set -u
fi

export PYTHONPATH="${HAR_DEPLOYMENT_ROOT}:${HAR_DEPLOYMENT_ROOT}/vendor/HAR_reproduction/ws/VPOCLIP_plus_full:${HAR_DEPLOYMENT_ROOT}/vendor/RTMPose_orientation_MLP/VPOCLIP_plus_full:${PYTHONPATH:-}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"

HAR_CUDA_LIB_ROOT="/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia"
if [[ -d "$HAR_CUDA_LIB_ROOT" ]]; then
  export LD_LIBRARY_PATH="${HAR_CUDA_LIB_ROOT}/cublas/lib:${HAR_CUDA_LIB_ROOT}/cudnn/lib:${HAR_CUDA_LIB_ROOT}/cuda_runtime/lib:${HAR_CUDA_LIB_ROOT}/cuda_cupti/lib:${HAR_CUDA_LIB_ROOT}/cuda_nvrtc/lib:${HAR_CUDA_LIB_ROOT}/cufft/lib:${HAR_CUDA_LIB_ROOT}/curand/lib:${HAR_CUDA_LIB_ROOT}/cusolver/lib:${HAR_CUDA_LIB_ROOT}/cusparse/lib:${HAR_CUDA_LIB_ROOT}/nccl/lib:${HAR_CUDA_LIB_ROOT}/nvjitlink/lib:${HAR_CUDA_LIB_ROOT}/nvtx/lib:${LD_LIBRARY_PATH:-}"
fi

