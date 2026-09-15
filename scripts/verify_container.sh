#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
python3 - "$ROOT" <<'PY'
import importlib
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
required = {
    "clip checkpoint": root / "local_models/clip/ViT-B-32.pt",
    "CLIPGCN checkpoint": root / "vendor/HAR_reproduction/ws/VPOCLIP_plus_full/work_dir/merged45_10_groupAB/selected_stage_b/last_model.pth",
    "DDQN checkpoint": root / "vendor/HAR_reproduction/ws/VPOCLIP_plus_full/work_dir/dual_robot_depth_ddqn_strict45_5/models/full_depth19/seed_20260911/best.pt",
    "orientation checkpoint": root / "vendor/RTMPose_orientation_MLP/work_dir/rtmpose_orientation_mlp/best.pt",
    "X3D checkpoint": root / "vendor/HAR_reproduction/ws/X3D_full/outputs/etri_allviews_cs45_merged_groupAB_7000/model_best.pth",
    "CTR-GCN checkpoint": root / "vendor/HAR_reproduction/ws/CTR-GCN_17_full/work_dir/etri_coco17/allviews_cs45_merged_groupAB/runs-51-4029.pt",
    "YOLO checkpoint": root / "vendor/HAR_reproduction/ws/yolov5_full/best.pt",
    "RTMPose detector": root / "vendor/HAR_reproduction/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
    "RTMPose pose": root / "vendor/HAR_reproduction/.cache/rtmlib/hub/checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx",
}
missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
if missing:
    raise SystemExit("Missing deployment assets:\n  " + "\n  ".join(missing))

import torch
print(f"python={sys.executable}")
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")

import rclpy
import cv2
import onnxruntime
import rtmlib
print(f"ros={rclpy.__file__}")
print(f"opencv={cv2.__version__}")
print(f"onnxruntime={onnxruntime.__version__} providers={onnxruntime.get_available_providers()}")
print(f"rtmlib={rtmlib.__file__}")

sys.path.insert(0, str(root))
from viewpoint_policy import LiveViewPolicy
policy = LiveViewPolicy(
    checkpoint=required["DDQN checkpoint"],
    orientation_checkpoint=required["orientation checkpoint"],
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print(f"ddqn=loaded parameters={sum(parameter.numel() for parameter in policy.model.parameters())}")
print("deployment assets and imports: OK")
PY
