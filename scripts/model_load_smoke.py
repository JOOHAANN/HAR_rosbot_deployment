#!/usr/bin/env python3
"""Load every model used by the Docker ROS runtime and exit.

This intentionally follows the same path resolution and loader functions as
``webcam_realtime.py``.  It catches missing local weights, incompatible
checkpoints, and missing Python/runtime dependencies without requiring a live
camera topic.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from webcam_realtime import (
    DEFAULT_ACTION_DURATION_TYPES,
    RTMPoseSource,
    apply_runtime_device_override,
    load_action_display_names,
    load_clipgcn_model,
    load_ctrgcn_model,
    load_x3d_model,
    load_yolo_model,
    parse_args,
    resolve_existing_path,
    resolve_pose_backend,
    resolve_realtime_class_labels,
    resolve_runtime_pose_source,
    validate_args,
)
from train import get_device, get_path_from_config, load_config, print_device_info


def main():
    rtmpose_root = ROOT / "vendor" / "HAR_reproduction" / ".cache" / "rtmlib" / "hub" / "checkpoints"
    args = parse_args(
        [
            "--config",
            str(ROOT / "config" / "config_runtime.yaml"),
            "--class-split-dir",
            str(ROOT / "config" / "class_splits" / "strict45_10"),
            "--candidate-scope",
            "all",
            "--clipgcn-checkpoint",
            str(
                ROOT
                / "vendor"
                / "HAR_reproduction"
                / "ws"
                / "VPOCLIP_plus_full"
                / "work_dir"
                / "merged45_10_groupAB"
                / "selected_stage_b"
                / "last_model.pth"
            ),
            "--runtime-device",
            "cuda",
            "--pose-source",
            "rtmpose",
            "--rtmpose-device",
            "cuda",
            "--rtmpose-det",
            str(rtmpose_root / "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"),
            "--rtmpose-pose",
            str(rtmpose_root / "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx"),
            "--x3d-root",
            str(ROOT / "X3D"),
            "--x3d-config",
            str(ROOT / "X3D" / "configs" / "x3d-s_clipgcn_tensor_cross_subject_70_10_20_182.yaml"),
            "--x3d-checkpoint",
            str(
                ROOT
                / "vendor"
                / "HAR_reproduction"
                / "ws"
                / "X3D_full"
                / "outputs"
                / "etri_allviews_cs45_merged_groupAB_7000"
                / "model_best.pth"
            ),
            "--ctrgcn-root",
            str(ROOT / "CTR-GCN_17"),
            "--ctrgcn-config",
            str(ROOT / "CTR-GCN_17" / "config" / "etri-coco17" / "ctrgcn_joint_coco17_13.yaml"),
            "--ctrgcn-weights",
            str(
                ROOT
                / "vendor"
                / "HAR_reproduction"
                / "ws"
                / "CTR-GCN_17_full"
                / "work_dir"
                / "etri_coco17"
                / "allviews_cs45_merged_groupAB"
                / "runs-51-4029.pt"
            ),
            "--yolo-repo",
            str(ROOT / "yolov5"),
            "--yolo-weights",
            str(ROOT / "vendor" / "HAR_reproduction" / "ws" / "yolov5_full" / "best.pt"),
            "--yolo-half",
            "--headless",
        ]
    )
    validate_args(args)
    args.action_duration_types = DEFAULT_ACTION_DURATION_TYPES

    config_path = resolve_existing_path(args.config, label="config")
    config = load_config(config_path)
    args = resolve_pose_backend(args, config)
    args.runtime_pose_source = resolve_runtime_pose_source(args)
    apply_runtime_device_override(args, config)
    args.label_display_names = load_action_display_names(config, config_path)

    args.x3d_root = resolve_existing_path(args.x3d_root, label="X3D root")
    args.x3d_config = resolve_existing_path(args.x3d_config, label="X3D config")
    args.x3d_checkpoint = resolve_existing_path(args.x3d_checkpoint, label="X3D checkpoint")
    args.ctrgcn_root = resolve_existing_path(args.ctrgcn_root, label="CTR-GCN root")
    args.ctrgcn_config = resolve_existing_path(args.ctrgcn_config, label="CTR-GCN config")
    args.ctrgcn_weights = resolve_existing_path(args.ctrgcn_weights, label="CTR-GCN weights")
    args.yolo_repo = resolve_existing_path(args.yolo_repo, label="YOLO repo")
    args.yolo_weights = resolve_existing_path(args.yolo_weights, label="YOLO weights")

    device = get_device(config["runtime"]["device"])
    print_device_info(device)
    class_split_dir = get_path_from_config(config_path, args.class_split_dir)
    selection = resolve_realtime_class_labels(
        args,
        class_split_dir,
        all_labels=args.label_display_names.keys(),
    )

    x3d_model = ctrgcn_model = yolo_model = pose_source = None
    x3d_hook = ctrgcn_hook = None
    try:
        print("[smoke] loading X3D...", flush=True)
        x3d_model, _x3d_captured, x3d_hook, _x3d_cfg = load_x3d_model(args, device)
        print("[smoke] loading CTR-GCN...", flush=True)
        ctrgcn_model, _ctrgcn_captured, ctrgcn_hook = load_ctrgcn_model(args, device)
        print("[smoke] loading YOLO...", flush=True)
        yolo_model = load_yolo_model(args, device)
        print("[smoke] loading CLIPGCN + local CLIP text encoder...", flush=True)
        clipgcn_model, checkpoint_path, candidate_labels = load_clipgcn_model(
            args,
            config,
            config_path,
            device,
            selection["candidate_labels"],
        )
        print("[smoke] initializing RTMPose...", flush=True)
        pose_source = RTMPoseSource(args)
        print(
            "MODEL LOAD SMOKE OK: "
            f"x3d={type(x3d_model).__name__}, "
            f"ctrgcn={type(ctrgcn_model).__name__}, "
            f"yolo={type(yolo_model).__name__}, "
            f"clipgcn={type(clipgcn_model).__name__}, "
            f"clipgcn_checkpoint={checkpoint_path}, "
            f"classes={len(candidate_labels)}, "
            f"rtmpose={type(pose_source.body).__name__}",
            flush=True,
        )
    finally:
        if pose_source is not None:
            pose_source.close()
        if x3d_hook is not None:
            x3d_hook.remove()
        if ctrgcn_hook is not None:
            ctrgcn_hook.remove()
        del x3d_model, ctrgcn_model, yolo_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
