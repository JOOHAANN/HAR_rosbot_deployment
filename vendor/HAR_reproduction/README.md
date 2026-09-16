# VPOCLIP + dual-robot DDQN portable source/weights archive

This archive corresponds to `ws/VPOCLIP_plus_full/work_dir/dual_robot_depth_ddqn_strict45_5/RESULTS.md`: Full DDQN 65.33%, Random 64.56%, single view 63.64%. This is the 45/5/5 protocol and must not be read as a 50/5 result.

## Contents

- Code, configs, documentation, text experiment logs, and split lists from the five designated directories; raw videos, images, skeleton data, visual feature arrays, and dataset archives are not included.
- All policy weights under `dual_robot_depth_ddqn_strict45_5/models`, with three training seeds for each of the eight architectures; the selection results and the 30 evaluation seed records are saved alongside.
- All fusion training weights under `fusion_lightweight_gating_strict45_5`. The main table actually uses `models/seed_20260910/best.pt`.
- VPOCLIP: `work_dir/merged45_10_groupAB/selected_stage_b/last_model.pth`.
- CTR-GCN: `ws/CTR-GCN_17_full/work_dir/etri_coco17/allviews_cs45_merged_groupAB/runs-51-4029.pt`.
- X3D: `ws/X3D_full/outputs/etri_allviews_cs45_merged_groupAB_7000/model_best.pth`.
- YOLO: `ws/yolov5_full/best.pt`.
- `.cache/clip/ViT-B-32.pt`, RTMPose and YOLOX ONNX files, MiDaS small and EfficientNet weights, and the torch hub source code.
- Frozen text prototype files are retained. `MANIFEST.json` lists the original path, size, and SHA256 of each file; `SHA256SUMS` is used for post-transfer verification.
- The two HAR directory links are rewritten as in-package relative links so that source code is not stored twice.

## Environment and paths

Linux + NVIDIA GPU and Python 3.10 are recommended. The complete pip version record of the original environment is in `environment/pip-freeze.txt`. This is a snapshot of the original machine; the CUDA/ONNX Runtime runtimes must still match the target GPU and driver. Install a compatible PyTorch CUDA build first, then install the dependencies per the project requirements and the snapshot; the local editable/file URLs in the snapshot should be rewritten to the corresponding in-package source paths.

`environment/requirements-portable.txt` already converts the conda build machine's local file URLs into package names and versions, and keeps the Git commit for CLIP. The original versions can be installed with `python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r environment/requirements-portable.txt`; this step requires network access and a compatible NVIDIA driver.

```bash
tar -xzf HAR_reproduction_strict45_5.tar.gz
cd HAR_reproduction
sha256sum -c SHA256SUMS
```

The historical scripts contain absolute paths such as `/home/youhan/ws`, `/home/youhan/HAR`, and the original conda Python. The minimal-change reproduction path is to place the packaged ws/HAR at the same paths on the target Linux machine and use the `/home/youhan/.conda/envs/clipgcn` environment. If other locations are used, these paths must be replaced consistently in the scripts and YAML/JSON files with the target paths; do not modify only one launch config. Models under `.cache` should be copied to the same-named cache directory of the user running the jobs, so that RTMPose, CLIP, and torch hub can use the packaged models.

## Data preparation and cache rebuild

The dataset is not included in the package, as required. Recomputing accuracy requires preparing the original ETRI RGB, 3D Skeleton, and the same sample splits yourself; weights alone cannot yield dataset accuracy. The exact sample/window/candidate order is stored in the manifest/metadata text files under the relevant caches. Each seed in the full table uses 291 complete final-unseen samples.

Config entry: `ws/VPOCLIP_plus_full/work_dir/frame0_body_angle_rank_strict45_10/build_cache.yaml`, which contains the three-stream weights, hook layers, input sizes, the 8-second/13-frame window, YOLO thresholds, and the subject list.

Rebuild order:

1. Prepare `ws/ETRI-Activity3D-RGB` and `ws/ETRI-Activity3D-Skeleton`, and restore the splits according to the saved CSV/subject manifests.
2. Use CTR-GCN's `tools/extract_rtmpose_coco17.py` to generate `data/etri_coco17/allviews_cs45/ETRI_55_CS_rtmpose_coco17_13.npz`; check `--help` before invoking and keep the original sampling settings.
3. Run `python -m rl.frame0_body_angle_rank.prepare_strict45_10_inputs` in the VPOCLIP directory to prepare the pose cache and the manifests for each split.
4. Following the cache-building section before training in `rl/frame0_body_angle_rank/run_strict45_10.sh`, call `rl.build_cache.export_cache` to generate dqn_train/val/test, generate `body_yaw_deg.npy`, extract object tracks, and convert the six-channel tracks to `[presence,x,y,confidence]` four channels. The latter half of that script trains other historical policies and does not need to be run.
5. `python -m rl.prepare_strict45_depth` generates `data/frame0_body_angle_rank_strict45_10_depth`. If no old depth data exists, create an empty `data/new50_5_group1_zsl_depth_first_frame_fp32` directory and the script will re-extract the missing samples.

Note: the package retains reference metadata and records such as complete.json. When rebuilding, back up the reference records to another directory and use a clean output directory, so that stale completion markers do not trigger skips. Keep the manifests to verify sample order; do not assume that the existence of metadata implies the binary caches already exist.

## Direct evaluation of selected weights

After restoring the caches above, run from the package root:

```bash
python evaluate_bundle.py --root "$PWD" --output "$PWD/reproduced_results"
```

This entry point reads the eight selected checkpoints from config_resolved, runs the original random-start dual-robot evaluation, Random, two Cyclic variants, and single view, and outputs 30 seeds, means, the 95% CI under the original definition, and the numeric deltas against the saved results. It does not re-select checkpoints. The final fusion contains only the two target views, and the fusion gate is fixed. Pseudo-unseen `{1,7,14,15,18}`, final unseen `{25,39,46,52,54}`.

To train a policy from scratch, use `python -m rl.dual_robot_depth_ddqn_v1.experiment --help`, specifying the cache/track/depth/gate paths from the saved configs, the eight variants, the training seeds, and `--distance-lambda 0.25 --true-unseen-classes 1 7 14 15 18 25 39 46 52 54 --pseudo-unseen-classes 1 7 14 15 18`. Outputs should go to a new directory.

## Reproduction verification scope

The packaging program computes a SHA256 for every archived file and re-reads each file to verify it. evaluate_bundle.py was rerun on the original machine with existing caches for the full 30 seeds: the accuracy and movement-cost deltas of the eight models and three baselines against the reference results are all zero, and single view is 63.6426%. See verification/reference_delta.json, verification/single_view.json, and the per-seed records. After migration, the data/caches must be restored before repeating the same verification; different hardware and preprocessing versions may introduce numeric differences, and bit-identical results cannot be guaranteed by file packaging alone.
