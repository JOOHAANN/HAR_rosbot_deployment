# RTMPose 2-D Skeleton to Body Orientation MLP

This is a lightweight body-orientation module shipped separately from the main HAR/DQN package. It maps the single-frame RTMPose-17 `(x, y, confidence)` input to eight-direction body orientation probabilities, providing direction features aligned with the 3-D skeleton yaw when only 2-D human keypoints are available.

## 1. Network and labels

The input is the 17 joints of an RTMPose frame, each with `(x, y, score)`, 51 dimensions in total. Preprocessing translates to the hip center and normalizes by the torso scale; invalid keypoints are handled as in the original training code. The network is:

```text
17 x (x, y, score) = 51
    -> Linear(51, 128) -> ReLU
    -> Linear(128, 64)  -> ReLU
    -> Linear(64, 8)
```

The output is 8 adjacent 45° circular bins. Training labels compute the anatomical body-forward yaw from the 3-D skeletons released by ETRI; the 3-D skeleton is only used to generate supervision labels and for offline evaluation. Deployment-time inference requires only RTMPose 2-D input and does not read the 3-D skeleton.

## 2. Contents

```text
RTMPose_orientation_MLP/
├── README.md
├── VPOCLIP_plus_full/rl/
│   ├── train_rtmpose_orientation_mlp.py
│   └── evaluate_rtmpose_orientation_direct.py
├── CTR-GCN_17_full/tools/extract_rtmpose_coco17.py
├── models/rtmlib/                         # RTMPose + YOLOX ONNX
├── work_dir/rtmpose_orientation_mlp/      # best.pt, 30 seeds, results
├── data/derived_orientation/              # train/val/test npz ready for direct evaluation
├── data/orientation_source_cache/         # metadata/view mask needed to rebuild labels
├── data/low_view_camera_angles_split55.csv
├── data/ETRI_subject_split_45_25_10_20.json
└── environment/                           # usable requirements/environment records
```

The original ETRI RGB, 3-D skeleton, and full raw RTMPose arrays are not packaged, to avoid bringing the dataset into the release. The derived `orientation_test_frame6.npz` already contains the test inputs and 3-D yaw labels, so the orientation error of the saved model can be re-checked without the dataset.

## 3. Unpacking and verification

```bash
tar -xzf RTMPose_orientation_MLP_strict50_5.tar.gz
cd RTMPose_orientation_MLP
sha256sum -c SHA256SUMS
```

Python 3.10, PyTorch, and NumPy are recommended. When a GPU is available, the evaluation script uses CUDA by default; otherwise pass `--device cpu` to the script.

## 4. Direct re-check of the 30 seeds

```bash
cd RTMPose_orientation_MLP
python VPOCLIP_plus_full/rl/evaluate_rtmpose_orientation_direct.py \
  --data work_dir/rtmpose_orientation_mlp/orientation_test_frame6.npz \
  --checkpoint work_dir/rtmpose_orientation_mlp/best.pt \
  --seed-root work_dir/rtmpose_orientation_mlp/seed_runs \
  --output reproduced/direct_orientation_comparison_30seed.json \
  --pairs-output reproduced/test_orientation_pairs.csv
```

The default `best.pt` is a representative checkpoint; `seed_runs/seed_*/best.pt` contains checkpoints for 30 random-initialization seeds, numbered `20260901`--`20260930`.

## 5. Saved test results

The test set is a strictly subject-disjoint 50/5 orientation-module experiment with 11,326 test samples. The values below come from `direct_orientation_comparison_30seed.json`; the 95% CI is the mean confidence interval across the 30 random seeds:

| Output mode | Circular-angle MAE | 95% CI | Within-45° ratio | 95% CI |
|---|---:|---:|---:|---:|
| soft expected yaw | 21.814° | [21.763°, 21.865°] | 88.224% | [88.166%, 88.281%] |
| argmax bin center | 26.137° | [26.055°, 26.220°] | 85.193% | [85.073%, 85.314%] |

The representative `best.pt` achieves a soft expected yaw MAE of 21.806°, RMSE of 35.282°, and 88.416% within 45°. The full p90/p95, 5°/10°/20°/30°/90° metrics and the per-sample prediction comparison are saved in:

```text
work_dir/rtmpose_orientation_mlp/direct_orientation_comparison_30seed.json
work_dir/rtmpose_orientation_mlp/test_orientation_pairs.csv
```

Here `soft expected yaw` is the continuous angle obtained by taking the circular expectation over the 8 bin probabilities, which is better suited to the active-view policy; `argmax bin center` is the discrete 45° direction center.

## 6. Rebuilding and retraining from raw data

The original training script keeps its complete data-building logic:

```bash
python VPOCLIP_plus_full/rl/train_rtmpose_orientation_mlp.py \
  --cache-root /path/to/rl_cache \
  --raw-pose-root /path/to/.rtmpose_parts \
  --angle-csv data/low_view_camera_angles_split55.csv \
  --output work_dir/rtmpose_orientation_mlp_new \
  --frame-index 6
```

`--build-only` generates only the orientation supervision cache. Rebuilding requires the original ETRI skeleton, RGB data, and raw RTMPose arrays; `extract_rtmpose_coco17.py` is the corresponding 13-frame RTMPose extraction script. The skeleton paths for the orientation labels are recorded in `low_view_camera_angles_split55.csv`; when migrating to another machine, either rewrite the root paths in the CSV to the target machine's paths or create the same directory structure outside the script.

## 7. Relationship to the main package

This archive only estimates body orientation from RTMPose 2-D skeletons; it does not include the VPOCLIP three-stream recognizer, the dual-robot DDQN, or the fusion weights of the main experiments. The main HAR/DQN reproduction package remains `HAR_reproduction_strict45_5.tar.gz`.
