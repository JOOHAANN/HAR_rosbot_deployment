# RTMPose 2-D Skeleton to Body Orientation MLP

这是与主 HAR/DQN 包分开的轻量人体朝向模块。它把单帧 RTMPose-17 的
`(x, y, confidence)` 输入映射为人体的八方向朝向概率，用于在只有 2-D
人体关键点时提供与 3-D skeleton yaw 对齐的方向特征。

## 1. 网络和标签

输入是一个 RTMPose 帧的 17 个关节点，每个关节点包含 `(x, y, score)`，共
51 维。预处理以髋部中心平移，并用躯干尺度归一化；无效关键点按原训练代码
处理。网络结构为：

```text
17 x (x, y, score) = 51
    -> Linear(51, 128) -> ReLU
    -> Linear(128, 64)  -> ReLU
    -> Linear(64, 8)
```

输出是 8 个相邻的 45° circular bins。训练标签由 ETRI 发布的 3-D skeleton
计算人体 anatomical body-forward yaw；3-D skeleton 只用于生成监督标签和离线
评测，部署推理阶段只需要 RTMPose 2-D 输入，不读取 3-D skeleton。

## 2. 包含内容

```text
RTMPose_orientation_MLP/
├── README.md
├── VPOCLIP_plus_full/rl/
│   ├── train_rtmpose_orientation_mlp.py
│   └── evaluate_rtmpose_orientation_direct.py
├── CTR-GCN_17_full/tools/extract_rtmpose_coco17.py
├── models/rtmlib/                         # RTMPose + YOLOX ONNX
├── work_dir/rtmpose_orientation_mlp/      # best.pt、30 seeds、结果
├── data/derived_orientation/              # 可直接评测的 train/val/test npz
├── data/orientation_source_cache/         # 重建标签所需的 metadata/view mask
├── data/low_view_camera_angles_split55.csv
├── data/ETRI_subject_split_45_25_10_20.json
└── environment/                           # 可用的 requirements/环境记录
```

原始 ETRI RGB、3-D skeleton 和完整 RTMPose 原始数组没有打包，以避免把数据集
带入发布包。派生的 `orientation_test_frame6.npz` 已经包含测试输入和 3-D yaw
标签，因此无需数据集即可复核已保存模型的 orientation 误差。

## 3. 解包和校验

```bash
tar -xzf RTMPose_orientation_MLP_strict50_5.tar.gz
cd RTMPose_orientation_MLP
sha256sum -c SHA256SUMS
```

建议使用 Python 3.10、PyTorch 和 NumPy。GPU 可用时评测脚本默认使用 CUDA；
没有 GPU 时将 `--device cpu` 传给脚本即可。

## 4. 直接复核 30 个种子

```bash
cd RTMPose_orientation_MLP
python VPOCLIP_plus_full/rl/evaluate_rtmpose_orientation_direct.py \
  --data work_dir/rtmpose_orientation_mlp/orientation_test_frame6.npz \
  --checkpoint work_dir/rtmpose_orientation_mlp/best.pt \
  --seed-root work_dir/rtmpose_orientation_mlp/seed_runs \
  --output reproduced/direct_orientation_comparison_30seed.json \
  --pairs-output reproduced/test_orientation_pairs.csv
```

默认的 `best.pt` 是代表性 checkpoint；`seed_runs/seed_*/best.pt` 包含 30 个
随机初始化种子的 checkpoint，编号为 `20260901`--`20260930`。

## 5. 已保存测试结果

测试集为严格 subject-disjoint 的 50/5 方向模块实验，测试样本数为 11,326。
以下数值来自 `direct_orientation_comparison_30seed.json`，95% CI 是跨 30 个
随机种子的均值置信区间：

| 输出方式 | 圆周角度 MAE | 95% CI | 45° 内比例 | 95% CI |
|---|---:|---:|---:|---:|
| soft expected yaw | 21.814° | [21.763°, 21.865°] | 88.224% | [88.166%, 88.281%] |
| argmax bin center | 26.137° | [26.055°, 26.220°] | 85.193% | [85.073%, 85.314%] |

代表性 `best.pt` 的 soft expected yaw 结果为 MAE 21.806°、RMSE 35.282°、
45° 内 88.416%。完整的 p90、p95、5°/10°/20°/30°/90° 指标以及逐样本
预测对照分别保存在：

```text
work_dir/rtmpose_orientation_mlp/direct_orientation_comparison_30seed.json
work_dir/rtmpose_orientation_mlp/test_orientation_pairs.csv
```

其中 `soft expected yaw` 是根据 8 个 bin 的概率在圆周上求期望得到的连续角度，
更适合给主动视角策略使用；`argmax bin center` 是离散 45° 方向中心。

## 6. 从原始数据重建和重新训练

原始训练脚本仍保留其完整的数据构建逻辑：

```bash
python VPOCLIP_plus_full/rl/train_rtmpose_orientation_mlp.py \
  --cache-root /path/to/rl_cache \
  --raw-pose-root /path/to/.rtmpose_parts \
  --angle-csv data/low_view_camera_angles_split55.csv \
  --output work_dir/rtmpose_orientation_mlp_new \
  --frame-index 6
```

`--build-only` 可只生成方向监督缓存。重建需要原始 ETRI skeleton、RGB 数据和
RTMPose 原始数组；`extract_rtmpose_coco17.py` 是对应的 13 帧 RTMPose 提取脚本。
方向标签的 skeleton 路径记录在 `low_view_camera_angles_split55.csv` 中，迁移到
另一台机器时需要按目标机器路径修改 CSV 中的根路径，或者在脚本外建立相同目录
结构。

## 7. 与主包的关系

这个压缩包只负责从 RTMPose 2-D skeleton 估计人体朝向，不包含 VPOCLIP 三流
recognizer、双机器人 DDQN 或主实验的融合权重。主 HAR/DQN 复现包仍是
`HAR_reproduction_strict45_5.tar.gz`。
