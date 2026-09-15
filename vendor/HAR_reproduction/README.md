# VPOCLIP + dual-robot DDQN portable source/weights archive

本包对应 `ws/VPOCLIP_plus_full/work_dir/dual_robot_depth_ddqn_strict45_5/RESULTS.md`：Full DDQN 65.33%，Random 64.56%，单视角 63.64%。这是 45/5/5 协议，不能直接当作 50/5 的结果。

## 内容

- 五个指定目录中的代码、配置、文档、文本实验记录与划分清单；不含原始视频、图片、骨骼数据、视觉特征数组和数据集压缩包。
- `dual_robot_depth_ddqn_strict45_5/models` 下全部策略权重，包含八种结构的三个训练种子；选择结果和 30 个评测种子记录一并保存。
- `fusion_lightweight_gating_strict45_5` 下全部融合训练权重。主表实际采用 `models/seed_20260910/best.pt`。
- VPOCLIP：`work_dir/merged45_10_groupAB/selected_stage_b/last_model.pth`。
- CTR-GCN：`ws/CTR-GCN_17_full/work_dir/etri_coco17/allviews_cs45_merged_groupAB/runs-51-4029.pt`。
- X3D：`ws/X3D_full/outputs/etri_allviews_cs45_merged_groupAB_7000/model_best.pth`。
- YOLO：`ws/yolov5_full/best.pt`。
- `.cache/clip/ViT-B-32.pt`、RTMPose 与 YOLOX ONNX、MiDaS small 和 EfficientNet 权重，以及 torch hub 源码。
- 保留冻结文本原型文件。`MANIFEST.json` 列出每个文件的原路径、大小及 SHA256；`SHA256SUMS` 用于传输后校验。
- HAR 的两个目录链接改为包内相对链接，不重复存储源码。

## 环境与路径

推荐 Linux + NVIDIA GPU，Python 3.10。原环境的完整 pip 版本记录在 `environment/pip-freeze.txt`。这是原机环境快照，CUDA/ONNX Runtime 的运行库仍需与目标显卡和驱动匹配。先安装适配的 PyTorch CUDA，再按项目 requirements 和快照安装依赖；快照中的本机 editable/file URL 应改成包内对应源码路径。

`environment/requirements-portable.txt` 已将 conda 构建机的本地 file URL 转成包名与版本，并保留 CLIP 的 Git commit。可使用 `python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r environment/requirements-portable.txt` 安装原版本；这一步需要网络和兼容的 NVIDIA 驱动。

```bash
tar -xzf HAR_reproduction_strict45_5.tar.gz
cd HAR_reproduction
sha256sum -c SHA256SUMS
```

历史脚本含 `/home/youhan/ws`、`/home/youhan/HAR` 和原 conda Python 的绝对路径。最少改动的复现方式是在目标 Linux 上将包中 ws/HAR 放到相同路径，并使用 `/home/youhan/.conda/envs/clipgcn` 环境。若采用其他位置，需要把脚本与 YAML/JSON 内这些路径统一替换为目标路径；不要只修改一个启动配置。`.cache` 下模型应复制到运行用户的同名缓存目录；这样 RTMPose、CLIP、torch hub 可以使用打包的模型。

## 数据准备与缓存重建

数据集按要求不在包中。重算精度需要自行准备原 ETRI RGB、3D Skeleton 和相同样本划分；只有权重无法计算数据集准确率。精确的样本/窗口/候选顺序保存在相关 cache 下的 manifest/metadata 文本文件中。全表每个种子使用 291 个完整 final-unseen 样本。

配置入口：`ws/VPOCLIP_plus_full/work_dir/frame0_body_angle_rank_strict45_10/build_cache.yaml`，其中包含三流权重、hook 层、输入尺寸、8 秒/13 帧窗口、YOLO 阈值和 subject 列表。

重建顺序：

1. 准备 `ws/ETRI-Activity3D-RGB`、`ws/ETRI-Activity3D-Skeleton`，按保存的 CSV/subject manifest 恢复划分。
2. 使用 CTR-GCN 的 `tools/extract_rtmpose_coco17.py` 生成 `data/etri_coco17/allviews_cs45/ETRI_55_CS_rtmpose_coco17_13.npz`；调用前查看 `--help` 并保持原采样设置。
3. 在 VPOCLIP 目录运行 `python -m rl.frame0_body_angle_rank.prepare_strict45_10_inputs`，准备 pose cache 与各 split manifest。
4. 参考 `rl/frame0_body_angle_rank/run_strict45_10.sh` 中训练之前的缓存构建部分，调用 `rl.build_cache.export_cache` 生成 dqn_train/val/test，生成 `body_yaw_deg.npy`，提取物体 tracks，并将六通道 tracks 转成 `[presence,x,y,confidence]` 四通道。该脚本后半部是其他历史策略训练，不需要执行。
5. `python -m rl.prepare_strict45_depth` 生成 `data/frame0_body_angle_rank_strict45_10_depth`。若没有旧 depth 数据，创建空的 `data/new50_5_group1_zsl_depth_first_frame_fp32` 目录，脚本会对缺失样本重新提取。

注意：包保留了参考 metadata 和 complete.json 等记录。重建时应把参考记录备份到其他目录，并使用干净输出目录，避免旧完成标记触发跳过。保留 manifest 用于核对样本顺序；不要误认为存在 metadata 就已有二进制缓存。

## 直接评测已选权重

恢复上述缓存后，在包根目录执行：

```bash
python evaluate_bundle.py --root "$PWD" --output "$PWD/reproduced_results"
```

此入口读取 config_resolved 中八个已选 checkpoint，运行原始随机起点双机器人评测、Random、两种 Cyclic 和单视角，输出 30 seeds、均值、原定义 95% CI 以及相对于保存结果的数值差。不会重新选择 checkpoint。最终融合只含两个目标视角，融合 gate 固定。伪未见 `{1,7,14,15,18}`，最终未见 `{25,39,46,52,54}`。

需要从头训练策略时，使用 `python -m rl.dual_robot_depth_ddqn_v1.experiment --help`，指定保存配置中的 cache/track/depth/gate 路径、八个 variants、训练 seeds 和 `--distance-lambda 0.25 --true-unseen-classes 1 7 14 15 18 25 39 46 52 54 --pseudo-unseen-classes 1 7 14 15 18`。输出应另设新目录。

## 复现验证范围

打包程序对全部归档文件逐项计算并重新读取验证 SHA256。evaluate_bundle.py 已在原机已有缓存上重新运行完整 30 seeds：八种模型与三个基线的精度和移动代价相对参考结果差值全部为零，单视角为 63.6426%。详见 verification/reference_delta.json、verification/single_view.json 和逐种子记录。迁移后必须补齐数据/缓存再进行同样验证；不同硬件及预处理版本可能带来数值差异，不能仅凭文件打包保证逐位相同。
