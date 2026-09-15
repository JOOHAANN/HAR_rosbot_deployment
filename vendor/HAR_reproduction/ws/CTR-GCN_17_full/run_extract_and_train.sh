#!/bin/bash
# RTMPose COCO-17 extraction followed by CTR-GCN training, all on one V100.
# The extraction needs the isolated cuDNN 8 for onnxruntime-gpu; the training
# process is plain torch (cuDNN 9) and must NOT inherit that LD_LIBRARY_PATH.
set -e
cd /workspace/CTR-GCN_17

PY=/root/miniconda3/envs/clipgcn/bin/python
ENV=/root/miniconda3/envs/clipgcn/lib/python3.10/site-packages/nvidia
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-1}

echo "==== stage 1: RTMPose COCO-17 extraction ===="
LD_LIBRARY_PATH=ort_libs/nvidia/cudnn/lib:ort_libs/extra:$ENV/cublas/lib:$ENV/curand/lib:$ENV/cufft/lib:$ENV/cuda_runtime/lib:$ENV/cuda_nvrtc/lib \
  $PY tools/extract_rtmpose_coco17.py

echo "==== stage 2: CTR-GCN training (COCO-17) ===="
rm -rf work_dir/etri_coco17/ctrgcn_joint_coco17_13
$PY main.py --config config/etri-coco17/ctrgcn_joint_coco17_13.yaml

echo "==== pipeline done ===="
