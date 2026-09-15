#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/youhan/ws/CTR-GCN_17_full
python_bin=/home/youhan/.conda/envs/clipgcn/bin/python
log_dir="$repo_dir/logs"
log_file="$log_dir/rtmpose_extract.log"

mkdir -p "$log_dir"
cd "$repo_dir"

# Four decoder workers keep the GPU fed without saturating the host CPU.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
  echo "[$(date --iso-8601=seconds)] Starting/resuming ETRI RTMPose extraction"
  exec "$python_bin" -u tools/extract_rtmpose_coco17.py \
    --rgb-root /home/youhan/ws/ETRI-Activity3D-RGB \
    --split-dir /home/youhan/ws/X3D_full/data/etri_rgb_c001_cs_70_10_20 \
    --output-dir "$repo_dir/data/etri_coco17" \
    --frames 13 \
    --mode balanced \
    --decode-workers 4 \
    --checkpoint-every 25
} 2>&1 | tee -a "$log_file"
