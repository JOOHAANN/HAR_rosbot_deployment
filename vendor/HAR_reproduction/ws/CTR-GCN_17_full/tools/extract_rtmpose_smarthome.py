"""Extract COCO-17 skeletons from the Toyota Smarthome RGB videos with RTMPose.

Replaces the Kinect v2 csv skeletons: the crossed-swap experiment put the whole
12-point sensor gap on the skeleton stream, so the fix is to train CTR-GCN on
the same estimator the robot will run. 13 frames are sampled uniformly per clip
(matching the deployed window), and each keypoint is stored as (x, y, score)
with x/y normalised to [-1, 1] over the RGB image.

Torch-free on purpose: onnxruntime-gpu 1.18 needs the isolated cuDNN 8 in
ort_libs/, which must not be loaded into a torch (cuDNN 9) process.

Usage:
  cd /workspace/CTR-GCN_17
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  LD_LIBRARY_PATH=<see run_extract_and_train.sh> \
    python tools/extract_rtmpose_coco17.py
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

VPO_DATA = Path("/workspace/VPOCLIP_plus/data")
SPLIT_DIR = VPO_DATA / "smarthome_splits"
VIDEO_ROOT = VPO_DATA / "toyota_smarthome/mp4"

# Smarthome sample names are bare filenames, so the video root is prefixed here
# rather than being part of the name as in ETRI.
SPLITS = {
    "train": (SPLIT_DIR, "train"),
    "val": (SPLIT_DIR, "val"),
    "test": (SPLIT_DIR, "test"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="RTMPose COCO-17 extraction for ETRI clips.")
    parser.add_argument("--output-dir", default="data/smarthome_coco17")
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--max-samples", type=int, default=None, help="Per split, for smoke tests.")
    return parser.parse_args()


def read_uniform_frames_bgr(video_path, num_frames):
    """Decode sequentially and keep the uniformly spaced frames.

    Seeking 13 times re-decodes from the previous keyframe each time and made
    the pipeline decode-bound (0.28 clip/s); one sequential pass is ~6x faster.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    wanted = np.linspace(0, total - 1, num_frames).round().astype(np.int64)
    frames = []
    position = 0
    take = 0
    while take < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        while take < len(wanted) and wanted[take] == position:
            frames.append(frame)
            take += 1
        position += 1
    cap.release()
    if not frames:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


def top_persons(keypoints, scores, max_persons=2):
    """Keep the most confident bodies; ETRI classes 44-47 are two-person."""

    if keypoints.ndim != 3 or len(keypoints) == 0:
        return None, None
    order = np.argsort(scores.sum(axis=1))[::-1][:max_persons]
    return keypoints[order], scores[order]


def extract_split(body, items, frames, tag, decode_workers=6, max_persons=2):
    clips = np.zeros((len(items), 3, frames, 17, max_persons), dtype=np.float32)
    labels = np.zeros(len(items), dtype=np.int64)
    names = []
    misses = 0
    start = time.time()

    # Decode ahead in threads (cv2 releases the GIL) so the GPU never waits.
    executor = ThreadPoolExecutor(max_workers=decode_workers)
    futures = [
        executor.submit(read_uniform_frames_bgr, VIDEO_ROOT / name, frames)
        for name, _ in items[: decode_workers * 2]
    ]

    for index, (sample_name, label) in enumerate(items):
        labels[index] = label
        names.append(sample_name)
        ahead = index + decode_workers * 2
        if ahead < len(items):
            futures.append(
                executor.submit(read_uniform_frames_bgr, VIDEO_ROOT / items[ahead][0], frames)
            )
        video = futures[index].result()
        futures[index] = None
        if video is None:
            misses += 1
            continue

        height, width = video[0].shape[:2]
        for t, frame in enumerate(video):
            kpts, scores = top_persons(*body(frame), max_persons)
            if kpts is None:
                continue
            for m in range(len(kpts)):
                clips[index, 0, t, :, m] = kpts[m, :, 0] / width * 2.0 - 1.0
                clips[index, 1, t, :, m] = kpts[m, :, 1] / height * 2.0 - 1.0
                clips[index, 2, t, :, m] = scores[m]

        if (index + 1) % 100 == 0:
            rate = (index + 1) / (time.time() - start)
            eta = (len(items) - index - 1) / rate / 60
            print(f"  [{tag} {index + 1}/{len(items)}] {rate:.2f} clip/s, eta {eta:.0f} min", flush=True)

    executor.shutdown()
    if misses:
        print(f"  [{tag}] unreadable videos: {misses}")
    return clips, labels, np.array(names)


def main():
    args = parse_args()
    from rtmlib import Body

    body = Body(mode=args.mode, backend="onnxruntime", device="cuda")

    # onnxruntime falls back to CPU silently when a CUDA library is missing
    # (which once turned this hour-long job into a six-hour one), so probe the
    # actual speed instead of trusting the provider list.
    probe = np.zeros((480, 640, 3), dtype=np.uint8)
    body(probe)
    start = time.time()
    for _ in range(5):
        body(probe)
    per_frame = (time.time() - start) / 5
    print(f"inference probe: {per_frame * 1000:.0f} ms/frame")
    if per_frame > 0.1:
        raise RuntimeError(
            f"RTMPose runs at {per_frame * 1000:.0f} ms/frame - that is CPU speed. "
            "Check LD_LIBRARY_PATH covers cudnn8/cublas/curand/cufft/cudart."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, (split_dir, prefix) in SPLITS.items():
        part_path = output_dir / f"{split}.npz"
        if part_path.exists():
            print(f"[{split}] already extracted, skipping")
            continue

        sample_names = np.load(split_dir / f"{prefix}_sample_names.npy", allow_pickle=True)
        labels = np.load(split_dir / f"{prefix}_labels.npy")
        items = [(str(name), int(label)) for name, label in zip(sample_names, labels)]
        if args.max_samples:
            items = items[: args.max_samples]

        print(f"[{split}] {len(items)} clips from {split_dir}/{prefix}", flush=True)
        clips, label_array, names = extract_split(body, items, args.frames, split)
        np.savez_compressed(part_path, x=clips, y=label_array, sample_name=names)
        print(f"[{split}] saved {part_path}", flush=True)

    merged = {}
    for split in SPLITS:
        part = np.load(output_dir / f"{split}.npz", allow_pickle=True)
        merged[f"x_{split}"] = part["x"]
        merged[f"y_{split}"] = part["y"]
        merged[f"{split}_sample_name"] = part["sample_name"]
    combined = output_dir / "SMARTHOME_31_CS_rtmpose_coco17_13.npz"
    np.savez_compressed(combined, **merged)
    print(f"combined dataset: {combined}")
    for split in SPLITS:
        print(f"  {split}: {merged[f'x_{split}'].shape}")


if __name__ == "__main__":
    main()
