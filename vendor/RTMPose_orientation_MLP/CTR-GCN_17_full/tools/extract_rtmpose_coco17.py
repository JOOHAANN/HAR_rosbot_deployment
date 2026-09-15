"""Extract resumable COCO-17 skeleton clips from ETRI RGB with RTMPose.

The sample lists come from the X3D stream so RGB and skeleton features have
identical ordering, labels, camera selection, and subject splits. Each
clip stores 13 uniformly sampled frames as [x, y, confidence], ready for the
CTR-GCN COCO-17 feeder.
"""

import argparse
import json
import os
import time
import threading
from queue import Queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np


DEFAULT_RGB_ROOT = Path("/home/youhan/ws/ETRI-Activity3D-RGB")
DEFAULT_SPLIT_DIR = Path("/home/youhan/ws/X3D_full/data/etri_rgb_c001_cs_70_10_20")
DEFAULT_OUTPUT_DIR = Path("/home/youhan/ws/CTR-GCN_17_full/data/etri_coco17")
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable RTMPose extraction for ETRI clips")
    parser.add_argument("--rgb-root", type=Path, default=DEFAULT_RGB_ROOT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=["train", "val", "test", "dqn"])
    parser.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--inference-workers", type=int, default=2,
                        help="Independent GPU sessions overlap inference and CPU postprocessing; 1 restores serial mode")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=None, help="Per split; intended for smoke tests")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow ONNX Runtime CPU fallback")
    parser.add_argument("--skip-merge", action="store_true", help="Worker mode: leave independent npy parts for coordinator")
    return parser.parse_args()


def read_uniform_frames_bgr(video_path, num_frames):
    """Decode once sequentially and retain uniformly spaced frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    wanted = np.linspace(0, total - 1, num_frames).round().astype(np.int64)
    frames, position, take = [], 0, 0
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
    frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames


def top_persons(keypoints, scores, max_persons=2):
    """Keep the two most confident bodies (ETRI actions 44-47 use two people)."""
    keypoints, scores = np.asarray(keypoints), np.asarray(scores)
    if keypoints.ndim != 3 or len(keypoints) == 0:
        return None, None
    order = np.argsort(scores.sum(axis=1))[::-1][:max_persons]
    return keypoints[order], scores[order]


def session_providers(body):
    found = []
    for model_name in ("det_model", "pose_model"):
        model = getattr(body, model_name, None)
        session = getattr(model, "session", None)
        if session is not None:
            found.append((model_name, session.get_providers()))
    return found


def load_or_create_parts(part_dir, split, count, frames, names, labels):
    shape = (count, 3, frames, 17, 2)
    x_path = part_dir / f"{split}_x.npy"
    done_path = part_dir / f"{split}_done.npy"
    meta_path = part_dir / f"{split}_meta.json"
    expected = {"shape": list(shape), "first": str(names[0]), "last": str(names[-1])}
    if meta_path.exists():
        actual = json.loads(meta_path.read_text())
        if actual != expected:
            raise RuntimeError(f"Resume metadata mismatch for {split}: {actual} != {expected}")
        clips = np.lib.format.open_memmap(x_path, mode="r+")
        done = np.lib.format.open_memmap(done_path, mode="r+")
    else:
        clips = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=shape)
        clips[:] = 0
        clips.flush()
        done = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.bool_, shape=(count,))
        done[:] = False
        done.flush()
        np.save(part_dir / f"{split}_labels.npy", labels)
        np.save(part_dir / f"{split}_sample_names.npy", names)
        meta_path.write_text(json.dumps(expected, indent=2) + "\n")
    return clips, done


def infer_video(body, video, frames):
    result = np.zeros((3, frames, 17, 2), dtype=np.float32)
    if video is None:
        return result, True
    height, width = video[0].shape[:2]
    for t, frame in enumerate(video):
        kpts, scores = top_persons(*body(frame))
        if kpts is None:
            continue
        for person in range(len(kpts)):
            result[0, t, :, person] = kpts[person, :, 0] / width * 2.0 - 1.0
            result[1, t, :, person] = kpts[person, :, 1] / height * 2.0 - 1.0
            result[2, t, :, person] = scores[person]
    return result, False


def extract_split(body, rgb_root, names, labels, frames, split, part_dir,
                  decode_workers, checkpoint_every):
    clips, done = load_or_create_parts(part_dir, split, len(names), frames, names, labels)
    pending = np.flatnonzero(~np.asarray(done)).tolist()
    print(f"[{split}] total={len(names)}, completed={int(done.sum())}, pending={len(pending)}", flush=True)
    if not pending:
        return

    start, processed, unreadable = time.time(), 0, 0
    prefetch = max(1, decode_workers * 2)
    bodies = body if isinstance(body, list) else [body]
    available_bodies = Queue()
    for instance in bodies:
        available_bodies.put(instance)
    local = threading.local()
    def initialize_worker():
        local.body = available_bodies.get_nowait()
    def infer_future(decoded):
        return infer_video(local.body, decoded.result(), frames)
    with ThreadPoolExecutor(max_workers=decode_workers) as executor, \
            ThreadPoolExecutor(max_workers=len(bodies), initializer=initialize_worker) as inference:
        queue = deque()
        iterator = iter(pending)

        def submit_one():
            try:
                index = next(iterator)
            except StopIteration:
                return False
            path = rgb_root / str(names[index])
            decoded = executor.submit(read_uniform_frames_bgr, path, frames)
            queue.append((index, inference.submit(infer_future, decoded)))
            return True

        for _ in range(min(prefetch, len(pending))):
            submit_one()

        while queue:
            index, future = queue.popleft()
            submit_one()
            result, failed = future.result()
            clips[index] = result
            unreadable += int(failed)
            done[index] = True
            processed += 1

            if processed % checkpoint_every == 0 or not queue:
                clips.flush()
                done.flush()
            if processed % 25 == 0 or not queue:
                elapsed = max(time.time() - start, 1e-6)
                rate = processed / elapsed
                remaining = len(pending) - processed
                print(
                    f"[{split}] {int(done.sum())}/{len(names)} complete; "
                    f"run={rate:.2f} clip/s, eta={remaining / rate / 60:.1f} min, "
                    f"unreadable={unreadable}", flush=True
                )


def merge_dataset(output_dir, part_dir, splits=SPLITS):
    merged = {}
    for split in splits:
        done = np.load(part_dir / f"{split}_done.npy", mmap_mode="r")
        if not bool(done.all()):
            raise RuntimeError(f"Cannot merge: {split} has {int((~done).sum())} unfinished clips")
        merged[f"x_{split}"] = np.load(part_dir / f"{split}_x.npy", mmap_mode="r")
        merged[f"y_{split}"] = np.load(part_dir / f"{split}_labels.npy")
        merged[f"{split}_sample_name"] = np.load(
            part_dir / f"{split}_sample_names.npy", allow_pickle=True
        )
    combined = output_dir / "ETRI_55_CS_rtmpose_coco17_13.npz"
    np.savez_compressed(combined, **merged)
    print(f"combined dataset: {combined}", flush=True)
    for split in splits:
        print(f"  {split}: {merged[f'x_{split}'].shape}", flush=True)


def main():
    args = parse_args()
    cv2.setNumThreads(1)

    # Import torch first so ORT can reuse the matching CUDA 12/cuDNN 9 libs.
    import torch  # noqa: F401
    import onnxruntime as ort
    from rtmlib import Body

    available = ort.get_available_providers()
    print(f"ONNX Runtime {ort.__version__}; available providers: {available}", flush=True)
    if "CUDAExecutionProvider" not in available and not args.allow_cpu:
        raise RuntimeError("CUDAExecutionProvider is unavailable")

    body = Body(mode=args.mode, backend="onnxruntime", device="cuda")
    providers = session_providers(body)
    print(f"RTMLib model providers: {providers}", flush=True)
    if not args.allow_cpu and any("CUDAExecutionProvider" not in value for _, value in providers):
        raise RuntimeError(f"RTMLib did not create CUDA sessions: {providers}")

    probe = np.zeros((480, 640, 3), dtype=np.uint8)
    body(probe)
    start = time.time()
    for _ in range(5):
        body(probe)
    per_frame = (time.time() - start) / 5
    print(f"inference probe: {per_frame * 1000:.1f} ms/frame", flush=True)
    if per_frame > 0.15 and not args.allow_cpu:
        raise RuntimeError(f"RTMPose is unexpectedly slow ({per_frame * 1000:.0f} ms/frame)")

    if not 1 <= args.inference_workers <= 4:
        raise ValueError('inference-workers must be between 1 and 4 to bound resource usage')
    bodies = [body]
    for _ in range(args.inference_workers - 1):
        instance = Body(mode=args.mode, backend="onnxruntime", device="cuda")
        if not args.allow_cpu and any('CUDAExecutionProvider' not in p for _, p in session_providers(instance)):
            raise RuntimeError('Additional inference session lacks CUDA')
        instance(probe)
        bodies.append(instance)
    print(f'inference workers={len(bodies)}, independent sessions; decode workers={args.decode_workers}', flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    part_dir = args.output_dir / ".rtmpose_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        names = np.load(args.split_dir / f"{split}_sample_names.npy", allow_pickle=True)
        labels = np.load(args.split_dir / f"{split}_labels.npy").astype(np.int64)
        if len(names) != len(labels):
            raise RuntimeError(f"{split}: names/labels length mismatch")
        if args.max_samples is not None:
            names, labels = names[:args.max_samples], labels[:args.max_samples]
        missing = [str(name) for name in names if not (args.rgb_root / str(name)).is_file()]
        if missing:
            raise FileNotFoundError(f"{split}: {len(missing)} RGB videos are missing; first={missing[0]}")
        extract_split(
            bodies, args.rgb_root, names, labels, args.frames, split, part_dir,
            max(1, args.decode_workers), max(1, args.checkpoint_every)
        )
    if not args.skip_merge:
        merge_dataset(args.output_dir, part_dir, args.splits)


if __name__ == "__main__":
    main()
