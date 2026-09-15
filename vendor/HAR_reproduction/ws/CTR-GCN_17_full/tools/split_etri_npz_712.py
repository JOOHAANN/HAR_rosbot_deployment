#!/usr/bin/env python
"""Re-split an ETRI CTR-GCN npz file into train/val/test sets.

The source ETRI npz in this repository stores only train/test arrays. This
script merges those arrays and writes a new npz with x_train/y_train,
x_val/y_val, and x_test/y_test using a deterministic 7:1:2 split by default.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_ratio(value):
    parts = [float(part) for part in value.replace(":", ",").split(",")]
    if len(parts) != 3 or any(part < 0 for part in parts) or sum(parts) <= 0:
        raise argparse.ArgumentTypeError("ratio must contain three non-negative values, e.g. 7,1,2")
    total = sum(parts)
    return tuple(part / total for part in parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a 7:1:2 train/val/test ETRI npz split.")
    parser.add_argument(
        "--input",
        default="data/etri/ETRI_P1_P230_CS_raw_uniform13.npz",
        help="Input npz path. Relative paths are resolved from the CTR-GCN root.",
    )
    parser.add_argument(
        "--output",
        default="data/etri/ETRI_P1_P230_CS_raw_uniform13_712.npz",
        help="Output npz path. Relative paths are resolved from the CTR-GCN root.",
    )
    parser.add_argument("--ratio", default="7,1,2", type=parse_ratio, help="Split ratio, default: 7,1,2.")
    parser.add_argument("--seed", type=int, default=20260614, help="Random seed for deterministic splitting.")
    parser.add_argument(
        "--split-unit",
        choices=["subject", "sample"],
        default="subject",
        help="Use subject-level split for cross-subject data, or sample-level stratified split.",
    )
    parser.add_argument("--no-compress", action="store_true", help="Use np.savez instead of np.savez_compressed.")
    return parser.parse_args()


def resolve_path(path, repo_root):
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def combine(npz, base_key):
    train_key = f"train_{base_key}"
    test_key = f"test_{base_key}"
    if train_key not in npz.files or test_key not in npz.files:
        return None
    return np.concatenate([npz[train_key], npz[test_key]], axis=0)


def split_counts(total, ratio):
    train_count = int(round(total * ratio[0]))
    val_count = int(round(total * ratio[1]))
    test_count = total - train_count - val_count
    if total >= 3:
        val_count = max(1, val_count)
        test_count = max(1, test_count)
        train_count = total - val_count - test_count
    if train_count < 0:
        train_count = max(0, total - val_count - test_count)
    return train_count, val_count, total - train_count - val_count


def split_by_subject(subjects, ratio, seed):
    rng = np.random.default_rng(seed)
    unique_subjects = np.array(sorted(np.unique(subjects).tolist()))
    order = rng.permutation(len(unique_subjects))
    unique_subjects = unique_subjects[order]

    num_train, num_val, _num_test = split_counts(len(unique_subjects), ratio)
    train_subjects = set(unique_subjects[:num_train].tolist())
    val_subjects = set(unique_subjects[num_train:num_train + num_val].tolist())
    test_subjects = set(unique_subjects[num_train + num_val:].tolist())

    indices = {"train": [], "val": [], "test": []}
    for idx, subject in enumerate(subjects):
        if subject in train_subjects:
            indices["train"].append(idx)
        elif subject in val_subjects:
            indices["val"].append(idx)
        elif subject in test_subjects:
            indices["test"].append(idx)
        else:
            raise RuntimeError(f"Subject {subject} was not assigned to a split")

    split_subjects = {
        "train": [int(subject) for subject in sorted(train_subjects)],
        "val": [int(subject) for subject in sorted(val_subjects)],
        "test": [int(subject) for subject in sorted(test_subjects)],
    }
    return {key: np.asarray(value, dtype=np.int64) for key, value in indices.items()}, split_subjects


def split_by_sample(labels, ratio, seed):
    rng = np.random.default_rng(seed)
    class_ids = labels.argmax(axis=1) if labels.ndim == 2 else labels.astype(np.int64)
    indices = {"train": [], "val": [], "test": []}

    for class_id in sorted(np.unique(class_ids).tolist()):
        class_indices = np.flatnonzero(class_ids == class_id)
        class_indices = class_indices[rng.permutation(len(class_indices))]
        num_train, num_val, _num_test = split_counts(len(class_indices), ratio)
        indices["train"].extend(class_indices[:num_train].tolist())
        indices["val"].extend(class_indices[num_train:num_train + num_val].tolist())
        indices["test"].extend(class_indices[num_train + num_val:].tolist())

    for split in indices:
        indices[split] = np.asarray(sorted(indices[split]), dtype=np.int64)
    return indices, None


def add_split(payload, split, indices, arrays):
    payload[f"x_{split}"] = arrays["x"][indices]
    payload[f"y_{split}"] = arrays["y"][indices]
    for key in ("sample_name", "subject", "action", "frames"):
        value = arrays.get(key)
        if value is not None:
            payload[f"{split}_{key}"] = value[indices]


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_path = resolve_path(args.input, repo_root)
    output_path = resolve_path(args.output, repo_root)

    npz = np.load(input_path, allow_pickle=True)
    arrays = {
        "x": np.concatenate([npz["x_train"], npz["x_test"]], axis=0),
        "y": np.concatenate([npz["y_train"], npz["y_test"]], axis=0),
        "sample_name": combine(npz, "sample_name"),
        "subject": combine(npz, "subject"),
        "action": combine(npz, "action"),
        "frames": combine(npz, "frames"),
    }

    if args.split_unit == "subject":
        if arrays["subject"] is None:
            raise KeyError("subject split requested, but train_subject/test_subject were not found in the npz")
        split_indices, split_subjects = split_by_subject(arrays["subject"], args.ratio, args.seed)
    else:
        split_indices, split_subjects = split_by_sample(arrays["y"], args.ratio, args.seed)

    payload = {}
    for split in ("train", "val", "test"):
        add_split(payload, split, split_indices[split], arrays)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez if args.no_compress else np.savez_compressed
    save(output_path, **payload)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "normalized_ratio": list(args.ratio),
        "seed": args.seed,
        "split_unit": args.split_unit,
        "source_shape": list(arrays["x"].shape),
        "split_shapes": {split: list(payload[f"x_{split}"].shape) for split in ("train", "val", "test")},
        "split_counts": {split: int(len(split_indices[split])) for split in ("train", "val", "test")},
        "split_subjects": split_subjects,
    }
    summary_path = output_path.with_suffix("").as_posix() + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved {output_path}")
    print(json.dumps(summary["split_shapes"], indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
