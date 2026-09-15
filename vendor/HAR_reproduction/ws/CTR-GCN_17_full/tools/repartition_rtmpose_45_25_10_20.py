#!/usr/bin/env python3
"""Reindex the completed RTMPose NPZ with the canonical subject split."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


SUBJECT_RE = re.compile(r"P\d{3}")


def subject_of(name):
    match = SUBJECT_RE.search(Path(str(name)).name)
    if not match:
        raise ValueError(f"No Pxxx subject id in {name}")
    return match.group(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    groups = {key: set(value) for key, value in manifest["subject_groups"].items()}
    with np.load(args.input, allow_pickle=True) as source:
        train_names = source["train_sample_name"]
        train_subjects = np.asarray([subject_of(name) for name in train_names])
        vpo_mask = np.isin(train_subjects, sorted(groups["vpo_train"]))
        dqn_mask = np.isin(train_subjects, sorted(groups["dqn_train"]))
        if np.any(vpo_mask & dqn_mask) or not np.all(vpo_mask | dqn_mask):
            raise RuntimeError("Original train samples do not partition cleanly into VPO and DQN")

        arrays = {
            "x_train": source["x_train"][vpo_mask],
            "y_train": source["y_train"][vpo_mask],
            "train_sample_name": train_names[vpo_mask],
            "x_dqn": source["x_train"][dqn_mask],
            "y_dqn": source["y_train"][dqn_mask],
            "dqn_sample_name": train_names[dqn_mask],
        }
        for split in ("val", "test"):
            names = source[f"{split}_sample_name"]
            actual_subjects = {subject_of(name) for name in names}
            if actual_subjects != groups[split]:
                raise RuntimeError(
                    f"{split} subject mismatch: expected={sorted(groups[split])}, "
                    f"actual={sorted(actual_subjects)}"
                )
            arrays[f"x_{split}"] = source[f"x_{split}"]
            arrays[f"y_{split}"] = source[f"y_{split}"]
            arrays[f"{split}_sample_name"] = names

        if len(arrays["x_train"]) != manifest["sample_counts"]["vpo_train"]:
            raise RuntimeError("VPO sample count disagrees with canonical manifest")
        if len(arrays["x_dqn"]) != manifest["sample_counts"]["dqn_train"]:
            raise RuntimeError("DQN sample count disagrees with canonical manifest")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)

    summary = {
        "source": str(args.input.resolve()),
        "subject_manifest": str(args.manifest.resolve()),
        "output": str(args.output.resolve()),
        "shapes": {key: list(value.shape) for key, value in arrays.items() if key.startswith("x_")},
        "zsl_unseen_labels_0_based": manifest["zsl_unseen_labels_0_based"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
