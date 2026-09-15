#!/usr/bin/env python3
"""Live adapter for the exported dual-robot active-view policy.

The checkpoint in ``har_export`` was trained with four signed-bearing view
slots, a 13-frame human/object history, and a two-stage masked Double-DQN
decision.  This module keeps that state layout and reuses the exact
``PairDuelingQNetwork`` implementation from the exported experiment.

The live ROS node can provide RGB pose observations and optional object
tracks.  A physical depth sensor is not assumed: when no depth stream is
available, the depth channels are explicitly zero-filled and reported in the
decision diagnostics.  Navigation is intentionally outside this adapter
because the export contains normalized angular costs, not a map-specific
Nav2 goal transform.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent
HAR_EXPORT_ROOT = PROJECT_ROOT / "vendor" / "HAR_reproduction"
VPOCLIP_ROOT = HAR_EXPORT_ROOT / "ws" / "VPOCLIP_plus_full"
ORIENTATION_ROOT = PROJECT_ROOT / "vendor" / "RTMPose_orientation_MLP"
POLICY_CHECKPOINT = (
    VPOCLIP_ROOT
    / "work_dir"
    / "dual_robot_depth_ddqn_strict45_5"
    / "models"
    / "full_depth19"
    / "seed_20260911"
    / "best.pt"
)
ORIENTATION_CHECKPOINT = (
    ORIENTATION_ROOT / "work_dir" / "rtmpose_orientation_mlp" / "best.pt"
)

NUM_VIEWS = 4
NUM_FRAMES = 13
NUM_OBJECTS = 50
PAIR_LIST = tuple((left, right) for left in range(NUM_VIEWS) for right in range(left + 1, NUM_VIEWS))
TORSO_IDS = np.asarray([5, 6, 11, 12], dtype=np.int64)


def _load_exported_policy_api():
    """Import the exact network class shipped with the main export."""

    root = str(VPOCLIP_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from rl.dual_robot_depth_ddqn_v1.experiment import PairDuelingQNetwork

    return PairDuelingQNetwork


def _load_orientation_api():
    """Load the orientation implementation without colliding with the main ``rl`` package."""

    path = ORIENTATION_ROOT / "VPOCLIP_plus_full" / "rl" / "train_rtmpose_orientation_mlp.py"
    spec = importlib.util.spec_from_file_location("har_export_orientation_mlp", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load orientation implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ViewObservation:
    """One live view slot in the DDQN input protocol.

    ``rtmpose`` is [13,17,3] in normalized [-1,1] x/y plus confidence.  The
    remaining fields are already in the policy's training convention.  The
    dataclass deliberately accepts missing fields so the ROS node can keep
    operating while a camera is warming up.
    """

    geometry: np.ndarray
    rtmpose: np.ndarray | None = None
    object_track: np.ndarray | None = None
    depth: np.ndarray | None = None
    valid: bool = True
    object_detections: list[list[dict[str, float]]] = field(default_factory=list)
    source: str = "live"

    def __post_init__(self) -> None:
        self.geometry = np.asarray(self.geometry, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(self.geometry))
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError("view geometry must be a finite non-zero [sin, cos] vector")
        self.geometry = self.geometry / norm
        if self.rtmpose is not None:
            self.rtmpose = np.asarray(self.rtmpose, dtype=np.float32)
            if self.rtmpose.shape != (NUM_FRAMES, 17, 3):
                raise ValueError(f"rtmpose must have shape [13,17,3], got {self.rtmpose.shape}")
        if self.object_track is not None:
            self.object_track = np.asarray(self.object_track, dtype=np.float32)
            if self.object_track.shape != (NUM_FRAMES, NUM_OBJECTS, 4):
                raise ValueError(
                    "object_track must have shape [13,50,4], "
                    f"got {self.object_track.shape}"
                )
        if self.depth is not None:
            self.depth = np.asarray(self.depth, dtype=np.float32).reshape(NUM_OBJECTS)


def geometry_from_angle(angle_deg: float) -> np.ndarray:
    """Convert a signed bearing to the exported [sin, cos] representation."""

    radians = math.radians(float(angle_deg))
    return np.asarray([math.sin(radians), math.cos(radians)], dtype=np.float32)


def _person_center_velocity(rtmpose: np.ndarray) -> np.ndarray:
    """Match ``experiment.py::person_center_velocity`` for a live pose window."""

    xy = np.asarray(rtmpose, dtype=np.float32)[..., :2]
    present = np.linalg.norm(xy, axis=-1) > 0.02
    torso = xy[:, TORSO_IDS]
    denominator = np.maximum(present[:, TORSO_IDS].sum(axis=1, keepdims=True), 1).astype(np.float32)
    center = (torso * present[:, TORSO_IDS, None]).sum(axis=1) / denominator
    center = np.nan_to_num(center, copy=False)
    velocity = np.zeros_like(center)
    velocity[1:] = center[1:] - center[:-1]
    return np.concatenate([center, velocity], axis=-1).astype(np.float32, copy=False)


def human_features_from_rtmpose(
    rtmpose: np.ndarray,
    orientation_model: torch.nn.Module | None = None,
    orientation_device: torch.device | str = "cpu",
) -> np.ndarray:
    """Build the six policy channels: center, velocity, sin(yaw), cos(yaw)."""

    rtmpose = np.asarray(rtmpose, dtype=np.float32)
    if rtmpose.shape != (NUM_FRAMES, 17, 3):
        raise ValueError(f"expected RTMPose window [13,17,3], got {rtmpose.shape}")
    base = _person_center_velocity(rtmpose)
    yaw = np.zeros(NUM_FRAMES, dtype=np.float32)
    if orientation_model is not None:
        orientation_api = _load_orientation_api()
        for frame_index, frame in enumerate(rtmpose):
            # The orientation MLP was trained on normalized [0,1] image x/y,
            # while the CTR-GCN input uses [-1,1].
            raw_frame = np.stack(
                ((frame[:, 0] + 1.0) * 0.5, (frame[:, 1] + 1.0) * 0.5, frame[:, 2]),
                axis=0,
            )
            result = orientation_api.predict_orientation(
                orientation_model, raw_frame, device=orientation_device
            )
            if result.get("valid") and result.get("yaw_deg") is not None:
                yaw[frame_index] = float(result["yaw_deg"])
    yaw_rad = np.deg2rad(yaw)
    return np.concatenate(
        [base, np.sin(yaw_rad)[:, None], np.cos(yaw_rad)[:, None]], axis=-1
    ).astype(np.float32, copy=False)


def object_track_from_detections(
    detections_by_frame: Sequence[Sequence[dict[str, Any]]],
) -> np.ndarray:
    """Pack optional detector boxes into the exported [presence,x,y,confidence] tracks."""

    result = np.zeros((NUM_FRAMES, NUM_OBJECTS, 4), dtype=np.float32)
    for frame_index, detections in enumerate(list(detections_by_frame)[-NUM_FRAMES:]):
        for object_index, detection in enumerate(list(detections)[:NUM_OBJECTS]):
            if "x1" in detection:
                x = (float(detection["x1"]) + float(detection["x2"])) * 0.5
                y = (float(detection["y1"]) + float(detection["y2"])) * 0.5
            else:
                x = float(detection.get("x", detection.get("center_x", 0.0)))
                y = float(detection.get("y", detection.get("center_y", 0.0)))
            result[frame_index, object_index] = [
                1.0,
                float(np.clip(x, -1.0, 1.0)),
                float(np.clip(y, -1.0, 1.0)),
                float(np.clip(detection.get("confidence", 1.0), 0.0, 1.0)),
            ]
    return result


def _pooled_object_features(
    human: torch.Tensor,
    object_track: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the exported 19-channel pooled object sensor."""

    person_xy = human[..., :2]
    relative_xy = object_track[..., 1:3] - person_xy[:, :, None, :]
    object_depth = depth[:, None, :].expand(-1, NUM_FRAMES, -1)
    relative_depth = object_depth - object_depth[:, :, :1]
    numeric_values = torch.cat(
        [object_track[..., 1:4], relative_xy, relative_depth[..., None]], dim=-1
    )
    presence = object_track[..., 0].clamp(0.0, 1.0)
    count = presence.sum(dim=-1, keepdim=True)
    denominator = count.clamp_min(1.0)
    weights = presence.unsqueeze(-1)
    mean = (numeric_values * weights).sum(dim=-2) / denominator
    present_mask = presence.unsqueeze(-1) > 0.0
    minimum = numeric_values.masked_fill(~present_mask, float("inf")).amin(dim=-2)
    maximum = numeric_values.masked_fill(~present_mask, float("-inf")).amax(dim=-2)
    has_object = count > 0.0
    minimum = torch.where(has_object, minimum, torch.zeros_like(minimum))
    maximum = torch.where(has_object, maximum, torch.zeros_like(maximum))
    pooled = torch.cat([count / float(NUM_OBJECTS), mean, minimum, maximum], dim=-1)
    return torch.nan_to_num(pooled)


def _relative_pair_angles(geometry: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    sin_delta = geometry[..., 0] * source[..., 1] - geometry[..., 1] * source[..., 0]
    cos_delta = (geometry * source).sum(dim=-1)
    return torch.stack([sin_delta, cos_delta], dim=-1)


def _finite(value: torch.Tensor, fill: float = 1.0) -> torch.Tensor:
    return torch.nan_to_num(value.float(), nan=fill, posinf=fill, neginf=fill)


def _assignment_matrix(
    cost: torch.Tensor,
    reachable: torch.Tensor,
    start_a: int,
    start_b: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the minimum two-robot assignment cost and feasibility."""

    pair_cost = torch.full((NUM_VIEWS, NUM_VIEWS), float("inf"), dtype=torch.float32, device=cost.device)
    pair_valid = torch.zeros((NUM_VIEWS, NUM_VIEWS), dtype=torch.bool, device=cost.device)
    for left, right in PAIR_LIST:
        reach_a_left = bool(reachable[start_a, left]) or start_a == left
        reach_a_right = bool(reachable[start_a, right]) or start_a == right
        reach_b_left = bool(reachable[start_b, left]) or start_b == left
        reach_b_right = bool(reachable[start_b, right]) or start_b == right
        cost_a_left = 0.0 if start_a == left else float(_finite(cost[start_a, left]).item())
        cost_a_right = 0.0 if start_a == right else float(_finite(cost[start_a, right]).item())
        cost_b_left = 0.0 if start_b == left else float(_finite(cost[start_b, left]).item())
        cost_b_right = 0.0 if start_b == right else float(_finite(cost[start_b, right]).item())
        first = reach_a_left and reach_b_right
        second = reach_a_right and reach_b_left
        candidates = []
        if first:
            candidates.append(cost_a_left + cost_b_right)
        if second:
            candidates.append(cost_a_right + cost_b_left)
        if candidates:
            value = min(candidates)
            pair_cost[left, right] = value
            pair_cost[right, left] = value
            pair_valid[left, right] = True
            pair_valid[right, left] = True
    return pair_cost, pair_valid


class LiveViewPolicy:
    """Load and run the exported two-stage masked DDQN on live view slots."""

    def __init__(
        self,
        checkpoint: Path = POLICY_CHECKPOINT,
        orientation_checkpoint: Path = ORIENTATION_CHECKPOINT,
        variant: str = "full_depth19",
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.variant = variant
        pair_network = _load_exported_policy_api()
        self.model = pair_network(variant).to(self.device)
        payload = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["online"], strict=True)
        self.model.eval()

        orientation_api = _load_orientation_api()
        self.orientation_model = orientation_api.OrientationMLP().to(self.device)
        orientation_payload = torch.load(
            Path(orientation_checkpoint), map_location=self.device, weights_only=False
        )
        self.orientation_model.load_state_dict(orientation_payload["model"], strict=True)
        self.orientation_model.eval()
        self.checkpoint = str(Path(checkpoint).resolve())
        self.orientation_checkpoint = str(Path(orientation_checkpoint).resolve())

    def _state(
        self,
        views: Sequence[ViewObservation],
        valid: torch.Tensor,
        cost: torch.Tensor,
        reachable: torch.Tensor,
        start_a: int,
        start_b: int,
        selected: int,
    ) -> dict[str, torch.Tensor]:
        if len(views) != NUM_VIEWS:
            raise ValueError(f"the exported policy requires {NUM_VIEWS} view slots")
        geometry = torch.as_tensor(
            np.stack([view.geometry for view in views]), dtype=torch.float32, device=self.device
        )[None]
        human_np = views[start_a].rtmpose
        if human_np is None:
            human_np = np.zeros((NUM_FRAMES, 17, 3), dtype=np.float32)
        human = torch.from_numpy(
            human_features_from_rtmpose(human_np, self.orientation_model, self.device)
        ).to(self.device)[None]
        track_np = views[start_a].object_track
        if track_np is None:
            track_np = np.zeros((NUM_FRAMES, NUM_OBJECTS, 4), dtype=np.float32)
        depth_np = views[start_a].depth
        if depth_np is None:
            depth_np = np.zeros(NUM_OBJECTS, dtype=np.float32)
        object_track = torch.from_numpy(track_np).to(self.device)[None]
        depth = torch.from_numpy(np.asarray(depth_np, dtype=np.float32)).to(self.device)[None]
        object_features = _pooled_object_features(human, object_track, depth)

        source_a = geometry[0, start_a]
        source_b = geometry[0, start_b]
        relative_a = _relative_pair_angles(geometry, source_a[None, None, :])[0]
        relative_b = _relative_pair_angles(geometry, source_b[None, None, :])[0]
        pair_cost, pair_feasible = _assignment_matrix(cost, reachable, start_a, start_b)
        selected_present = selected >= 0
        safe_selected = max(selected, 0)
        selected_geometry = geometry[0, safe_selected]
        selected_relative = _relative_pair_angles(
            geometry, selected_geometry[None, None, :]
        )[0]
        if not selected_present:
            selected_relative.zero_()
        row_pair_cost = pair_cost[safe_selected]
        row_pair_feasible = pair_feasible[safe_selected]
        diagonal = torch.eye(NUM_VIEWS, dtype=torch.bool, device=self.device)
        best_partner_cost = pair_cost.masked_fill(~pair_feasible | diagonal, float("inf")).amin(dim=-1)
        best_partner_feasible = (pair_feasible & ~diagonal).any(dim=-1)
        pair_cost_feature = row_pair_cost if selected_present else best_partner_cost
        pair_feasible_feature = row_pair_feasible if selected_present else best_partner_feasible
        pair_cost_feature = _finite(pair_cost_feature).clamp(0.0, 2.0)

        cost_a = _finite(cost[start_a]).clamp(0.0, 1.0)
        cost_b = _finite(cost[start_b]).clamp(0.0, 1.0)
        cost_a[start_a] = 0.0
        cost_b[start_b] = 0.0
        reach_a = reachable[start_a].clone()
        reach_b = reachable[start_b].clone()
        reach_a[start_a] = True
        reach_b[start_b] = True
        selected_flag = F.one_hot(
            torch.as_tensor(safe_selected, dtype=torch.long, device=self.device), NUM_VIEWS
        ).float()
        if not selected_present:
            selected_flag.zero_()
        candidate = torch.cat(
            [
                geometry[0],
                relative_a,
                relative_b,
                cost_a[:, None],
                cost_b[:, None],
                selected_relative,
                pair_cost_feature[:, None],
                pair_feasible_feature.float()[:, None],
                selected_flag[:, None],
            ],
            dim=-1,
        )[None]
        if candidate.shape[-1] != 13:
            raise AssertionError(f"exported candidate width is 13, got {candidate.shape}")
        action_mask = valid.clone()
        if selected_present:
            action_mask[safe_selected] = False
            action_mask &= row_pair_feasible
        else:
            action_mask &= best_partner_feasible
        return {
            "human": human,
            "object": object_features,
            "starts": torch.cat([source_a, source_b])[None],
            "selection": torch.cat(
                [
                    selected_flag,
                    F.one_hot(
                        torch.as_tensor(int(selected_present), dtype=torch.long, device=self.device), 2
                    ).float(),
                ]
            )[None],
            "candidate": candidate,
            "mask": action_mask[None],
        }

    @staticmethod
    def _masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> int | None:
        if not bool(mask.any().item()):
            return None
        masked = q_values.masked_fill(~mask, float("-inf"))
        return int(masked.argmax(dim=-1).item())

    @torch.inference_mode()
    def select(
        self,
        views: Sequence[ViewObservation],
        *,
        start_a: int = 0,
        start_b: int = 1,
        angles_deg: Sequence[float] | None = None,
        cost: np.ndarray | None = None,
        reachable: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Select two distinct target slots and return a JSON-safe diagnostic."""

        if len(views) != NUM_VIEWS:
            raise ValueError(f"views must contain exactly {NUM_VIEWS} entries")
        if start_a == start_b or not (0 <= start_a < NUM_VIEWS and 0 <= start_b < NUM_VIEWS):
            raise ValueError("start_a and start_b must be two different slots in 0..3")
        valid_np = np.asarray([bool(view.valid) for view in views], dtype=bool)
        if int(valid_np.sum()) < 2:
            return None
        if angles_deg is not None:
            if len(angles_deg) != NUM_VIEWS:
                raise ValueError("angles_deg must contain four bearings")
            for view, angle in zip(views, angles_deg):
                view.geometry = geometry_from_angle(float(angle))
        geometry = np.stack([view.geometry for view in views]).astype(np.float32)
        if cost is None:
            # The export's movement cost is a normalized angular cost.  This is
            # a documented fallback until a map-specific planner supplies costs.
            dots = np.clip(geometry @ geometry.T, -1.0, 1.0)
            cost_np = np.arccos(dots) / math.pi
        else:
            cost_np = np.asarray(cost, dtype=np.float32)
            if cost_np.shape != (NUM_VIEWS, NUM_VIEWS):
                raise ValueError("cost must have shape [4,4]")
        cost_tensor = torch.as_tensor(cost_np, dtype=torch.float32, device=self.device)
        if reachable is None:
            reachable_np = valid_np[:, None].repeat(NUM_VIEWS, axis=1)
        else:
            reachable_np = np.asarray(reachable, dtype=bool)
            if reachable_np.shape != (NUM_VIEWS, NUM_VIEWS):
                raise ValueError("reachable must have shape [4,4]")
        reachable_tensor = torch.as_tensor(reachable_np, dtype=torch.bool, device=self.device)
        valid_tensor = torch.as_tensor(valid_np, dtype=torch.bool, device=self.device)

        state0 = self._state(
            views, valid_tensor, cost_tensor, reachable_tensor, start_a, start_b, -1
        )
        target_a = self._masked_argmax(self.model(state0), state0["mask"])
        if target_a is None:
            return None
        state1 = self._state(
            views, valid_tensor, cost_tensor, reachable_tensor, start_a, start_b, target_a
        )
        target_b = self._masked_argmax(self.model(state1), state1["mask"])
        if target_b is None:
            return None

        pair_cost, pair_feasible = _assignment_matrix(cost_tensor, reachable_tensor, start_a, start_b)
        if not bool(pair_feasible[target_a, target_b].item()):
            return None
        direct_a = 0.0 if target_a == start_a else float(cost_np[start_a, target_a])
        direct_b = 0.0 if target_b == start_b else float(cost_np[start_b, target_b])
        swapped_a = 0.0 if target_b == start_a else float(cost_np[start_a, target_b])
        swapped_b = 0.0 if target_a == start_b else float(cost_np[start_b, target_a])
        if direct_a + direct_b <= swapped_a + swapped_b:
            assignment = {"robot_a": target_a, "robot_b": target_b}
            movement_cost = direct_a + direct_b
        else:
            assignment = {"robot_a": target_b, "robot_b": target_a}
            movement_cost = swapped_a + swapped_b
        return {
            "variant": self.variant,
            "targets": [target_a, target_b],
            "assignment": assignment,
            "start_slots": {"robot_a": start_a, "robot_b": start_b},
            "movement_cost_normalized": float(movement_cost),
            "pair_cost_normalized": float(pair_cost[target_a, target_b].item()),
            "valid_views": [int(value) for value in valid_np.tolist()],
            "geometry_sin_cos": geometry.tolist(),
            "depth_input": "zero_fallback" if all(view.depth is None for view in views) else "provided",
            "policy_checkpoint": self.checkpoint,
            "orientation_checkpoint": self.orientation_checkpoint,
        }


def _synthetic_views(policy: LiveViewPolicy) -> list[ViewObservation]:
    """Deterministic smoke input that exercises both policy stages."""

    views = []
    for slot, angle in enumerate((-90.0, 0.0, 90.0, 180.0)):
        pose = np.zeros((NUM_FRAMES, 17, 3), dtype=np.float32)
        pose[:, :, 2] = 0.9
        for joint in range(17):
            pose[:, joint, 0] = -0.15 + 0.02 * joint + 0.01 * slot
            pose[:, joint, 1] = -0.35 + 0.035 * joint
        pose[:, :, 0] += np.linspace(0.0, 0.04, NUM_FRAMES)[:, None]
        human = human_features_from_rtmpose(pose, policy.orientation_model, policy.device)
        views.append(
            ViewObservation(
                geometry=geometry_from_angle(angle),
                rtmpose=pose,
                object_track=np.zeros((NUM_FRAMES, NUM_OBJECTS, 4), dtype=np.float32),
                depth=np.zeros(NUM_OBJECTS, dtype=np.float32),
                source=f"synthetic_slot_{slot}",
            )
        )
        _ = human
    return views


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-checkpoint", type=Path, default=POLICY_CHECKPOINT)
    parser.add_argument("--orientation-checkpoint", type=Path, default=ORIENTATION_CHECKPOINT)
    parser.add_argument("--variant", default="full_depth19")
    parser.add_argument("--device", default="cpu", help="Policy device: cpu, cuda, or cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Run a deterministic two-stage policy smoke test")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    policy = LiveViewPolicy(
        checkpoint=args.policy_checkpoint,
        orientation_checkpoint=args.orientation_checkpoint,
        variant=args.variant,
        device=args.device,
    )
    if args.smoke:
        result = policy.select(_synthetic_views(policy))
        if result is None:
            raise RuntimeError("policy smoke test produced no feasible two-view decision")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(
            json.dumps(
                {
                    "policy_checkpoint": policy.checkpoint,
                    "orientation_checkpoint": policy.orientation_checkpoint,
                    "variant": policy.variant,
                    "status": "loaded",
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
