#!/usr/bin/env python3
"""ROS 2 live adapter for the exported four-view view-selection policies.

The node subscribes to the two physical RGB camera topics, routes each camera
to its currently occupied ring slot, and publishes a view decision as JSON on
``/har/selected_view_pair``. ``rl`` uses the exported two-stage DDQN;
``random`` and ``cyclic`` reproduce the archive's random-pair and cyclic-pair
baselines. The node deliberately does not send Nav2 goals: the orbit
coordinator owns map/TF conversion and physical motion.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from ros_realtime import decode_compressed_image, decode_raw_image, qos_profile
from viewpoint_policy import (
    HAR_EXPORT_ROOT,
    NUM_FRAMES,
    NUM_VIEWS,
    LiveViewPolicy,
    ViewObservation,
    geometry_from_angle,
)


DEFAULT_TOPICS = [
    "/rosbot_1/camera/rgb/image_raw/compressed",
    "/rosbot_2/camera/rgb/image_raw/compressed",
    "/har/view/slot_2/compressed",
    "/har/view/slot_3/compressed",
]
DEFAULT_ANGLES_DEG = [-90.0, 0.0, 90.0, 180.0]
DEFAULT_RTMPOSE_ROOT = HAR_EXPORT_ROOT / ".cache" / "rtmlib" / "hub" / "checkpoints"
DEFAULT_RTMPOSE_DET = DEFAULT_RTMPOSE_ROOT / "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
DEFAULT_RTMPOSE_POSE = (
    DEFAULT_RTMPOSE_ROOT
    / "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx"
)
DEFAULT_ROBOT_NAMESPACES = ["rosbot_1", "rosbot_2"]
DEFAULT_ROBOT_TOPICS = [
    "/rosbot_1/camera/rgb/image_raw/compressed",
    "/rosbot_2/camera/rgb/image_raw/compressed",
]
PAIR_LIST = tuple((left, right) for left in range(NUM_VIEWS) for right in range(left + 1, NUM_VIEWS))


def parse_csv_topics(value: str) -> list[str]:
    topics = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(topics) != NUM_VIEWS:
        raise ValueError(f"--view-topics must contain exactly {NUM_VIEWS} comma-separated topics")
    if any(not topic.startswith("/") for topic in topics):
        raise ValueError("every view topic must be an absolute ROS topic name")
    return topics


def parse_csv_floats(value: str) -> list[float]:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(values) != NUM_VIEWS:
        raise ValueError(f"--view-angles-deg must contain exactly {NUM_VIEWS} values")
    return [float(item) for item in values]


def parse_csv_robot_namespaces(value: str) -> list[str]:
    namespaces = [item.strip().strip("/") for item in str(value).split(",") if item.strip()]
    if len(namespaces) != 2:
        raise ValueError("--robot-namespaces must contain exactly two namespaces")
    if any(not item or "/" in item for item in namespaces):
        raise ValueError("robot namespaces must be simple names such as rosbot_1")
    return namespaces


def parse_csv_robot_topics(value: str) -> list[str]:
    topics = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(topics) != 2:
        raise ValueError("--robot-topics must contain exactly two comma-separated topics")
    if any(not topic.startswith("/") for topic in topics):
        raise ValueError("every robot topic must be an absolute ROS topic name")
    return topics


def normalized_rtmpose(keypoints, scores, frame_shape) -> np.ndarray:
    """Keep the most confident person and return one [17,3] [-1,1] frame."""

    height, width = frame_shape[:2]
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)
    output = np.zeros((17, 3), dtype=np.float32)
    if keypoints.ndim != 3 or scores.ndim != 2 or len(keypoints) == 0:
        return output
    if keypoints.shape[1:] != (17, 2) or scores.shape[1:] != (17,):
        return output
    person_index = int(np.argmax(scores.sum(axis=1)))
    output[:, 0] = keypoints[person_index, :, 0] / max(width, 1) * 2.0 - 1.0
    output[:, 1] = keypoints[person_index, :, 1] / max(height, 1) * 2.0 - 1.0
    output[:, 2] = np.clip(scores[person_index], 0.0, 1.0)
    output[~np.isfinite(output)] = 0.0
    return output


class ActiveViewNode(Node):
    def __init__(self, args, topics: list[str], angles_deg: list[float]) -> None:
        super().__init__(args.ros_node_name)
        self.args = args
        self.selection_mode = args.selection_mode
        self.topics = topics
        self.angles_deg = angles_deg
        self.geometry = [geometry_from_angle(value) for value in angles_deg]
        self.robot_namespaces = list(args.robot_namespaces)
        self.robot_topics = list(args.robot_topics)
        self.robot_slots = {
            self.robot_namespaces[0]: int(args.start_a),
            self.robot_namespaces[1]: int(args.start_b),
        }
        self.start_a = int(args.start_a)
        self.start_b = int(args.start_b)
        self.latest_frames: list[np.ndarray | None] = [None] * NUM_VIEWS
        self.latest_sequence = [0] * NUM_VIEWS
        self.processed_sequence = [0] * NUM_VIEWS
        self.last_robot_frame_time: dict[str, float | None] = {
            robot: None for robot in self.robot_namespaces
        }
        self.pose_history = [deque(maxlen=NUM_FRAMES) for _ in range(NUM_VIEWS)]
        self.last_result = None
        self.last_warning = 0.0
        self._baseline_rng = np.random.default_rng(args.random_seed)
        self._baseline_step = 0
        self._baseline_result = None
        self._baseline_next_time = 0.0
        self._selection_requested = not args.no_initial_selection
        self._trigger_payload = None
        self._last_slot_mapping = dict(self.robot_slots)

        self.policy = None
        self.body = None
        if self.selection_mode == "rl":
            self.policy = LiveViewPolicy(
                checkpoint=args.policy_checkpoint,
                orientation_checkpoint=args.orientation_checkpoint,
                variant=args.variant,
                device=args.policy_device,
            )
            try:
                from rtmlib import Body
            except ImportError as exc:
                raise RuntimeError("RTMPose ROS view selection needs rtmlib in the container") from exc
            detector_size = {
                "lightweight": (416, 416),
                "balanced": (640, 640),
                "performance": (640, 640),
            }[args.rtmpose_mode]
            pose_size = {
                "lightweight": (192, 256),
                "balanced": (192, 256),
                "performance": (288, 384),
            }[args.rtmpose_mode]
            self.body = Body(
                det=str(args.rtmpose_det),
                det_input_size=detector_size,
                pose=str(args.rtmpose_pose),
                pose_input_size=pose_size,
                mode=args.rtmpose_mode,
                backend=args.rtmpose_backend,
                device=args.rtmpose_device,
            )

        message_type = CompressedImage if args.image_transport == "compressed" else Image
        profile = qos_profile(args.ros_qos_reliability)
        self._subscriptions = [
            self.create_subscription(
                message_type,
                topic,
                self._make_robot_callback(robot),
                profile,
            )
            for robot, topic in zip(self.robot_namespaces, self.robot_topics)
        ]
        self._subscriptions.append(
            self.create_subscription(
                String,
                args.slot_map_topic,
                self._on_slot_mapping,
                10,
            )
        )
        self._subscriptions.append(
            self.create_subscription(
                String,
                args.trigger_topic,
                self._on_selection_trigger,
                10,
            )
        )
        self.publisher = self.create_publisher(String, args.output_topic, 10)
        self.timer = self.create_timer(1.0 / args.rate_hz, self.tick)

        self.get_logger().info(f"HAR view-selection mode: {self.selection_mode}")
        if self.policy is not None:
            self.get_logger().info("HAR active-view DDQN loaded")
            self.get_logger().info(f"policy checkpoint: {self.policy.checkpoint}")
            self.get_logger().info(f"orientation checkpoint: {self.policy.orientation_checkpoint}")
        else:
            self.get_logger().info(
                f"baseline seed={args.random_seed} hold={args.baseline_hold_seconds:.1f}s"
            )
        self.get_logger().info(f"view topics: {topics}")
        self.get_logger().info(
            "safe mode: publishes selected view pairs only; Nav2 goal dispatch is disabled"
        )

    def _make_robot_callback(self, robot: str):
        def callback(message):
            slot = self.robot_slots.get(robot)
            if slot is None or not 0 <= int(slot) < NUM_VIEWS:
                return
            try:
                frame = (
                    decode_compressed_image(message)
                    if self.args.image_transport == "compressed"
                    else decode_raw_image(message)
                )
            except Exception as exc:
                self.get_logger().warning(f"slot {slot} image decode failed: {exc}")
                return
            self.latest_frames[slot] = frame
            self.latest_sequence[slot] += 1
            self.last_robot_frame_time[robot] = time.monotonic()

        return callback

    def _on_slot_mapping(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            mapping = payload.get("slots", payload)
            updated = {
                robot: int(mapping[robot])
                for robot in self.robot_namespaces
                if robot in mapping
            }
            if len(updated) != len(self.robot_namespaces):
                return
            values = list(updated.values())
            if len(set(values)) != len(values) or any(
                value < 0 or value >= NUM_VIEWS for value in values
            ):
                return
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            self.get_logger().warning("invalid ring robot-slot mapping")
            return

        if updated == self.robot_slots:
            return
        old_slots = dict(self.robot_slots)
        self.robot_slots = updated
        self.start_a = int(updated[self.robot_namespaces[0]])
        self.start_b = int(updated[self.robot_namespaces[1]])
        # A newly occupied slot must build a fresh 13-frame history. Histories
        # for slots that are no longer occupied are retained as past views.
        for robot, slot in updated.items():
            if old_slots.get(robot) != slot:
                self.pose_history[slot].clear()
                self.latest_frames[slot] = None
                self.latest_sequence[slot] = 0
                self.processed_sequence[slot] = 0
                # A slot remap means the next frame must come from the robot
                # at its new physical position. Do not reuse a frame received
                # before the move or while the robot is offline.
                self.last_robot_frame_time[robot] = None
        self._baseline_result = None
        self.get_logger().info(f"ring slot mapping updated: {self.robot_slots}")

    def _on_selection_trigger(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.get_logger().warning("invalid active-view selection trigger")
            return
        if not isinstance(payload, dict):
            self.get_logger().warning("active-view selection trigger must be a JSON object")
            return
        self._trigger_payload = payload
        self._selection_requested = True

    def _process_new_frames(self) -> None:
        for slot, frame in enumerate(self.latest_frames):
            if frame is None or self.latest_sequence[slot] == self.processed_sequence[slot]:
                continue
            if self.body is None:
                # Random/cyclic baselines only need to know which RGB slots are
                # alive. Keep the same 13-frame readiness contract without
                # loading a second RTMPose session.
                self.pose_history[slot].append(np.zeros((17, 3), dtype=np.float32))
                self.processed_sequence[slot] = self.latest_sequence[slot]
                continue
            try:
                # ROS image decoding returns OpenCV BGR frames, which is the
                # channel order expected by rtmlib's Body wrapper.
                keypoints, scores = self.body(frame)
                pose = normalized_rtmpose(keypoints, scores, frame.shape)
            except Exception as exc:
                self.get_logger().warning(f"slot {slot} RTMPose failed: {exc}")
                pose = np.zeros((17, 3), dtype=np.float32)
            self.pose_history[slot].append(pose)
            self.processed_sequence[slot] = self.latest_sequence[slot]

    def _view_observations(self) -> list[ViewObservation]:
        observations = []
        for slot in range(NUM_VIEWS):
            ready = len(self.pose_history[slot]) >= NUM_FRAMES
            pose = (
                np.stack(list(self.pose_history[slot])[-NUM_FRAMES:]).astype(np.float32)
                if ready
                else None
            )
            observations.append(
                ViewObservation(
                    geometry=self.geometry[slot],
                    rtmpose=pose,
                    # The exported full_depth19 policy accepts the same 19-D
                    # pooled layout. No live depth topic is assumed here, so
                    # missing objects/depth are explicit zero-valued inputs.
                    object_track=np.zeros((NUM_FRAMES, 50, 4), dtype=np.float32)
                    if ready
                    else None,
                    depth=None,
                    # In the physical ring, a candidate slot is known even
                    # before a robot has visited it. This lets the exported
                    # policy choose an unobserved next position; its missing
                    # feature tensors remain explicit zero fallbacks.
                    valid=ready or self.args.allow_unobserved_candidates,
                    source=self.topics[slot],
                )
            )
        return observations

    def _live_robot_inputs_ready(self) -> bool:
        """Require both physical camera streams to be current before selecting.

        The policy keeps temporal histories by design. Without this guard, a
        disconnected robot could leave a complete 13-frame history in memory,
        and a later trigger would make the selector act on stale data as if the
        robot were still observing the person.
        """

        now = time.monotonic()
        stale = []
        for robot in self.robot_namespaces:
            received = self.last_robot_frame_time.get(robot)
            if received is None or now - received > self.args.input_stale_seconds:
                stale.append(robot)
        if not stale:
            return True
        if now - self.last_warning > 10.0:
            self.get_logger().warning(
                "waiting for fresh RGB input from "
                f"{', '.join(stale)} "
                f"(timeout={self.args.input_stale_seconds:g}s); "
                "active-view selection is paused"
            )
            self.last_warning = now
        return False

    def _baseline_select(self, observations: list[ViewObservation]):
        """Select a live pair using the archive's random/cyclic protocol."""

        now = time.monotonic()
        valid = [bool(view.valid) for view in observations]
        if (
            self._baseline_result is not None
            and now < self._baseline_next_time
            and all(valid[index] for index in self._baseline_result["targets"])
        ):
            return self._baseline_result

        legal = [
            pair
            for pair in PAIR_LIST
            if valid[pair[0]] and valid[pair[1]]
        ]
        if not legal:
            return None

        if self.selection_mode == "random":
            pair = legal[int(self._baseline_rng.integers(len(legal)))]
        else:
            phase = self._baseline_step % NUM_VIEWS
            wanted = tuple(sorted((phase, (phase + 2) % NUM_VIEWS)))
            pair = wanted if wanted in legal else legal[self._baseline_step % len(legal)]
        self._baseline_step += 1

        geometry = np.stack([view.geometry for view in observations]).astype(np.float32)
        cost = np.arccos(np.clip(geometry @ geometry.T, -1.0, 1.0)) / np.pi
        left, right = pair
        direct = float(cost[self.start_a, left] + cost[self.start_b, right])
        swapped = float(cost[self.start_a, right] + cost[self.start_b, left])
        if direct <= swapped:
            assignment = {"robot_a": left, "robot_b": right}
            movement_cost = direct
        else:
            assignment = {"robot_a": right, "robot_b": left}
            movement_cost = swapped

        result = {
            "selection_mode": self.selection_mode,
            "variant": self.args.variant,
            "targets": [int(left), int(right)],
            "assignment": assignment,
            "start_slots": {"robot_a": self.start_a, "robot_b": self.start_b},
            "movement_cost_normalized": movement_cost,
            "pair_cost_normalized": movement_cost,
            "valid_views": [int(value) for value in valid],
            "geometry_sin_cos": geometry.tolist(),
            "depth_input": "zero_fallback",
            "baseline_seed": int(self.args.random_seed),
            "baseline_step": int(self._baseline_step),
            "robot_slots": dict(self.robot_slots),
        }
        self._baseline_result = result
        self._baseline_next_time = now + float(self.args.baseline_hold_seconds)
        return result

    def tick(self) -> None:
        self._process_new_frames()
        if not self._selection_requested:
            return
        if not self._live_robot_inputs_ready():
            return
        observations = self._view_observations()
        if self.selection_mode == "rl" and not all(
            len(self.pose_history[slot]) >= NUM_FRAMES
            for slot in (self.start_a, self.start_b)
        ):
            now = time.monotonic()
            if now - self.last_warning > 10.0:
                self.get_logger().warning(
                    "waiting for 13-frame histories at the two current ring slots"
                )
                self.last_warning = now
            return
        valid_count = sum(int(view.valid) for view in observations)
        if valid_count < 2:
            now = time.monotonic()
            if now - self.last_warning > 10.0:
                self.get_logger().warning(
                    f"waiting for two complete {NUM_FRAMES}-frame view histories; ready={valid_count}/{NUM_VIEWS}"
                )
                self.last_warning = now
            return
        try:
            if self.selection_mode == "rl":
                result = self.policy.select(
                    observations,
                    start_a=self.start_a,
                    start_b=self.start_b,
                )
                if result is not None:
                    result["selection_mode"] = "rl"
            else:
                result = self._baseline_select(observations)
        except Exception as exc:
            self.get_logger().error(f"view-selection policy failed: {exc}")
            return
        if result is None:
            self.get_logger().warning("active-view policy found no feasible pair")
            return
        result["stamp_sec"] = time.time()
        result["output_topic"] = self.args.output_topic
        result["robot_slots"] = dict(self.robot_slots)
        if self._trigger_payload is not None:
            # The coordinator deliberately sends only robot_1's latest
            # recognition as the feedback context. The exported checkpoint's
            # fixed input contract has no class-logit channel, so preserve the
            # trained state layout and record this causal trigger explicitly.
            result["recognition_basis"] = {
                "source": "rosbot_1",
                "trigger": self._trigger_payload,
            }
        message = String()
        message.data = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        self.publisher.publish(message)
        stable_result = {
            "targets": result["targets"],
            "assignment": result["assignment"],
            "start_slots": result["start_slots"],
            "movement_cost_normalized": result["movement_cost_normalized"],
        }
        if stable_result != self.last_result:
            self.get_logger().info(
                f"selected views={result['targets']} assignment={result['assignment']} "
                f"cost={result['movement_cost_normalized']:.3f}"
            )
            self.last_result = stable_result
        self._selection_requested = not self.args.wait_for_trigger
        self._trigger_payload = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-topics", default=None, help="Four comma-separated RGB topics")
    parser.add_argument(
        "--robot-topics",
        default=",".join(DEFAULT_ROBOT_TOPICS),
        help="Two physical RGB topics, routed to the current ring slots",
    )
    parser.add_argument(
        "--robot-namespaces",
        default=",".join(DEFAULT_ROBOT_NAMESPACES),
        help="Two robot namespaces corresponding to --robot-topics",
    )
    parser.add_argument("--image-transport", choices=["compressed", "raw"], default="compressed")
    parser.add_argument("--ros-qos-reliability", choices=["reliable", "best-effort"], default="best-effort")
    parser.add_argument("--ros-node-name", default="har_active_view_policy")
    parser.add_argument("--output-topic", default="/har/selected_view_pair")
    parser.add_argument("--slot-map-topic", default="/har/orbit/robot_slots")
    parser.add_argument(
        "--trigger-topic",
        default="/har/orbit/select_next",
        help="JSON trigger emitted after a fused recognition cycle",
    )
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument(
        "--input-stale-seconds",
        type=float,
        default=float(os.environ.get("HAR_VIEW_INPUT_STALE_SECONDS") or "3.0"),
        help="Pause selection if either physical RGB stream is older than this",
    )
    parser.add_argument("--view-angles-deg", default=",".join(str(value) for value in DEFAULT_ANGLES_DEG))
    parser.add_argument("--start-a", type=int, default=0)
    parser.add_argument("--start-b", type=int, default=1)
    parser.add_argument(
        "--allow-unobserved-candidates",
        action="store_true",
        help="Treat known ring slots as legal before a robot has visited them",
    )
    parser.add_argument(
        "--wait-for-trigger",
        action="store_true",
        help="Publish one initial decision, then wait for /har/orbit/select_next",
    )
    parser.add_argument(
        "--no-initial-selection",
        action="store_true",
        help="Do not publish the initial decision until a trigger arrives",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["rl", "random", "cyclic"],
        default=os.environ.get("HAR_VIEW_SELECTION_MODE") or "rl",
        help="rl=exported DDQN, random=random_pair baseline, cyclic=cyclic_pair baseline",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=int(os.environ.get("HAR_VIEW_RANDOM_SEED") or "20260915"),
    )
    parser.add_argument(
        "--baseline-hold-seconds",
        type=float,
        default=float(os.environ.get("HAR_VIEW_BASELINE_HOLD_SECONDS") or "3.0"),
    )
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--orientation-checkpoint", type=Path)
    parser.add_argument("--variant", default="full_depth19")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--rtmpose-det", type=Path, default=DEFAULT_RTMPOSE_DET)
    parser.add_argument("--rtmpose-pose", type=Path, default=DEFAULT_RTMPOSE_POSE)
    parser.add_argument("--rtmpose-mode", choices=["lightweight", "balanced", "performance"], default="balanced")
    parser.add_argument("--rtmpose-backend", default="onnxruntime")
    parser.add_argument("--rtmpose-device", default="cuda")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.view_topics:
        topics = parse_csv_topics(args.view_topics)
    else:
        topics = parse_csv_topics(os.environ.get("HAR_VIEW_TOPICS", ",".join(DEFAULT_TOPICS)))
    angles = parse_csv_floats(args.view_angles_deg)
    args.robot_namespaces = parse_csv_robot_namespaces(args.robot_namespaces)
    args.robot_topics = parse_csv_robot_topics(args.robot_topics)
    if args.policy_checkpoint is None:
        from viewpoint_policy import POLICY_CHECKPOINT

        args.policy_checkpoint = POLICY_CHECKPOINT
    if args.orientation_checkpoint is None:
        from viewpoint_policy import ORIENTATION_CHECKPOINT

        args.orientation_checkpoint = ORIENTATION_CHECKPOINT
    if not 0 <= args.start_a < NUM_VIEWS or not 0 <= args.start_b < NUM_VIEWS or args.start_a == args.start_b:
        raise ValueError("--start-a and --start-b must be different slots in 0..3")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if args.input_stale_seconds <= 0:
        raise ValueError("--input-stale-seconds must be positive")
    if args.baseline_hold_seconds <= 0:
        raise ValueError("--baseline-hold-seconds must be positive")
    if not args.output_topic.startswith("/"):
        raise ValueError("--output-topic must be an absolute ROS topic")
    if not args.slot_map_topic.startswith("/") or not args.trigger_topic.startswith("/"):
        raise ValueError("--slot-map-topic and --trigger-topic must be absolute ROS topics")

    rclpy.init(args=None)
    node = None
    try:
        node = ActiveViewNode(args, topics, angles)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
