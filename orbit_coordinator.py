#!/usr/bin/env python3
"""Coordinate two ROSbots on a human-centred four-position orbit.

The default mode is intentionally non-actuating. In ``dry_run`` the node
computes and prints map-frame targets and sampled arc waypoints, publishes the
same diagnostics for RViz/monitoring, and never creates or sends a Nav2 goal.
Live mode keeps the same plan but sends one ``NavigateToPose`` goal at a time:
rosbot_1 finishes before rosbot_2 starts, and both follow controllers stay
disabled until both robots have settled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, Pose, PoseArray, PoseStamped, Quaternion
from person_follow_interfaces.msg import PersonDetection
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import SetBool
import tf2_ros

try:  # The separate jazzy-rosbot container owns Nav2 interfaces.
    from nav2_msgs.action import NavigateToPose
except ImportError:  # Dry-run can run in the lighter HAR/ML container.
    NavigateToPose = None

# Importing the geometry adapter registers PointStamped support with tf2 in
# ROS 2 distributions where Buffer.transform discovers types at import time.
try:  # pragma: no cover - availability is determined by the ROS image
    import tf2_geometry_msgs  # noqa: F401
except ImportError:  # pragma: no cover
    tf2_geometry_msgs = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "orbit_system_config.csv"
NUM_SLOTS = 4


@dataclass
class HumanEstimate:
    x: float
    y: float
    z: float
    source: str
    stamp_sec: float


@dataclass
class RobotPose:
    x: float
    y: float
    yaw: float
    stamp_sec: float


def load_config(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        result = {}
        for row in rows:
            key = str(row.get("key", "")).strip()
            if key and not key.startswith("#"):
                result[key] = str(row.get("value", "")).strip()
    return result


def config_bool(config: dict[str, str], key: str, default: bool) -> bool:
    value = config.get(key)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(f"invalid boolean for {key}: {value}")


def config_float(config: dict[str, str], key: str, default: float) -> float:
    value = config.get(key)
    return default if value is None or value.strip() == "" else float(value)


def config_list(config: dict[str, str], key: str, default: list[str]) -> list[str]:
    value = config.get(key)
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]


def parse_pair(value: str, name: str) -> list[int]:
    items = [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    result = [int(item) for item in items]
    if len(set(result)) != 2 or any(item < 0 or item >= NUM_SLOTS for item in result):
        raise ValueError(f"{name} must contain two distinct slots in 0..3")
    return result


def parse_angles(value: str) -> list[float]:
    items = [float(item) for item in value.replace(",", ";").split(";") if item.strip()]
    if len(items) != NUM_SLOTS:
        raise ValueError("ring_angles_deg must contain exactly four values")
    return items


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(quaternion: Quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def yaw_quaternion(yaw: float) -> Quaternion:
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(float(yaw) * 0.5),
        w=math.cos(float(yaw) * 0.5),
    )


def stamp_to_sec(stamp: Any) -> float:
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


class OrbitCoordinator(Node):
    """Human-centred ring planner with a serialized two-robot state machine."""

    def __init__(self, args: argparse.Namespace, config: dict[str, str]) -> None:
        super().__init__("har_orbit_coordinator")
        self.args = args
        self.config = config
        self.robots = list(config_list(config, "robot_namespaces", ["rosbot_1", "rosbot_2"]))
        if len(self.robots) != 2 or len(set(self.robots)) != 2:
            raise ValueError("robot_namespaces must contain two different names")

        self.dry_run = (
            bool(args.dry_run)
            if args.dry_run is not None
            else config_bool(config, "dry_run", True)
        )
        self.once = config_bool(config, "dry_run_once", True) and not args.continuous
        # In dry-run mode no robot physically changes position. Keeping the
        # physical slot map unchanged prevents the active-view selector from
        # routing the old camera stream into a fictitious new ring slot.
        self.dry_run_update_slot_mapping = config_bool(
            config, "dry_run_update_slot_mapping", False
        )
        self.radius = (
            float(args.ring_radius_m)
            if args.ring_radius_m is not None
            else config_float(config, "ring_radius_m", 1.5)
        )
        angle_text = args.ring_angles_deg or ";".join(config_list(config, "ring_angles_deg", ["0", "90", "180", "270"]))
        self.angles_deg = parse_angles(angle_text)
        self.angles_rad = [math.radians(value) for value in self.angles_deg]
        self.initial_robot_slots = parse_pair(
            ";".join(config_list(config, "initial_robot_slots", ["0", "1"])),
            "initial_robot_slots",
        )
        self.bootstrap_pair = parse_pair(
            args.bootstrap_pair
            or ";".join(config_list(config, "bootstrap_pair", ["0", "2"])),
            "bootstrap_pair",
        )
        self.human_frame = args.human_frame or config.get("human_frame", "map")
        self.target_tolerance = config_float(config, "target_tolerance_m", 0.15)
        self.arc_step_rad = math.radians(config_float(config, "arc_step_deg", 15.0))
        self.allow_radial_acquire = config_bool(config, "allow_radial_acquire", True)
        self.min_robot_separation = config_float(config, "min_robot_separation_m", 0.8)
        self.person_timeout = config_float(config, "person_timeout_sec", 1.0)
        self.settle_seconds = config_float(config, "settle_seconds", 3.0)
        self.recognition_timeout = config_float(config, "recognition_timeout_sec", 10.0)
        self.initial_selection_timeout = config_float(
            config, "initial_selection_timeout_sec", 8.0
        )
        self.depth_suffix = config.get("depth_topic_suffix", "camera/depth/image")
        self.info_suffix = config.get("camera_info_topic_suffix", "camera/depth/camera_info")
        self.base_suffix = config.get("base_frame_suffix", "base_link")
        self.follow_suffix = config.get(
            "follow_service_suffix", "person_follow_controller/set_enabled"
        )

        self.person_topics = [
            f"/{robot}/follow/person_detection" for robot in self.robots
        ]
        self.depth_topics = [f"/{robot}/{self.depth_suffix}" for robot in self.robots]
        self.info_topics = [f"/{robot}/{self.info_suffix}" for robot in self.robots]
        self.prediction_topics = [
            config.get("prediction_topic_1", "/rosbot_1/vpoclip/prediction"),
            config.get("prediction_topic_2", "/rosbot_2/vpoclip/prediction"),
        ]

        self.latest_person: dict[str, tuple[PersonDetection, float]] = {}
        self.latest_depth: dict[str, tuple[np.ndarray, str, float, float]] = {}
        self.latest_info: dict[str, CameraInfo] = {}
        self.latest_robot1_prediction: dict[str, Any] | None = None
        self.latest_robot2_prediction: dict[str, Any] | None = None
        self.latest_selection: dict[str, Any] | None = None
        self.selection_version = 0
        self.last_used_selection_version = -1
        self.final_action: dict[str, Any] | None = None
        self.last_warning_time = 0.0
        self.first_human_time: float | None = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.state_topic = config.get("state_topic", "/har/orbit/state")
        self.plan_topic = config.get("plan_topic", "/har/orbit/plan")
        self.target_pose_topic = config.get("target_pose_topic", "/har/orbit/target_poses")
        self.slot_map_topic = config.get("slot_map_topic", "/har/orbit/robot_slots")
        self.recognition_cycle_topic = config.get(
            "recognition_cycle_topic", "/har/orbit/recognition_cycle"
        )
        self.select_next_topic = config.get("select_next_topic", "/har/orbit/select_next")
        self.fusion_topic = config.get("fusion_topic", "/har/final_action")
        # Keep the most recent diagnostics available to RViz/monitoring tools
        # that are started after a plan was generated.  This is ROS 2's
        # latched-style transient-local behavior and does not affect control.
        diagnostic_qos = QoSProfile(depth=1)
        diagnostic_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.state_publisher = self.create_publisher(String, self.state_topic, diagnostic_qos)
        self.plan_publisher = self.create_publisher(String, self.plan_topic, diagnostic_qos)
        self.pose_publisher = self.create_publisher(PoseArray, self.target_pose_topic, diagnostic_qos)
        self.slot_publisher = self.create_publisher(String, self.slot_map_topic, diagnostic_qos)
        self.recognition_publisher = self.create_publisher(
            String, self.recognition_cycle_topic, 10
        )
        self.select_next_publisher = self.create_publisher(String, self.select_next_topic, 10)

        self.create_subscription(
            String, config.get("selected_view_topic", "/har/selected_view_pair"), self._on_selection, 10
        )
        self.create_subscription(String, self.prediction_topics[0], self._on_robot1_prediction, 10)
        self.create_subscription(String, self.prediction_topics[1], self._on_robot2_prediction, 10)
        self.create_subscription(String, self.fusion_topic, self._on_final_action, 10)

        for robot, person_topic, depth_topic, info_topic in zip(
            self.robots, self.person_topics, self.depth_topics, self.info_topics
        ):
            self.create_subscription(
                PersonDetection,
                person_topic,
                self._make_person_callback(robot),
                10,
            )
            self.create_subscription(
                Image,
                depth_topic,
                self._make_depth_callback(robot),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo,
                info_topic,
                self._make_info_callback(robot),
                qos_profile_sensor_data,
            )

        self.follow_clients = {
            robot: self.create_client(SetBool, f"/{robot}/{self.follow_suffix}")
            for robot in self.robots
        }
        self.nav_clients = {}
        if not self.dry_run:
            if NavigateToPose is None:
                raise RuntimeError(
                    "LIVE orbit mode needs nav2_msgs in the coordinator container. "
                    "Run the coordinator in the Nav2 Jazzy environment or install ros-jazzy-nav2-msgs."
                )
            self.nav_clients = {
                robot: ActionClient(self, NavigateToPose, f"/{robot}/navigate_to_pose")
                for robot in self.robots
            }

        # These are the positions occupied before the first move.  The
        # bootstrap pair is a fallback target pair, not the current physical
        # location; keeping the two settings separate is important for both
        # the DDQN movement cost and the dry-run printout.
        self.robot_slots = {
            self.robots[0]: self.initial_robot_slots[0],
            self.robots[1]: self.initial_robot_slots[1],
        }
        self.motion_cycle = 0
        self.recognition_cycle = 0
        self.state = "WAIT_HUMAN"
        self.active_plan: dict[str, Any] | None = None
        self.plan_target_slots: dict[str, int] = {}
        self.settle_deadline = 0.0
        self.recognition_deadline = 0.0
        self.dry_stage_deadline = 0.0
        self.nav_robot: str | None = None
        self.nav_waypoint_index = 0
        self.nav_goal_handle = None
        self.nav_request_in_flight = False
        self.nav_retry_time = 0.0

        self._publish_slot_mapping()
        self.timer = self.create_timer(0.2, self._tick)
        self.get_logger().info(
            f"orbit coordinator: dry_run={self.dry_run} radius={self.radius:.2f}m "
            f"angles={self.angles_deg} robots={self.robots}"
        )
        if self.dry_run:
            self.get_logger().warning(
                "DRY-RUN active: target coordinates are printed, no Nav2 goal and no follow enable call will be made"
            )

    # ------------------------------------------------------------------ ROS input
    def _make_person_callback(self, robot: str):
        def callback(message: PersonDetection) -> None:
            self.latest_person[robot] = (message, time.monotonic())

        return callback

    def _make_depth_callback(self, robot: str):
        def callback(message: Image) -> None:
            try:
                array = self._decode_depth(message)
            except (TypeError, ValueError) as exc:
                self._warn_throttled(f"depth decode failed for {robot}: {exc}")
                return
            self.latest_depth[robot] = (
                array,
                str(message.header.frame_id),
                stamp_to_sec(message.header.stamp),
                time.monotonic(),
            )

        return callback

    def _make_info_callback(self, robot: str):
        def callback(message: CameraInfo) -> None:
            self.latest_info[robot] = message

        return callback

    @staticmethod
    def _decode_depth(message: Image) -> np.ndarray:
        encoding = str(message.encoding).upper()
        if encoding == "32FC1":
            dtype = np.dtype("<f4")
            scale = 1.0
        elif encoding == "16UC1":
            dtype = np.dtype("<u2")
            scale = 0.001
        elif encoding == "16SC1":
            dtype = np.dtype("<i2")
            scale = 0.001
        else:
            raise ValueError(f"unsupported encoding {message.encoding!r}")
        stride = int(message.step) // dtype.itemsize
        expected = int(message.height) * stride
        data = np.frombuffer(message.data, dtype=dtype)
        if data.size < expected or stride < int(message.width):
            raise ValueError(
                f"depth buffer has {data.size} values, expected {expected}"
            )
        return (
            data[:expected]
            .reshape(int(message.height), stride)[:, : int(message.width)]
            .astype(np.float32, copy=False)
            * scale
        )

    def _on_selection(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self._warn_throttled("invalid /har/selected_view_pair JSON")
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("targets"), list):
            self._warn_throttled("active-view selection has no targets list")
            return
        self.latest_selection = payload
        self.selection_version += 1
        self.get_logger().info(
            f"active-view decision v{self.selection_version}: targets={payload.get('targets')}"
        )

    def _on_robot1_prediction(self, message: String) -> None:
        self.latest_robot1_prediction = self._parse_json_message(message, "rosbot_1 prediction")

    def _on_robot2_prediction(self, message: String) -> None:
        self.latest_robot2_prediction = self._parse_json_message(message, "rosbot_2 prediction")

    def _on_final_action(self, message: String) -> None:
        payload = self._parse_json_message(message, "fused action")
        if payload is None or self.state != "RECOGNIZING":
            return
        if int(payload.get("cycle_id", -1)) != self.recognition_cycle:
            return
        self.final_action = payload
        action_label = payload.get("action_label")
        self.get_logger().info(
            f"recognition cycle {self.recognition_cycle} final action={action_label} "
            f"decision={payload.get('decision')}"
        )
        if self.once and self.dry_run:
            self.state = "DRY_RUN_COMPLETE"
            self._publish_state()
            return
        self.state = "WAIT_NEXT_SELECTION"
        # The only recognition payload forwarded to active-view selection is
        # the exact rosbot_1 event that the fusion node used for this cycle.
        robot1_basis = payload.get("robot1_prediction") or self.latest_robot1_prediction
        trigger = {
            "schema": "har.orbit.select_next.v1",
            "cycle_id": self.recognition_cycle,
            "reason": "fused_recognition_complete",
            "final_action": payload,
            "robot1_prediction": robot1_basis,
            "robot1_only_feedback": True,
        }
        output = String()
        output.data = json.dumps(trigger, ensure_ascii=False, separators=(",", ":"))
        self.select_next_publisher.publish(output)
        self._publish_state()

    @staticmethod
    def _parse_json_message(message: String, description: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    # --------------------------------------------------------------- human/TF math
    def _estimate_human(self) -> HumanEstimate | None:
        if self.args.human_x is not None and self.args.human_y is not None:
            return HumanEstimate(
                float(self.args.human_x),
                float(self.args.human_y),
                float(self.args.human_z),
                "cli_override",
                time.time(),
            )

        now = time.monotonic()
        for robot in self.robots:
            person_item = self.latest_person.get(robot)
            depth_item = self.latest_depth.get(robot)
            camera_info = self.latest_info.get(robot)
            if person_item is None or depth_item is None or camera_info is None:
                continue
            detection, received_at = person_item
            if now - received_at > self.person_timeout or not detection.detected:
                continue
            depth, depth_frame, depth_stamp, depth_received = depth_item
            if now - depth_received > max(1.0, self.person_timeout * 2.0):
                continue
            if len(camera_info.k) < 9 or camera_info.k[0] <= 0.0 or camera_info.k[4] <= 0.0:
                continue
            u = float(detection.center_x_px)
            v = float(detection.center_y_px)
            height, width = depth.shape[:2]
            if not (0.0 <= u < width and 0.0 <= v < height):
                continue
            half_x = max(3, int(round((detection.bbox_x_max_px - detection.bbox_x_min_px) * 0.04)))
            half_y = max(3, int(round((detection.bbox_y_max_px - detection.bbox_y_min_px) * 0.04)))
            x0 = max(0, int(round(u)) - half_x)
            x1 = min(width, int(round(u)) + half_x + 1)
            y0 = max(0, int(round(v)) - half_y)
            y1 = min(height, int(round(v)) + half_y + 1)
            values = depth[y0:y1, x0:x1]
            values = values[np.isfinite(values) & (values > 0.15) & (values < 12.0)]
            if values.size == 0:
                continue
            z = float(np.median(values))
            point = PointStamped()
            point.header.frame_id = depth_frame or f"{robot}/camera_color_optical_frame"
            point.header.stamp = self._latest_depth_stamp(depth_stamp)
            point.point.x = (u - float(camera_info.k[2])) / float(camera_info.k[0]) * z
            point.point.y = (v - float(camera_info.k[5])) / float(camera_info.k[4]) * z
            point.point.z = z
            try:
                transformed = self.tf_buffer.transform(
                    point,
                    self.human_frame,
                    timeout=Duration(seconds=0.2),
                )
            except Exception as exc:  # TF may be warming up or map unavailable
                self._warn_throttled(f"cannot transform human from {point.header.frame_id}: {exc}")
                continue
            return HumanEstimate(
                float(transformed.point.x),
                float(transformed.point.y),
                float(transformed.point.z),
                f"{robot}_rgbd_bbox_depth",
                time.time(),
            )
        return None

    @staticmethod
    def _latest_depth_stamp(depth_stamp: float):
        # A zero stamp asks tf2 for the latest transform and is preferable to
        # fabricating a ROS time when the camera stream has no valid stamp.
        from builtin_interfaces.msg import Time as BuiltinTime

        result = BuiltinTime()
        if depth_stamp > 0.0:
            result.sec = int(depth_stamp)
            result.nanosec = int(round((depth_stamp - int(depth_stamp)) * 1e9))
        return result

    def _robot_pose(self, robot: str) -> RobotPose | None:
        frame = f"{robot}/{self.base_suffix}"
        try:
            transform = self.tf_buffer.lookup_transform(
                self.human_frame,
                frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except Exception as exc:
            self._warn_throttled(f"cannot lookup {self.human_frame}->{frame}: {exc}")
            return None
        translation = transform.transform.translation
        return RobotPose(
            float(translation.x),
            float(translation.y),
            quaternion_to_yaw(transform.transform.rotation),
            stamp_to_sec(transform.header.stamp),
        )

    def _ring_point(self, human: HumanEstimate, slot: int) -> tuple[float, float]:
        angle = self.angles_rad[int(slot)]
        return (
            human.x + self.radius * math.cos(angle),
            human.y + self.radius * math.sin(angle),
        )

    def _target_pose_dict(self, human: HumanEstimate, slot: int) -> dict[str, Any]:
        x, y = self._ring_point(human, slot)
        yaw = math.atan2(human.y - y, human.x - x)
        return {
            "slot": int(slot),
            "angle_deg": float(self.angles_deg[int(slot)]),
            "x": float(x),
            "y": float(y),
            "yaw_rad": float(yaw),
            "yaw_deg": float(math.degrees(yaw)),
        }

    def _make_robot_waypoints(
        self,
        robot: str,
        target_slot: int,
        human: HumanEstimate,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        current = self._robot_pose(robot)
        warnings = []
        if current is None:
            start_angle = self.angles_rad[self.robot_slots.get(robot, 0)]
            current_radius = self.radius
            warnings.append(f"{robot}: current TF unavailable; arc start angle inferred from slot")
        else:
            start_angle = math.atan2(current.y - human.y, current.x - human.x)
            current_radius = math.hypot(current.x - human.x, current.y - human.y)

        target_angle = self.angles_rad[int(target_slot)]
        delta = wrap_angle(target_angle - start_angle)
        waypoints: list[dict[str, Any]] = []
        if abs(current_radius - self.radius) > self.target_tolerance:
            if self.allow_radial_acquire:
                x = human.x + self.radius * math.cos(start_angle)
                y = human.y + self.radius * math.sin(start_angle)
                waypoints.append(
                    {
                        "kind": "ring_entry",
                        "slot": None,
                        "angle_deg": math.degrees(start_angle),
                        "x": float(x),
                        "y": float(y),
                        "yaw_rad": float(math.atan2(human.y - y, human.x - x)),
                    }
                )
                warnings.append(
                    f"{robot}: initial radial acquisition {current_radius:.2f}m -> {self.radius:.2f}m"
                )
            else:
                warnings.append(
                    f"{robot}: start radius {current_radius:.2f}m is outside ring tolerance"
                )

        steps = max(1, int(math.ceil(abs(delta) / max(self.arc_step_rad, 1e-3))))
        for index in range(1, steps + 1):
            angle = start_angle + delta * index / steps
            x = human.x + self.radius * math.cos(angle)
            y = human.y + self.radius * math.sin(angle)
            waypoints.append(
                {
                    "kind": "arc" if index < steps else "target",
                    "slot": int(target_slot) if index == steps else None,
                    "angle_deg": float(math.degrees(angle)),
                    "x": float(x),
                    "y": float(y),
                    "yaw_rad": float(math.atan2(human.y - y, human.x - x)),
                }
            )
        if not waypoints:
            waypoints.append(self._target_pose_dict(human, target_slot))
        return waypoints, warnings

    def _build_plan(
        self,
        human: HumanEstimate,
        target_slots: list[int],
        assignment: dict[str, int],
        selection_source: str,
    ) -> dict[str, Any]:
        target_poses = [self._target_pose_dict(human, slot) for slot in range(NUM_SLOTS)]
        robot_waypoints = {}
        warnings = []
        for robot in self.robots:
            slot = int(assignment[robot])
            robot_waypoints[robot], robot_warnings = self._make_robot_waypoints(robot, slot, human)
            warnings.extend(robot_warnings)

        for left in range(len(self.robots)):
            for right in range(left + 1, len(self.robots)):
                left_target = target_poses[int(assignment[self.robots[left]])]
                right_target = target_poses[int(assignment[self.robots[right]])]
                separation = math.hypot(
                    left_target["x"] - right_target["x"],
                    left_target["y"] - right_target["y"],
                )
                if separation < self.min_robot_separation:
                    warnings.append(
                        f"target separation {separation:.2f}m < guard {self.min_robot_separation:.2f}m"
                    )

        return {
            "schema": "har.orbit.plan.v1",
            "motion_cycle": int(self.motion_cycle),
            "dry_run": bool(self.dry_run),
            "human_frame": self.human_frame,
            "human_center": {
                "x": float(human.x),
                "y": float(human.y),
                "z": float(human.z),
                "source": human.source,
                "stamp_sec": float(human.stamp_sec),
            },
            "radius_m": float(self.radius),
            "ring_angles_deg": [float(value) for value in self.angles_deg],
            "target_slots": [int(value) for value in target_slots],
            "assignment": {robot: int(assignment[robot]) for robot in self.robots},
            "selection_source": selection_source,
            "robot_slots_before": dict(self.robot_slots),
            "target_poses": target_poses,
            "robot_waypoints": robot_waypoints,
            "warnings": warnings,
        }

    # ------------------------------------------------------------- state machine
    def _tick(self) -> None:
        now = time.monotonic()
        if self.state == "WAIT_HUMAN":
            human = self._estimate_human()
            if human is not None:
                self.first_human_time = now
                self.state = "WAIT_INITIAL_SELECTION"
                self._publish_state(human)
            return

        if self.state == "WAIT_INITIAL_SELECTION":
            human = self._estimate_human()
            if human is None:
                self.state = "WAIT_HUMAN"
                self._publish_state()
                return
            if self.latest_selection is not None:
                self._start_plan(human, self.latest_selection, "active_view")
                return
            if self.first_human_time is not None and now - self.first_human_time >= self.initial_selection_timeout:
                self._start_plan(human, None, "bootstrap_pair_timeout")
            return

        if self.state.startswith("DRY_RUN_"):
            self._tick_dry_run(now)
            return

        if self.state == "MOVING_ROSBOT_1" or self.state == "MOVING_ROSBOT_2":
            if not self.dry_run and self.nav_goal_handle is None and not self.nav_request_in_flight and now >= self.nav_retry_time:
                self._send_next_nav_goal()
            return

        if self.state == "SETTLING":
            human = self._estimate_human()
            if now >= self.settle_deadline and human is not None:
                self._open_recognition_cycle(human)
            return

        if self.state == "RECOGNIZING":
            if now >= self.recognition_deadline:
                self.get_logger().warning(
                    f"recognition cycle {self.recognition_cycle} timed out waiting for both VPOCLIP results"
                )
                self.state = "DRY_RUN_COMPLETE" if self.dry_run else "WAIT_NEXT_SELECTION"
                self._publish_state()
            return

        if self.state == "WAIT_NEXT_SELECTION":
            if self.latest_selection is not None and self.selection_version > self.last_used_selection_version:
                human = self._estimate_human()
                if human is not None:
                    self._start_plan(human, self.latest_selection, "active_view_after_rosbot1_feedback")
            return

    def _start_plan(
        self,
        human: HumanEstimate,
        selection: dict[str, Any] | None,
        selection_source: str,
    ) -> None:
        if self.active_plan is not None and self.state not in {"WAIT_INITIAL_SELECTION", "WAIT_NEXT_SELECTION"}:
            return
        targets = self.bootstrap_pair
        assignment = {self.robots[0]: targets[0], self.robots[1]: targets[1]}
        if selection is not None:
            try:
                selected = [int(value) for value in selection.get("targets", [])]
                if len(selected) == 2 and len(set(selected)) == 2 and all(0 <= value < NUM_SLOTS for value in selected):
                    targets = selected
                candidate_assignment = selection.get("assignment") or {}
                robot_a = int(candidate_assignment.get("robot_a", targets[0]))
                robot_b = int(candidate_assignment.get("robot_b", targets[1]))
                if robot_a != robot_b and all(0 <= value < NUM_SLOTS for value in (robot_a, robot_b)):
                    assignment = {self.robots[0]: robot_a, self.robots[1]: robot_b}
                    targets = [robot_a, robot_b]
            except (TypeError, ValueError):
                self._warn_throttled("invalid active-view assignment; using bootstrap pair")
                selection_source = "bootstrap_after_invalid_selection"
        if len(set(assignment.values())) != 2:
            self._warn_throttled("refusing duplicate target slots")
            return

        self.motion_cycle += 1
        self.last_used_selection_version = self.selection_version
        self.plan_target_slots = dict(assignment)
        self.active_plan = self._build_plan(human, targets, assignment, selection_source)
        self._publish_plan(self.active_plan)
        self._print_plan(self.active_plan)
        self._set_follow_all(False)

        if self.dry_run:
            self.state = "DRY_RUN_ROSBOT_1"
            self.dry_stage_deadline = time.monotonic() + 0.8
            self._publish_state(human)
            return

        self.nav_robot = self.robots[0]
        self.nav_waypoint_index = 0
        self.nav_goal_handle = None
        self.nav_request_in_flight = False
        self.state = "MOVING_ROSBOT_1"
        self._send_next_nav_goal()
        self._publish_state(human)

    def _tick_dry_run(self, now: float) -> None:
        if now < self.dry_stage_deadline:
            return
        if self.state == "DRY_RUN_ROSBOT_1":
            robot = self.robots[0]
            self._mark_robot_arrived(robot)
            self.get_logger().info(
                "[DRY-RUN] rosbot_1 is considered settled for sequencing; rosbot_2 may start now"
            )
            self.state = "DRY_RUN_ROSBOT_2"
            self.dry_stage_deadline = now + 0.8
            self._publish_state()
        elif self.state == "DRY_RUN_ROSBOT_2":
            robot = self.robots[1]
            self._mark_robot_arrived(robot)
            self.get_logger().info("[DRY-RUN] both robot positions are considered reached")
            self.state = "SETTLING"
            self.settle_deadline = now + self.settle_seconds
            self._publish_state()

    def _mark_robot_arrived(self, robot: str) -> None:
        if robot in self.plan_target_slots:
            if self.dry_run and not self.dry_run_update_slot_mapping:
                self.get_logger().info(
                    f"[DRY-RUN] {robot} is simulated as arrived at slot "
                    f"{self.plan_target_slots[robot]}; physical slot map remains "
                    f"{self.robot_slots[robot]}"
                )
                return
            self.robot_slots[robot] = int(self.plan_target_slots[robot])
            self._publish_slot_mapping()

    def _open_recognition_cycle(self, human: HumanEstimate) -> None:
        self.recognition_cycle += 1
        self.state = "RECOGNIZING"
        self.recognition_deadline = time.monotonic() + self.recognition_timeout
        payload = {
            "schema": "har.orbit.recognition_cycle.v1",
            "cycle_id": int(self.recognition_cycle),
            "start_sec": time.time(),
            "settle_seconds": float(self.settle_seconds),
            "human_center": {
                "x": float(human.x),
                "y": float(human.y),
                "z": float(human.z),
                "source": human.source,
            },
            "targets": dict(self.plan_target_slots),
            "robot_slots": dict(self.robot_slots),
            "robot1_prediction_before_cycle": self.latest_robot1_prediction,
        }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.recognition_publisher.publish(message)
        self.get_logger().info(
            f"recognition cycle {self.recognition_cycle} opened after {self.settle_seconds:.1f}s settle"
        )
        if config_bool(self.config, "follow_enabled_after_settle", True) and not self.dry_run:
            self._set_follow_all(True)
        self._publish_state(human)

    # ------------------------------------------------------------- live Nav2 path
    def _send_next_nav_goal(self) -> None:
        if self.dry_run or self.nav_robot is None or self.active_plan is None:
            return
        waypoints = self.active_plan["robot_waypoints"].get(self.nav_robot, [])
        if self.nav_waypoint_index >= len(waypoints):
            self._on_robot_navigation_complete(self.nav_robot)
            return
        client = self.nav_clients[self.nav_robot]
        if not client.wait_for_server(timeout_sec=0.2):
            self.nav_retry_time = time.monotonic() + 1.0
            self._warn_throttled(f"waiting for /{self.nav_robot}/navigate_to_pose action server")
            return
        waypoint = waypoints[self.nav_waypoint_index]
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.human_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(waypoint["x"])
        goal.pose.pose.position.y = float(waypoint["y"])
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation = yaw_quaternion(float(waypoint["yaw_rad"]))
        self.nav_request_in_flight = True
        future = client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)
        self.get_logger().info(
            f"{self.nav_robot} waypoint {self.nav_waypoint_index + 1}/{len(waypoints)} "
            f"({waypoint['kind']}) -> ({waypoint['x']:.3f}, {waypoint['y']:.3f})"
        )

    def _on_goal_response(self, future) -> None:
        self.nav_request_in_flight = False
        try:
            handle = future.result()
        except Exception as exc:
            self._fail_live_motion(f"Nav2 goal request failed: {exc}")
            return
        if not handle.accepted:
            self._fail_live_motion(f"Nav2 rejected {self.nav_robot} waypoint")
            return
        self.nav_goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        try:
            result = future.result()
            status = int(result.status)
        except Exception as exc:
            self._fail_live_motion(f"Nav2 result unavailable: {exc}")
            return
        self.nav_goal_handle = None
        if status != GoalStatus.STATUS_SUCCEEDED:
            self._fail_live_motion(
                f"{self.nav_robot} waypoint failed with action status {status}"
            )
            return
        self.nav_waypoint_index += 1
        self._send_next_nav_goal()

    def _on_robot_navigation_complete(self, robot: str) -> None:
        self.nav_goal_handle = None
        self._mark_robot_arrived(robot)
        if robot == self.robots[0]:
            # rosbot_2 is not even sent a goal until rosbot_1 has completed all
            # arc waypoints and its final target has been accepted.
            self.nav_robot = self.robots[1]
            self.nav_waypoint_index = 0
            self.state = "MOVING_ROSBOT_2"
            self._send_next_nav_goal()
            self._publish_state()
            return
        self.state = "SETTLING"
        self.settle_deadline = time.monotonic() + self.settle_seconds
        self._publish_state()

    def _fail_live_motion(self, reason: str) -> None:
        self.state = "ERROR"
        self._set_follow_all(False)
        self.get_logger().error(reason)
        self._publish_state()

    # ---------------------------------------------------------------- diagnostics
    def _publish_slot_mapping(self) -> None:
        message = String()
        message.data = json.dumps(
            {
                "schema": "har.orbit.robot_slots.v1",
                "frame": self.human_frame,
                "slots": dict(self.robot_slots),
                "stamp_sec": time.time(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.slot_publisher.publish(message)

    def _publish_plan(self, plan: dict[str, Any]) -> None:
        message = String()
        message.data = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
        self.plan_publisher.publish(message)

        pose_array = PoseArray()
        pose_array.header.frame_id = self.human_frame
        pose_array.header.stamp = self.get_clock().now().to_msg()
        for target in plan["target_poses"]:
            pose = Pose()
            pose.position.x = float(target["x"])
            pose.position.y = float(target["y"])
            pose.position.z = 0.0
            pose.orientation = yaw_quaternion(float(target["yaw_rad"]))
            pose_array.poses.append(pose)
        self.pose_publisher.publish(pose_array)

    def _publish_state(self, human: HumanEstimate | None = None) -> None:
        payload = {
            "schema": "har.orbit.state.v1",
            "state": self.state,
            "dry_run": bool(self.dry_run),
            "motion_cycle": int(self.motion_cycle),
            "recognition_cycle": int(self.recognition_cycle),
            "robot_slots": dict(self.robot_slots),
            "stamp_sec": time.time(),
        }
        if human is not None:
            payload["human_center"] = {
                "x": float(human.x),
                "y": float(human.y),
                "z": float(human.z),
                "source": human.source,
            }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.state_publisher.publish(message)

    def _print_plan(self, plan: dict[str, Any]) -> None:
        print("\n=== HAR HUMAN-CENTRED ORBIT PLAN ===", flush=True)
        print(
            f"mode={'DRY-RUN (NO MOVEMENT)' if self.dry_run else 'LIVE'} "
            f"frame={plan['human_frame']} radius={plan['radius_m']:.3f} m",
            flush=True,
        )
        center = plan["human_center"]
        print(
            f"human center: x={center['x']:.3f} y={center['y']:.3f} "
            f"source={center['source']}",
            flush=True,
        )
        for target in plan["target_poses"]:
            print(
                f"slot {target['slot']} angle={target['angle_deg']:.1f}° -> "
                f"x={target['x']:.3f} y={target['y']:.3f} "
                f"yaw_to_human={target['yaw_deg']:.1f}°",
                flush=True,
            )
        for robot in self.robots:
            print(
                f"{robot} -> slot {plan['assignment'][robot]} "
                f"({len(plan['robot_waypoints'][robot])} sampled waypoints)",
                flush=True,
            )
            for index, waypoint in enumerate(plan["robot_waypoints"][robot], start=1):
                print(
                    f"  {index:02d} {waypoint['kind']:<10} "
                    f"x={waypoint['x']:.3f} y={waypoint['y']:.3f} "
                    f"yaw={math.degrees(waypoint['yaw_rad']):.1f}°",
                    flush=True,
                )
        for warning in plan["warnings"]:
            print(f"WARNING: {warning}", flush=True)
        if self.dry_run:
            print("NO NAV2 GOAL WAS SENT; FOLLOW REMAINS DISABLED.", flush=True)
        print("=====================================\n", flush=True)

    def _set_follow_all(self, enabled: bool) -> None:
        if self.dry_run:
            return
        for robot, client in self.follow_clients.items():
            if not client.service_is_ready():
                self._warn_throttled(f"follow service not ready: /{robot}/{self.follow_suffix}")
                continue
            request = SetBool.Request()
            request.data = bool(enabled)
            client.call_async(request)

    def _warn_throttled(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_warning_time >= 3.0:
            self.get_logger().warning(message)
            self.last_warning_time = now


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="dry_run", action="store_true")
    mode.add_argument("--live", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=None)
    parser.add_argument("--continuous", action="store_true", help="Do not stop after the first dry-run cycle")
    parser.add_argument("--ring-radius-m", type=float, default=None)
    parser.add_argument("--ring-angles-deg", default=None)
    parser.add_argument("--bootstrap-pair", default=None)
    parser.add_argument("--human-frame", default=None)
    parser.add_argument("--human-x", type=float, default=None, help="Optional dry-run/test map x override")
    parser.add_argument("--human-y", type=float, default=None, help="Optional dry-run/test map y override")
    parser.add_argument("--human-z", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if (args.human_x is None) != (args.human_y is None):
        raise ValueError("--human-x and --human-y must be supplied together")
    if args.ring_radius_m is not None and args.ring_radius_m <= 0.0:
        raise ValueError("--ring-radius-m must be positive")
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = load_config(config_path)
    rclpy.init(args=None)
    node = OrbitCoordinator(args, config)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._set_follow_all(False)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
