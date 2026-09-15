#!/usr/bin/env python3
"""Coordinate two ROSbots on a human-centred four-position orbit.

The default mode is intentionally non-actuating. In ``dry_run`` the node
computes and prints map-frame targets and sampled arc waypoints, publishes the
same diagnostics for RViz/monitoring, and never creates or sends a Nav2 goal.
Live mode keeps the same plan but sends one ``NavigateToPose`` goal at a time:
rosbot_1 finishes before rosbot_2 starts. View-only person following is enabled
while stationary and disabled while either robot is being moved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped, Quaternion
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

try:  # The separate jazzy-rosbot container owns Nav2 interfaces.
    from nav2_msgs.action import NavigateToPose
except ImportError:  # Dry-run can run in the lighter HAR/ML container.
    NavigateToPose = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "orbit_system_config.csv"
NUM_SLOTS = 4


@dataclass
class HumanEstimate:
    x: float
    y: float
    z: float
    yaw: float
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
        angle_text = args.ring_angles_deg or ";".join(
            config_list(config, "ring_angles_deg", ["270", "0", "90", "180"])
        )
        raw_angles_deg = parse_angles(angle_text)
        try:
            self.human_front_slot = int(config.get("human_front_slot", "1"))
        except (TypeError, ValueError) as exc:
            raise ValueError("human_front_slot must be an integer in 0..3") from exc
        if not 0 <= self.human_front_slot < NUM_SLOTS:
            raise ValueError("human_front_slot must be an integer in 0..3")
        # Slot angles are expressed relative to an arbitrary ring reference.
        # Re-reference them so human_front_slot is always exactly 0 degrees
        # from the human's forward direction. This keeps old 0;90;180;270
        # tables usable while enforcing the slot-1 front convention.
        raw_angles_rad = [math.radians(value) for value in raw_angles_deg]
        front_reference = raw_angles_rad[self.human_front_slot]
        self.angles_rad = [wrap_angle(value - front_reference) for value in raw_angles_rad]
        self.angles_deg = [math.degrees(value) for value in self.angles_rad]
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
        self.human_position_mode = config.get("human_position_mode", "fixed").strip().lower()
        if self.human_position_mode not in {"fixed", "fixed_map"}:
            raise ValueError(
                "human_position_mode must be fixed; this deployment does not use depth-based human positioning"
            )
        self.fixed_human_x = config_float(config, "human_x_m", 0.0)
        self.fixed_human_y = config_float(config, "human_y_m", 0.0)
        self.fixed_human_z = config_float(config, "human_z_m", 0.0)
        self.fixed_human_yaw_deg = config_float(config, "human_yaw_deg", 0.0)
        self.fixed_human_yaw_rad = math.radians(self.fixed_human_yaw_deg)
        if not all(math.isfinite(value) for value in (
            self.fixed_human_x,
            self.fixed_human_y,
            self.fixed_human_z,
            self.fixed_human_yaw_deg,
        )):
            raise ValueError(
                "human_x_m, human_y_m, human_z_m and human_yaw_deg must be finite numbers"
            )
        self.runtime_human_pose: HumanEstimate | None = None
        self.persist_rviz_human_pose = config_bool(
            config, "persist_rviz_human_pose", True
        )
        self.target_tolerance = config_float(config, "target_tolerance_m", 0.15)
        self.arc_step_rad = math.radians(config_float(config, "arc_step_deg", 15.0))
        self.allow_radial_acquire = config_bool(config, "allow_radial_acquire", True)
        self.min_robot_separation = config_float(config, "min_robot_separation_m", 0.8)
        self.settle_seconds = config_float(config, "settle_seconds", 3.0)
        self.recognition_timeout = config_float(config, "recognition_timeout_sec", 10.0)
        self.initial_selection_timeout = config_float(
            config, "initial_selection_timeout_sec", 8.0
        )
        self.base_suffix = config.get("base_frame_suffix", "base_link")
        self.follow_suffix = config.get(
            "follow_service_suffix", "person_follow_controller/set_enabled"
        )
        self.follow_when_stationary = config_bool(
            config, "follow_when_stationary", True
        )

        self.prediction_topics = [
            config.get("prediction_topic_1", "/rosbot_1/vpoclip/prediction"),
            config.get("prediction_topic_2", "/rosbot_2/vpoclip/prediction"),
        ]

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
        self.human_marker_topic = config.get(
            "human_marker_topic", "/har/orbit/human_center_marker"
        )
        self.human_heading_marker_topic = config.get(
            "human_heading_marker_topic", "/har/orbit/human_heading_marker"
        )
        self.human_pose_topic = config.get(
            "human_pose_topic", "/har/orbit/human_pose"
        )
        self.initial_robot_pose_topic = config.get(
            "initial_robot_pose_topic", "/har/orbit/initial_robot_poses"
        )
        self.initial_robot_marker_topic = config.get(
            "initial_robot_marker_topic", "/har/orbit/initial_robot_markers"
        )
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
        self.human_marker_publisher = self.create_publisher(
            Marker, self.human_marker_topic, diagnostic_qos
        )
        self.human_heading_marker_publisher = self.create_publisher(
            Marker, self.human_heading_marker_topic, diagnostic_qos
        )
        self.initial_robot_pose_publisher = self.create_publisher(
            PoseArray, self.initial_robot_pose_topic, diagnostic_qos
        )
        self.initial_robot_marker_publisher = self.create_publisher(
            MarkerArray, self.initial_robot_marker_topic, diagnostic_qos
        )
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
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.human_pose_topic,
            self._on_human_pose,
            10,
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
        self.follow_desired: bool | None = None
        self.follow_last_sync_time = 0.0
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

        initial_human = self._estimate_human()
        self._publish_human_marker(initial_human)
        self._publish_initial_robot_poses(initial_human)
        self._publish_slot_mapping()
        self._sync_follow_state(force=True)
        self.timer = self.create_timer(0.2, self._tick)
        self.get_logger().info(
            f"orbit coordinator: dry_run={self.dry_run} radius={self.radius:.2f}m "
            f"angles={self.angles_deg} robots={self.robots} "
            f"human_center=({self.fixed_human_x:.3f},{self.fixed_human_y:.3f}) "
            f"human_yaw={self.fixed_human_yaw_deg:.1f}° front_slot={self.human_front_slot} "
            f"frame={self.human_frame} source=fixed_map_config"
        )
        if self.dry_run:
            self.get_logger().warning(
                "DRY-RUN active: target coordinates are printed and no Nav2 goal will be sent; "
                "stationary view-follow may still be enabled"
            )

    # ------------------------------------------------------------------ ROS input

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

    def _on_human_pose(self, message: PoseWithCovarianceStamped) -> None:
        frame = str(message.header.frame_id).strip()
        if frame and frame != self.human_frame:
            self._warn_throttled(
                f"ignoring online human pose in frame {frame!r}; expected {self.human_frame!r}"
            )
            return
        pose = message.pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        z = float(pose.position.z)
        yaw = quaternion_to_yaw(pose.orientation)
        if not all(math.isfinite(value) for value in (x, y, z, yaw)):
            self._warn_throttled("ignoring online human pose with non-finite values")
            return

        human = HumanEstimate(x, y, z, yaw, "rviz_online", time.time())
        self.runtime_human_pose = human
        self.fixed_human_x = x
        self.fixed_human_y = y
        self.fixed_human_z = z
        self.fixed_human_yaw_rad = yaw
        self.fixed_human_yaw_deg = math.degrees(yaw)
        if self.persist_rviz_human_pose:
            self._persist_human_pose(human)
        self._publish_human_marker(human)
        self._publish_initial_robot_poses(human)

        # A pose edited before the first plan is immediately used. During a
        # live move the active plan is kept unchanged for safety; the new pose
        # becomes the basis of the next plan. In dry-run, republish the plan so
        # RViz can be used interactively to tune the geometry.
        if self.dry_run and self.active_plan is not None and self.state.startswith("DRY_RUN_"):
            assignment = dict(self.plan_target_slots)
            targets = [assignment[robot] for robot in self.robots]
            self.active_plan = self._build_plan(human, targets, assignment, "rviz_online")
            self._publish_plan(self.active_plan)
            self._print_plan(self.active_plan)
        elif self.state in {"WAIT_HUMAN", "WAIT_INITIAL_SELECTION", "WAIT_NEXT_SELECTION"}:
            self._publish_state(human)
        else:
            self.get_logger().warning(
                f"online human pose updated to ({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f}°); "
                "current active plan is unchanged until the next cycle"
            )

    def _persist_human_pose(self, human: HumanEstimate) -> None:
        config_path = Path(getattr(self.args, "config_path", DEFAULT_CONFIG)).expanduser()
        temporary_path = None
        try:
            original_stat = config_path.stat()
            original_mode = original_stat.st_mode & 0o777
            with config_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or ["key", "value", "description"])
                rows = list(reader)
            values = {
                "human_x_m": f"{human.x:.6f}",
                "human_y_m": f"{human.y:.6f}",
                "human_z_m": f"{human.z:.6f}",
                "human_yaw_deg": f"{math.degrees(human.yaw):.6f}",
            }
            found = set()
            for row in rows:
                key = str(row.get("key", "")).strip()
                if key in values:
                    row["value"] = values[key]
                    found.add(key)
            missing = set(values) - found
            if missing:
                raise ValueError(f"missing config keys: {', '.join(sorted(missing))}")
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=str(config_path.parent),
                prefix=f".{config_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            # The coordinator normally runs as root inside Docker while the
            # project is owned by the desktop user. Preserve the original
            # mode/owner so an RViz edit never makes the CSV unreadable from
            # the host or from a later non-root process.
            os.chmod(temporary_path, original_mode)
            if hasattr(os, "chown") and os.geteuid() == 0:
                os.chown(temporary_path, original_stat.st_uid, original_stat.st_gid)
            os.replace(temporary_path, config_path)
        except Exception as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._warn_throttled(f"could not persist online human pose to {config_path}: {exc}")

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

    # --------------------------------------------------------------- fixed human point
    def _estimate_human(self) -> HumanEstimate:
        if self.runtime_human_pose is not None:
            return self.runtime_human_pose
        if self.args.human_x is not None and self.args.human_y is not None:
            human_yaw_deg = (
                self.args.human_yaw_deg
                if self.args.human_yaw_deg is not None
                else self.fixed_human_yaw_deg
            )
            return HumanEstimate(
                float(self.args.human_x),
                float(self.args.human_y),
                float(self.args.human_z),
                math.radians(float(human_yaw_deg)),
                "cli_override",
                time.time(),
            )
        return HumanEstimate(
            self.fixed_human_x,
            self.fixed_human_y,
            self.fixed_human_z,
            self.fixed_human_yaw_rad,
            "fixed_map_config",
            time.time(),
        )

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

    def _slot_bearing(self, human: HumanEstimate, slot: int) -> float:
        """Return a slot's absolute map bearing from the human pose."""
        return wrap_angle(human.yaw + self.angles_rad[int(slot)])

    def _initial_robot_pose_dict(
        self, human: HumanEstimate, robot: str, slot: int
    ) -> dict[str, Any]:
        """Return the virtual initial pose for a robot's configured slot.

        This is deliberately a diagnostic/initialization pose. It is not a TF
        override, so moving the human marker in RViz cannot move a real robot.
        """
        target = self._target_pose_dict(human, slot)
        return {
            "robot": robot,
            "slot": int(slot),
            "x": float(target["x"]),
            "y": float(target["y"]),
            "yaw_rad": float(target["yaw_rad"]),
            "yaw_deg": float(target["yaw_deg"]),
        }

    def _ring_point(self, human: HumanEstimate, slot: int) -> tuple[float, float]:
        angle = self._slot_bearing(human, slot)
        return (
            human.x + self.radius * math.cos(angle),
            human.y + self.radius * math.sin(angle),
        )

    def _target_pose_dict(self, human: HumanEstimate, slot: int) -> dict[str, Any]:
        x, y = self._ring_point(human, slot)
        yaw = math.atan2(human.y - y, human.x - x)
        relative_angle = self.angles_rad[int(slot)]
        map_bearing = self._slot_bearing(human, slot)
        return {
            "slot": int(slot),
            "angle_deg": float(math.degrees(map_bearing)),
            "relative_angle_deg": float(self.angles_deg[int(slot)]),
            "map_bearing_deg": float(math.degrees(map_bearing)),
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
            start_angle = self._slot_bearing(
                human, self.robot_slots.get(robot, 0)
            )
            current_radius = self.radius
            warnings.append(f"{robot}: current TF unavailable; arc start angle inferred from slot")
        else:
            start_angle = math.atan2(current.y - human.y, current.x - human.x)
            current_radius = math.hypot(current.x - human.x, current.y - human.y)

        target_angle = self._slot_bearing(human, target_slot)
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

        initial_robot_poses = {
            robot: self._initial_robot_pose_dict(
                human, robot, self.initial_robot_slots[index]
            )
            for index, robot in enumerate(self.robots)
        }

        return {
            "schema": "har.orbit.plan.v1",
            "motion_cycle": int(self.motion_cycle),
            "dry_run": bool(self.dry_run),
            "human_frame": self.human_frame,
            "human_center": {
                "x": float(human.x),
                "y": float(human.y),
                "z": float(human.z),
                "yaw_rad": float(human.yaw),
                "yaw_deg": float(math.degrees(human.yaw)),
                "source": human.source,
                "stamp_sec": float(human.stamp_sec),
            },
            "radius_m": float(self.radius),
            "ring_angles_deg": [float(value) for value in self.angles_deg],
            "human_front_slot": int(self.human_front_slot),
            "target_slots": [int(value) for value in target_slots],
            "assignment": {robot: int(assignment[robot]) for robot in self.robots},
            "selection_source": selection_source,
            "robot_slots_before": dict(self.robot_slots),
            "initial_robot_slots": {
                robot: int(self.initial_robot_slots[index])
                for index, robot in enumerate(self.robots)
            },
            "initial_robot_poses": initial_robot_poses,
            "target_poses": target_poses,
            "robot_waypoints": robot_waypoints,
            "warnings": warnings,
        }

    # ------------------------------------------------------------- state machine
    def _tick(self) -> None:
        now = time.monotonic()
        self._sync_follow_state(now=now)
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

        if self.dry_run:
            self.state = "DRY_RUN_ROSBOT_1"
            self.dry_stage_deadline = time.monotonic() + 0.8
            # No physical base is moving in dry-run, so the stationary
            # view-follow policy remains enabled.
            self._sync_follow_state(force=True)
            self._publish_state(human)
            return

        self.nav_robot = self.robots[0]
        self.nav_waypoint_index = 0
        self.nav_goal_handle = None
        self.nav_request_in_flight = False
        self.state = "MOVING_ROSBOT_1"
        # Set the moving state before disabling follow, so the diagnostic
        # state and the service request describe the same safety mode.
        self._set_follow_all(False)
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
                "yaw_rad": float(human.yaw),
                "yaw_deg": float(math.degrees(human.yaw)),
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
        self._sync_follow_state(force=True)
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
        self._publish_human_marker(
            HumanEstimate(
                float(plan["human_center"]["x"]),
                float(plan["human_center"]["y"]),
                float(plan["human_center"].get("z", 0.0)),
                float(
                    plan["human_center"].get(
                        "yaw_rad", math.radians(plan["human_center"].get("yaw_deg", 0.0))
                    )
                ),
                str(plan["human_center"].get("source", "fixed_map_config")),
                float(plan["human_center"].get("stamp_sec", time.time())),
            )
        )
        self._publish_initial_robot_poses(
            HumanEstimate(
                float(plan["human_center"]["x"]),
                float(plan["human_center"]["y"]),
                float(plan["human_center"].get("z", 0.0)),
                float(
                    plan["human_center"].get(
                        "yaw_rad", math.radians(plan["human_center"].get("yaw_deg", 0.0))
                    )
                ),
                str(plan["human_center"].get("source", "fixed_map_config")),
                float(plan["human_center"].get("stamp_sec", time.time())),
            )
        )

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

    def _publish_human_marker(self, human: HumanEstimate) -> None:
        stamp = self.get_clock().now().to_msg()

        center_marker = Marker()
        center_marker.header.frame_id = self.human_frame
        center_marker.header.stamp = stamp
        center_marker.ns = "har_orbit"
        center_marker.id = 0
        center_marker.type = Marker.SPHERE
        center_marker.action = Marker.ADD
        center_marker.pose.position.x = float(human.x)
        center_marker.pose.position.y = float(human.y)
        center_marker.pose.position.z = max(0.18, float(human.z))
        center_marker.pose.orientation.w = 1.0
        center_marker.scale.x = 0.35
        center_marker.scale.y = 0.35
        center_marker.scale.z = 0.35
        center_marker.color.r = 1.0
        center_marker.color.g = 0.15
        center_marker.color.b = 0.05
        center_marker.color.a = 0.95
        self.human_marker_publisher.publish(center_marker)

        heading_marker = Marker()
        heading_marker.header.frame_id = self.human_frame
        heading_marker.header.stamp = stamp
        heading_marker.ns = "har_orbit"
        heading_marker.id = 1
        heading_marker.type = Marker.ARROW
        heading_marker.action = Marker.ADD
        heading_marker.pose.position.x = float(human.x)
        heading_marker.pose.position.y = float(human.y)
        heading_marker.pose.position.z = max(0.22, float(human.z) + 0.04)
        heading_marker.pose.orientation = yaw_quaternion(human.yaw)
        heading_marker.scale.x = 0.8
        heading_marker.scale.y = 0.10
        heading_marker.scale.z = 0.10
        heading_marker.color.r = 1.0
        heading_marker.color.g = 0.85
        heading_marker.color.b = 0.05
        heading_marker.color.a = 0.95
        self.human_heading_marker_publisher.publish(heading_marker)

    def _publish_initial_robot_poses(self, human: HumanEstimate) -> None:
        """Publish virtual robot poses that follow the edited human pose.

        RViz's normal RobotModel is driven by real TF and must remain untouched.
        These two topics are a separate, clearly labelled visualization of the
        configured slot-0/slot-1 initial positions. They are also the exact
        poses used by ``start_nav.sh --restart`` in relative mode.
        """
        pose_array = PoseArray()
        pose_array.header.frame_id = self.human_frame
        pose_array.header.stamp = self.get_clock().now().to_msg()

        marker_array = MarkerArray()
        colors = ((0.15, 0.85, 1.0), (0.35, 1.0, 0.25))
        for index, robot in enumerate(self.robots):
            slot = self.initial_robot_slots[index]
            pose_data = self._initial_robot_pose_dict(human, robot, slot)

            pose = Pose()
            pose.position.x = float(pose_data["x"])
            pose.position.y = float(pose_data["y"])
            pose.position.z = 0.0
            pose.orientation = yaw_quaternion(float(pose_data["yaw_rad"]))
            pose_array.poses.append(pose)

            arrow = Marker()
            arrow.header = pose_array.header
            arrow.ns = "har_orbit_initial_robots"
            arrow.id = index
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = float(pose_data["x"])
            arrow.pose.position.y = float(pose_data["y"])
            arrow.pose.position.z = 0.12
            arrow.pose.orientation = yaw_quaternion(float(pose_data["yaw_rad"]))
            arrow.scale.x = 0.70
            arrow.scale.y = 0.16
            arrow.scale.z = 0.16
            arrow.color.r = colors[index][0]
            arrow.color.g = colors[index][1]
            arrow.color.b = colors[index][2]
            arrow.color.a = 0.95
            marker_array.markers.append(arrow)

            label = Marker()
            label.header = pose_array.header
            label.ns = "har_orbit_initial_robot_labels"
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(pose_data["x"])
            label.pose.position.y = float(pose_data["y"])
            label.pose.position.z = 0.45
            label.pose.orientation.w = 1.0
            label.scale.z = 0.20
            label.color.r = colors[index][0]
            label.color.g = colors[index][1]
            label.color.b = colors[index][2]
            label.color.a = 1.0
            label.text = f"{robot}  slot {slot}"
            marker_array.markers.append(label)

        self.initial_robot_pose_publisher.publish(pose_array)
        self.initial_robot_marker_publisher.publish(marker_array)

    def _publish_state(self, human: HumanEstimate | None = None) -> None:
        payload = {
            "schema": "har.orbit.state.v1",
            "state": self.state,
            "dry_run": bool(self.dry_run),
            "follow_policy": "stationary_view_yaw_only",
            "follow_when_stationary": bool(self.follow_when_stationary),
            "follow_desired": self.follow_desired,
            "follow_linear_x_mps": 0.0,
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
                "yaw_rad": float(human.yaw),
                "yaw_deg": float(math.degrees(human.yaw)),
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
            f"yaw={center.get('yaw_deg', 0.0):.1f}° source={center['source']}",
            flush=True,
        )
        for target in plan["target_poses"]:
            print(
                f"slot {target['slot']} relative={target.get('relative_angle_deg', target['angle_deg']):.1f}° "
                f"map_bearing={target.get('map_bearing_deg', target['angle_deg']):.1f}° -> "
                f"x={target['x']:.3f} y={target['y']:.3f} "
                f"yaw_to_human={target['yaw_deg']:.1f}°",
                flush=True,
            )
        print("virtual initial robot poses (follow human in RViz):", flush=True)
        for robot, pose in plan.get("initial_robot_poses", {}).items():
            print(
                f"  {robot} -> slot {pose['slot']} x={pose['x']:.3f} y={pose['y']:.3f} "
                f"yaw_to_human={pose['yaw_deg']:.1f}°",
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
            print(
                "NO NAV2 GOAL WAS SENT; FOLLOW IS VIEW-ONLY (linear.x=0).",
                flush=True,
            )
        print("=====================================\n", flush=True)

    def _sync_follow_state(
        self, now: float | None = None, force: bool = False
    ) -> None:
        """Keep view-only following enabled whenever Nav2 is not moving.

        The follow controller itself only publishes ``angular.z`` with
        ``linear.x = 0``. The arbiter still owns the final ``cmd_vel`` and
        gives Nav2 priority whenever a navigation action is active.
        """
        now = time.monotonic() if now is None else float(now)
        moving = self.state in {"MOVING_ROSBOT_1", "MOVING_ROSBOT_2"}
        desired = bool(self.follow_when_stationary and not moving and self.state != "ERROR")
        if (
            not force
            and desired == self.follow_desired
            and now - self.follow_last_sync_time < 2.0
        ):
            return
        self.follow_desired = desired
        self.follow_last_sync_time = now
        self._set_follow_all(desired)

    def _set_follow_all(self, enabled: bool) -> None:
        self.follow_desired = bool(enabled)
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
    parser.add_argument(
        "--human-yaw-deg",
        type=float,
        default=None,
        help="Optional map-frame human heading override in degrees",
    )
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
    args.config_path = config_path
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
