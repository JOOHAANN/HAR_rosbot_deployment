#!/usr/bin/env python3
"""Continuously display the latest human-centred orbit plan.

This is read-only: it subscribes to diagnostics and never publishes a goal or
velocity command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "orbit_system_config.csv"


def load_config(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row.get("key", "")).strip(): str(row.get("value", "")).strip()
            for row in rows
            if str(row.get("key", "")).strip()
        }


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def slot_target(plan: dict[str, Any], slot: Any) -> dict[str, Any] | None:
    try:
        wanted = int(slot)
    except (TypeError, ValueError):
        return None
    for target in plan.get("target_poses", []):
        if isinstance(target, dict) and int(target.get("slot", -1)) == wanted:
            return target
    return None


class OrbitTargetMonitor(Node):
    def __init__(self, config: dict[str, str], clear_screen: bool) -> None:
        super().__init__("har_orbit_target_monitor")
        self.clear_screen = clear_screen
        self.plan: dict[str, Any] | None = None
        self.state: dict[str, Any] | None = None
        self.last_plan_received = 0.0
        self.last_rendered = 0.0

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        plan_topic = config.get("plan_topic", "/har/orbit/plan")
        state_topic = config.get("state_topic", "/har/orbit/state")
        self.create_subscription(String, plan_topic, self._on_plan, qos)
        self.create_subscription(String, state_topic, self._on_state, qos)
        self.create_timer(1.0, self._on_timer)
        self._render()

    @staticmethod
    def _decode(message: String) -> dict[str, Any] | None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _on_plan(self, message: String) -> None:
        payload = self._decode(message)
        if payload is not None:
            self.plan = payload
            self.last_plan_received = time.time()
            self._render()

    def _on_state(self, message: String) -> None:
        payload = self._decode(message)
        if payload is not None:
            self.state = payload
            self._render()

    def _on_timer(self) -> None:
        self._render()

    def _render(self) -> None:
        now = time.time()
        if now - self.last_rendered < 0.3:
            return
        self.last_rendered = now

        lines: list[str] = []
        if self.clear_screen and sys.stdout.isatty():
            lines.append("\033[2J\033[H")
        lines.append("=" * 68)
        lines.append("HAR ORBIT TARGET MONITOR  (READ-ONLY)")
        lines.append("按 Ctrl-C 退出监视器；不会停止机器人系统。")

        state_name = (self.state or {}).get("state", "等待状态")
        lines.append(f"state: {state_name}")
        if self.plan is None:
            lines.append("\n等待 /har/orbit/plan ……")
            lines.append("请确认有人在相机中，并保持 orbit coordinator 正在运行。")
            print("\n".join(lines), flush=True)
            return

        plan = self.plan
        dry_run = bool(plan.get("dry_run", True))
        mode = "DRY-RUN（不移动）" if dry_run else "LIVE（会移动）"
        age = max(0.0, now - self.last_plan_received) if self.last_plan_received else 0.0
        center = plan.get("human_center") or {}
        lines.append(f"mode: {mode}    motion_cycle: {plan.get('motion_cycle', '?')}    plan_age: {age:.1f}s")
        lines.append(
            "human center (map): "
            f"x={number(center.get('x')):.3f}  y={number(center.get('y')):.3f}  "
            f"source={center.get('source', '?')}"
        )
        lines.append(f"ring radius: {number(plan.get('radius_m')):.3f} m")

        lines.append("\nFour ring candidates:")
        for target in plan.get("target_poses", []):
            if not isinstance(target, dict):
                continue
            lines.append(
                f"  slot {int(target.get('slot', -1))}: "
                f"angle={number(target.get('angle_deg')):.1f}°  "
                f"x={number(target.get('x')):.3f}  y={number(target.get('y')):.3f}  "
                f"yaw={number(target.get('yaw_deg')):.1f}°"
            )

        lines.append("\nRobot targets:")
        assignment = plan.get("assignment") or {}
        waypoints = plan.get("robot_waypoints") or {}
        for robot in ("rosbot_1", "rosbot_2"):
            slot = assignment.get(robot, "?")
            target = slot_target(plan, slot)
            if target is None:
                lines.append(f"  {robot}: slot {slot} (target unavailable)")
                continue
            lines.append(
                f"  {robot} -> slot {slot}: "
                f"x={number(target.get('x')):.3f}  y={number(target.get('y')):.3f}  "
                f"yaw={number(target.get('yaw_deg')):.1f}°"
            )
            robot_points = waypoints.get(robot, [])
            if robot_points:
                lines.append(f"    planned waypoints: {len(robot_points)}")
                for index, point in enumerate(robot_points, start=1):
                    if not isinstance(point, dict):
                        continue
                    yaw = math.degrees(number(point.get("yaw_rad")))
                    lines.append(
                        f"      {index:02d} {str(point.get('kind', 'point')):<10} "
                        f"x={number(point.get('x')):.3f}  y={number(point.get('y')):.3f}  "
                        f"yaw={yaw:.1f}°"
                    )

        lines.append("\n" + ("NO NAV2 GOAL SENT" if dry_run else "NAV2 GOAL MODE ACTIVE"))
        lines.append("=" * 68)
        print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--no-clear", action="store_true", help="append updates instead of refreshing the terminal")
    args = parser.parse_args()
    config = load_config(args.config.expanduser().resolve())
    rclpy.init(args=None)
    node = OrbitTargetMonitor(config, clear_screen=not args.no_clear)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
