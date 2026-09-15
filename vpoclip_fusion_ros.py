#!/usr/bin/env python3
"""Fuse the two VPOCLIP prediction streams.

The recognizers publish complete candidate score vectors on their own robot
topics. This node never fuses images or hidden features: it applies the same
weighted-logit fusion used by the exported experiments to the latest result
from each robot. In continuous mode, a new fused result is published whenever
either robot publishes a new prediction; the orbit cycle topic is retained only
to attach an optional cycle ID and diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_PREDICTION_TOPICS = {
    "rosbot_1": "/rosbot_1/vpoclip/prediction",
    "rosbot_2": "/rosbot_2/vpoclip/prediction",
}


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class VPOCLIPFusionNode(Node):
    """Synchronize and fuse the two robot-level VPOCLIP JSON streams."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args.node_name)
        self.args = args
        self.current_cycle: int | None = None
        self.cycle_start_sec = 0.0
        self.cycle_deadline = 0.0
        self.cycle_payload: dict[str, Any] = {}
        self.predictions: dict[str, dict[str, Any]] = {}
        self.published_cycle: int | None = None
        self.last_fused_signature: tuple[Any, ...] | None = None

        self.publisher = self.create_publisher(String, args.output_topic, 10)
        self.create_subscription(
            String, args.trigger_topic, self._on_cycle_trigger, 10
        )
        for robot, topic in args.prediction_topics.items():
            self.create_subscription(
                String,
                topic,
                self._make_prediction_callback(robot),
                10,
            )
        self.timer = self.create_timer(0.1, self._tick)
        if args.continuous:
            self.get_logger().info(
                "VPOCLIP fusion running continuously on the latest two predictions; "
                f"cycle diagnostics={args.trigger_topic}"
            )
        else:
            self.get_logger().info(
                f"VPOCLIP fusion waiting for cycles on {args.trigger_topic}; "
                f"predictions={args.prediction_topics}"
            )

    def _on_cycle_trigger(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.get_logger().warning("invalid orbit recognition-cycle JSON")
            return
        if not isinstance(payload, dict):
            self.get_logger().warning("recognition-cycle payload must be a JSON object")
            return
        try:
            cycle = int(payload["cycle_id"])
        except (KeyError, TypeError, ValueError):
            self.get_logger().warning("recognition-cycle payload has no valid cycle_id")
            return

        self.current_cycle = cycle
        self.cycle_start_sec = finite(payload.get("start_sec"), time.time())
        self.cycle_deadline = time.monotonic() + max(1.0, self.args.timeout_sec)
        self.cycle_payload = payload
        self.published_cycle = None
        if not self.args.continuous:
            self.predictions = {}
            self.last_fused_signature = None
            self.get_logger().info(
                f"recognition cycle {cycle} opened; waiting for rosbot_1 + rosbot_2"
            )
            return

        self.get_logger().info(
            f"recognition cycle {cycle} tagged while continuous fusion remains active"
        )
        self._maybe_publish_fused_result("cycle_trigger")

    def _make_prediction_callback(self, robot: str):
        def callback(message: String) -> None:
            if not self.args.continuous and self.current_cycle is None:
                return
            try:
                payload = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                self.get_logger().warning(f"invalid {robot} VPOCLIP prediction JSON")
                return
            if not isinstance(payload, dict):
                return
            stamp = finite(payload.get("stamp_sec"), 0.0)
            if not self.args.continuous and stamp < self.cycle_start_sec:
                return
            self.predictions[robot] = payload
            if self.args.continuous:
                self._maybe_publish_fused_result(robot)

        return callback

    def _maybe_publish_fused_result(self, source_update: str) -> None:
        """Fuse the latest two events once per pair of source sequences."""

        if not all(robot in self.predictions for robot in self.args.prediction_topics):
            return
        signature = (
            int(self.current_cycle or 0),
            tuple(
                (
                    robot,
                    self.predictions[robot].get("sequence"),
                    self.predictions[robot].get("stamp_sec"),
                )
                for robot in self.args.prediction_topics
            ),
        )
        if signature == self.last_fused_signature:
            return
        self.last_fused_signature = signature
        self._publish_fused_result(source_update)

    @staticmethod
    def _score_vector(payload: dict[str, Any], field: str):
        labels = payload.get("candidate_labels")
        scores = payload.get(field)
        if not isinstance(labels, list) or not isinstance(scores, list):
            # Compatibility with a reduced event: top-k labels and scores are
            # still useful for diagnostics, but cannot be fused safely unless
            # both streams expose the same complete candidate bank.
            return {}
        if len(labels) != len(scores):
            return {}
        result = {}
        for label, score in zip(labels, scores):
            try:
                result[int(label)] = finite(score)
            except (TypeError, ValueError):
                continue
        return result

    def _tick(self) -> None:
        if self.args.continuous:
            return
        if self.current_cycle is None or self.published_cycle == self.current_cycle:
            return
        if all(robot in self.predictions for robot in self.args.prediction_topics):
            self._publish_fused_result("timer")
            return
        if time.monotonic() >= self.cycle_deadline:
            missing = [
                robot for robot in self.args.prediction_topics if robot not in self.predictions
            ]
            self.get_logger().warning(
                f"recognition cycle {self.current_cycle} timed out; missing {missing}"
            )
            self.published_cycle = self.current_cycle

    def _publish_fused_result(self, source_update: str) -> None:
        robot1 = self.predictions["rosbot_1"]
        robot2 = self.predictions["rosbot_2"]
        field = self.args.score_field
        scores1 = self._score_vector(robot1, field)
        scores2 = self._score_vector(robot2, field)
        common = sorted(set(scores1) & set(scores2))
        if not common:
            self.get_logger().error(
                "latest VPOCLIP predictions have no common candidate labels"
            )
            if not self.args.continuous:
                self.published_cycle = self.current_cycle
            return

        weight1 = max(0.0, float(self.args.robot1_weight))
        weight2 = max(0.0, float(self.args.robot2_weight))
        if weight1 + weight2 <= 0.0:
            raise ValueError("robot fusion weights must not both be zero")
        fused = {
            label: (weight1 * scores1[label] + weight2 * scores2[label])
            / (weight1 + weight2)
            for label in common
        }
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        top_label, top_score = ranked[0]
        uncertain = any(
            bool(payload.get("unknown")) or bool(payload.get("analysing"))
            for payload in (robot1, robot2)
        )
        decision = "ANALYSING" if uncertain else "class"
        cycle_id = int(self.current_cycle or 0)
        output = {
            "schema": "har.vpoclip.fused_action.v1",
            "cycle_id": cycle_id,
            "stamp_sec": time.time(),
            "fusion_mode": "continuous_latest" if self.args.continuous else "recognition_cycle",
            "source_update": source_update,
            "decision": decision,
            "action_label": int(top_label),
            "action_score": float(top_score),
            "ranking": [
                {"label": int(label), "score": float(score)}
                for label, score in ranked[: max(1, int(self.args.top_k))]
            ],
            "score_field": field,
            "weights": {
                "rosbot_1": float(weight1),
                "rosbot_2": float(weight2),
            },
            "source_sequences": {
                robot: int(payload.get("sequence", -1))
                for robot, payload in self.predictions.items()
            },
            "source_temporal_modes": {
                robot: payload.get("temporal_mode")
                for robot, payload in self.predictions.items()
            },
            # Preserve the exact rosbot_1 event used in this fused decision.
            # The next active-view trigger is intentionally based on this
            # stream only; rosbot_2 remains an input to fusion, not to policy
            # feedback.
            "robot1_prediction": robot1,
            "robot1_basis": self.cycle_payload.get("robot1_prediction"),
            "human_center": self.cycle_payload.get("human_center"),
            "targets": self.cycle_payload.get("targets"),
        }
        message = String()
        message.data = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        self.publisher.publish(message)
        if not self.args.continuous:
            self.published_cycle = self.current_cycle
        self.get_logger().info(
            f"cycle {cycle_id} fused ({source_update}): label={top_label} "
            f"score={top_score:.4f} decision={decision}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger-topic", default="/har/orbit/recognition_cycle")
    parser.add_argument(
        "--prediction-topic-1", default=DEFAULT_PREDICTION_TOPICS["rosbot_1"]
    )
    parser.add_argument(
        "--prediction-topic-2", default=DEFAULT_PREDICTION_TOPICS["rosbot_2"]
    )
    parser.add_argument("--output-topic", default="/har/final_action")
    parser.add_argument("--node-name", default="har_vpoclip_fusion")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Fuse the latest two prediction streams whenever either one updates.",
    )
    parser.add_argument("--robot1-weight", type=float, default=1.0)
    parser.add_argument("--robot2-weight", type=float, default=1.0)
    parser.add_argument(
        "--score-field",
        choices=["candidate_ranking_scores", "candidate_raw_scores", "candidate_scores"],
        default="candidate_ranking_scores",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for name in ("trigger_topic", "prediction_topic_1", "prediction_topic_2", "output_topic"):
        if not getattr(args, name).startswith("/"):
            raise ValueError(f"--{name.replace('_', '-')} must be absolute")
    if args.timeout_sec <= 0.0 or args.top_k <= 0:
        raise ValueError("--timeout-sec and --top-k must be positive")
    args.prediction_topics = {
        "rosbot_1": args.prediction_topic_1,
        "rosbot_2": args.prediction_topic_2,
    }
    rclpy.init(args=None)
    node = VPOCLIPFusionNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
