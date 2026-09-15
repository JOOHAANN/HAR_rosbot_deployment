#!/usr/bin/env python3
"""Publish deterministic synthetic JPEG frames for an offline ROS smoke test."""

import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


class SyntheticCamera(Node):
    def __init__(self, topics, rate_hz, duration, width, height):
        super().__init__("har_synthetic_camera")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._publishers = [self.create_publisher(CompressedImage, topic, qos) for topic in topics]
        self.period = 1.0 / rate_hz
        self.duration = duration
        self.width = width
        self.height = height
        self.started = time.monotonic()
        self.frame_index = 0
        self.timer = self.create_timer(self.period, self.publish_frame)

    def publish_frame(self):
        elapsed = time.monotonic() - self.started
        if elapsed >= self.duration:
            rclpy.shutdown()
            return

        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        phase = self.frame_index % max(self.width, 1)
        image[:, :, 1] = 32
        cv2.rectangle(
            image,
            (max(0, phase - 80), self.height // 8),
            (min(self.width - 1, phase + 80), self.height * 7 // 8),
            (80, 140, 210),
            -1,
        )
        cv2.circle(image, (phase, self.height // 4), max(8, self.height // 12), (180, 200, 220), -1)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise RuntimeError("OpenCV could not encode synthetic image")
        message = CompressedImage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.format = "jpeg"
        message.data = encoded.tobytes()
        for publisher in self._publishers:
            publisher.publish(message)
        self.frame_index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topics",
        default="/rosbot_1/camera/rgb/image_raw/compressed,/rosbot_2/camera/rgb/image_raw/compressed",
    )
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    topics = [item.strip() for item in args.topics.split(",") if item.strip()]
    if not topics or any(not topic.startswith("/") for topic in topics):
        raise ValueError("--topics must be a comma-separated list of absolute ROS topics")
    if args.rate_hz <= 0 or args.duration <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("rate, duration, width, and height must be positive")

    rclpy.init(args=None)
    node = SyntheticCamera(topics, args.rate_hz, args.duration, args.width, args.height)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
