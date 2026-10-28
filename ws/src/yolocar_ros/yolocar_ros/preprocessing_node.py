"""
Stage 2 node: letterbox + BGR->RGB NCHW tensor.

All image logic is :func:`yolocar.preprocessing.preprocess`; this node only
deserialises the frame, calls it, and publishes the tensor together with the
letterbox geometry the coordinate-adaptation stage will need later.

The engine's input dtype cannot be queried without loading the engine (which
is the inference node's job), so the launch file passes ``imgsz`` and
``input_dtype`` here. The inference node validates them against the real engine
contract and fails loudly on a mismatch instead of silently casting.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from yolocar.preprocessing import preprocess

from yolocar_interfaces.msg import ImageFrame, TensorFrame

from . import qos, topics
from .conversions import frame_from_image, tensor_frame

LOGGER = logging.getLogger('yolocar')


class PreprocessingNode(Node):
    """Turn each ``ImageFrame`` into a ``TensorFrame``."""

    def __init__(self) -> None:
        super().__init__('yolocar_preprocessing')

        self.image_size = int(self.declare_parameter('imgsz', 640).value)
        self.input_dtype = np.dtype(
            str(self.declare_parameter('input_dtype', 'float32').value)
        )
        self.pad_value = int(self.declare_parameter('pad_value', 114).value)
        log_level = str(self.declare_parameter('log_level', 'INFO').value)
        LOGGER.setLevel(log_level.upper())

        self.publisher = self.create_publisher(
            TensorFrame, topics.TENSOR_TOPIC, qos.latest()
        )
        self.create_subscription(
            ImageFrame, topics.FRAME_TOPIC, self._on_frame, qos.latest()
        )
        self.get_logger().info(
            f'preprocessing: {self.image_size}x{self.image_size} -> '
            f'(1, 3, {self.image_size}, {self.image_size}) {self.input_dtype}'
        )

    def _on_frame(self, msg: ImageFrame) -> None:
        t0 = time.monotonic()
        frame = frame_from_image(msg)
        tensor, letterbox = preprocess(
            frame,
            (self.image_size, self.image_size),
            self.input_dtype,
            pad_value=self.pad_value,
        )
        t1 = time.monotonic()

        meta = msg.meta
        meta.pre_ms = (t1 - t0) * 1e3
        self.publisher.publish(tensor_frame(tensor, meta, letterbox))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    rclpy.init(args=argv)
    node = PreprocessingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Expected when the launch shuts the chain down after a run.
        if rclpy.ok():
            node.get_logger().info('shutting down')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
