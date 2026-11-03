"""
Stage 2 node: image conditioning at source resolution.

All image logic is :func:`yolocar.preprocessing.condition`; this node
deserialises the frame, applies the configured chain and publishes the result
as another ``ImageFrame`` -- same geometry, still BGR uint8, so this stage can
be split into more nodes without changing the wire type.

The chain is data, not code: ``pre_ops:='clahe;bilateral:d=9'`` (see
:func:`yolocar.preprocessing.parse`). Empty means identity: the node then
forwards the decoded frame untouched, which is what every measurement taken
before this stage existed recorded.

The node takes its topics as parameters (``in_topic``/``out_topic``) and the
launch file uses that to instantiate it more than once: with ``pre_split`` the
chain is cut into one node per op, wired head to tail, and ``pre_ms``/the op
arrays are accumulated additively along the way. No node code knows whether it
is the only stage-2 node or the second of five -- the partition is launch data.

The engine's input contract -- letterbox, colour order, NCHW, dtype -- is *not*
applied here. It belongs to the inference node, which owns the engine and can
therefore read the real ``InputSpec`` instead of taking ``imgsz`` and
``input_dtype`` on trust from a launch argument.
"""

from __future__ import annotations

import logging
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from yolocar.preprocessing import condition, describe, parse

from yolocar_interfaces.msg import ImageFrame

from . import qos, topics
from .conversions import frame_from_image, image_frame

LOGGER = logging.getLogger('yolocar')


class PreprocessingNode(Node):
    """Turn each ``ImageFrame`` into a conditioned ``ImageFrame``."""

    def __init__(self) -> None:
        super().__init__('yolocar_preprocessing')

        spec = str(self.declare_parameter('pre_ops', '').value)
        self.in_topic = str(self.declare_parameter('in_topic', topics.FRAME_TOPIC).value)
        self.out_topic = str(
            self.declare_parameter('out_topic', topics.INPUT_TOPIC).value
        )
        log_level = str(self.declare_parameter('log_level', 'INFO').value)
        LOGGER.setLevel(log_level.upper())

        self.chain = parse(spec)
        self.get_logger().info(
            f'preprocessing [{self.in_topic} -> {self.out_topic}]: '
            + (', '.join(describe(self.chain)) or 'no conditioning ops (identity)')
        )

        self.publisher = self.create_publisher(
            ImageFrame, self.out_topic, qos.latest()
        )
        self.create_subscription(
            ImageFrame, self.in_topic, self._on_frame, qos.latest()
        )

    def _on_frame(self, msg: ImageFrame) -> None:
        t0 = time.monotonic()
        frame = frame_from_image(msg)
        conditioned, op_ms = condition(frame, self.chain)
        t1 = time.monotonic()

        meta = msg.meta
        # Additive, not overwritten: when the stage is split over several
        # nodes each one adds its own share, so pre_ms is the whole stage
        # either way and the two configurations stay comparable.
        meta.pre_ms = float(meta.pre_ms) + (t1 - t0) * 1e3
        # Parallel arrays instead of a dict: fixed IDL, no map type, and the
        # terminal node rebuilds the per-op map the records contract expects.
        meta.pre_op_names = list(meta.pre_op_names) + list(op_ms)
        meta.pre_op_ms = list(meta.pre_op_ms) + list(op_ms.values())
        self.publisher.publish(image_frame(conditioned, meta))


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
