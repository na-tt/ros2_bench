"""
Stage 3 node: TensorRT invocation + detect-head decoding.

The node owns the engine (one node per engine: TensorRT execution contexts are
not thread-safe, and a single-threaded executor serialises the callbacks).

The stage boundary is *after* the decode/threshold step:
:func:`yolocar.inference.decode_predictions` +
:func:`yolocar.postprocessing.select_candidates` run here, and the node
publishes :class:`yolocar.postprocessing.Candidates` instead of the raw
``(1, 84, 8400)`` detect head. Serialising a ~2.8 MB raw head per frame would
dominate the measurement and is not how a real ROS 2 pipeline would split the
stages. Consequence: ``infer_ms`` here includes decode/threshold, so compare
``infer + post`` as one block against the monolith, not the two buckets
individually.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from yolocar.config import COCO_CLASS_NAMES, YoloCarConfig
from yolocar.inference import decode_predictions, first_output, infer
from yolocar.postprocessing import select_candidates
from yolocar.processor import YoloCarProcessor

from yolocar_interfaces.msg import CandidatesFrame, TensorFrame

from . import qos, topics
from .conversions import (
    candidates_frame,
    letterbox_from_message,
    tensor_from_frame,
)

LOGGER = logging.getLogger('yolocar')


class InferenceNode(Node):
    """Load the engine once, then turn each ``TensorFrame`` into candidates."""

    def __init__(self) -> None:
        super().__init__('yolocar_inference')

        model = str(
            self.declare_parameter(
                'model', 'assets/models/yolov8n_fp16.engine'
            ).value
        )
        device = int(self.declare_parameter('device', 0).value)
        image_size = int(self.declare_parameter('imgsz', 640).value)
        pinned = bool(self.declare_parameter('pinned', True).value)
        trt_log_level = str(
            self.declare_parameter('trt_log_level', 'warning').value
        )
        conf = float(self.declare_parameter('conf', 0.25).value)
        max_candidates = int(self.declare_parameter('max_candidates', 30_000).value)
        log_level = str(self.declare_parameter('log_level', 'INFO').value)
        LOGGER.setLevel(log_level.upper())

        self.config = YoloCarConfig(
            model_path=Path(model),
            device=device,
            image_size=image_size,
            pinned_host_memory=pinned,
            trt_log_level=trt_log_level,
            conf_threshold=conf,
            max_candidates=max_candidates,
            log_level=log_level,
        )
        self.processor = YoloCarProcessor(self.config)

        self.get_logger().info(f'loading engine {model}')
        t_load = time.monotonic()
        self.processor.load()
        self.engine_loaded_t = time.monotonic()
        self.get_logger().info(
            f'engine ready after {self.engine_loaded_t - t_load:.2f} s '
            f'(input={self.processor.input_spec.dtype} '
            f'{self.processor.input_spec.shape}, {len(COCO_CLASS_NAMES)} classes)'
        )

        self.publisher = self.create_publisher(
            CandidatesFrame, topics.CANDIDATES_TOPIC, qos.latest()
        )
        self.create_subscription(
            TensorFrame, topics.TENSOR_TOPIC, self._on_tensor, qos.latest()
        )

    def _on_tensor(self, msg: TensorFrame) -> None:
        t0 = time.monotonic()
        tensor = tensor_from_frame(msg)
        spec = self.processor.input_spec
        if tensor.dtype != spec.dtype or tensor.shape[2:] != spec.shape:
            raise ValueError(
                'the preprocessing node produced '
                f'{tensor.dtype} {tensor.shape}, but the engine expects '
                f'{spec.dtype} (1, 3, {spec.height}, {spec.width}); '
                'check the imgsz/input_dtype launch arguments'
            )

        outputs = infer(self.processor.engine, tensor)
        boxes_cxcywh, class_scores = decode_predictions(
            first_output(outputs), len(COCO_CLASS_NAMES)
        )
        candidates = select_candidates(
            boxes_cxcywh,
            class_scores,
            conf_threshold=self.config.conf_threshold,
            max_candidates=self.config.max_candidates,
        )
        t1 = time.monotonic()

        meta = msg.meta
        meta.infer_ms = (t1 - t0) * 1e3
        self.publisher.publish(
            candidates_frame(
                candidates,
                meta,
                letterbox_from_message(msg),
                engine_loaded_t=self.engine_loaded_t,
            )
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    rclpy.init(args=argv)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Expected when the launch shuts the chain down after a run.
        if rclpy.ok():
            node.get_logger().info('shutting down')
    finally:
        node.processor.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
