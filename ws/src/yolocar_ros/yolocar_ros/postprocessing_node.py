"""
Stages 4-6 node: NMS -> domain filter -> coordinate adaptation.

Terminal node of the chain: it writes the benchmark's per-frame
``records.jsonl``, acknowledges each frame back to the acquisition node (which
is what paces the graph), and writes ``summary.json`` when the acquisition node
signals that the run is over.

Stage order is the canonical one from ``AGENTS.md`` and the monolith: NMS over
all classes, then the road-class filter, then the inverse letterbox. The three
calls are imported from :mod:`yolocar.postprocessing`; nothing is reimplemented
here. The bookkeeping that makes this node the *terminal* one -- records,
summary, ack -- lives in :class:`.terminal.Terminal`, because any node that runs
the last stage needs exactly that and nothing about it is postprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Empty

from yolocar.config import ROAD_CLASS_IDS, YoloCarConfig
from yolocar.postprocessing import (
    coordinate_adaptation,
    domain_filter,
    nms,
    to_detections,
)

from yolocar.preprocessing import describe, parse

from yolocar_interfaces.msg import CandidatesFrame

from . import qos, topics
from .conversions import candidates_from_frame
from .terminal import Terminal

LOGGER = logging.getLogger('yolocar')


class PostprocessingNode(Node):
    """Run stages 4-6, emit the file contract, pace the acquisition node."""

    def __init__(self) -> None:
        super().__init__('yolocar_postprocessing')

        iou = float(self.declare_parameter('iou', 0.45).value)
        agnostic = bool(self.declare_parameter('agnostic_nms', False).value)
        max_det = int(self.declare_parameter('max_det', 300).value)
        classes = str(
            self.declare_parameter(
                'classes', ','.join(str(c) for c in ROAD_CLASS_IDS)
            ).value
        )
        class_ids = tuple(int(value) for value in classes.split(',') if value.strip())
        records = str(self.declare_parameter('records', '').value)
        summary = str(self.declare_parameter('summary', '').value)
        detections = str(self.declare_parameter('detections', '').value)
        model = str(
            self.declare_parameter(
                'model', 'assets/models/yolov8n_fp16.engine'
            ).value
        )
        source = str(self.declare_parameter('source', '').value)
        duration = float(self.declare_parameter('duration', -1.0).value)
        max_frames = int(self.declare_parameter('max_frames', -1).value)
        log_level = str(self.declare_parameter('log_level', 'INFO').value)
        LOGGER.setLevel(log_level.upper())

        # Provenance only: this node never runs the chain, but the summary it
        # writes must say which one ran (the monolith's config block does).
        self.pre_ops = describe(parse(str(self.declare_parameter('pre_ops', '').value)))
        self.config = YoloCarConfig(
            model_path=Path(model),
            device=int(self.declare_parameter('device', 0).value),
            image_size=int(self.declare_parameter('imgsz', 640).value),
            source=source,
            max_frames=None if max_frames <= 0 else max_frames,
            loops=int(self.declare_parameter('loops', 1).value),
            duration_s=None if duration <= 0.0 else duration,
            records_path=Path(records) if records else None,
            detections_path=Path(detections) if detections else None,
            json_path=Path(summary) if summary else None,
            rotate=int(self.declare_parameter('rotate', 0).value),
            conf_threshold=float(self.declare_parameter('conf', 0.25).value),
            iou_threshold=iou,
            agnostic_nms=agnostic,
            max_detections=max_det,
            max_candidates=int(
                self.declare_parameter('max_candidates', 30_000).value
            ),
            target_class_ids=class_ids or ROAD_CLASS_IDS,
            log_level=log_level,
        )

        self.terminal = Terminal(
            self,
            self.config,
            pre_ops=self.pre_ops,
            detections_header=Terminal.detections_header(
                self.config, producer='ros-4node'
            ),
        )

        self.create_subscription(
            CandidatesFrame, topics.CANDIDATES_TOPIC, self._on_candidates, qos.latest()
        )
        self.create_subscription(
            Empty, topics.STOP_TOPIC, self._on_stop, qos.control(depth=1)
        )

        self.get_logger().info(
            f'postprocessing: iou={self.config.iou_threshold:.2f} '
            f'agnostic={self.config.agnostic_nms} '
            f'max_det={self.config.max_detections} '
            f'classes={list(self.config.target_class_ids)}'
        )

    # --------------------------------------------------------------- frames
    def _on_candidates(self, msg: CandidatesFrame) -> None:
        t0 = time.monotonic()
        candidates, letterbox = candidates_from_frame(msg)
        candidates = nms(
            candidates,
            iou_threshold=self.config.iou_threshold,
            agnostic=self.config.agnostic_nms,
            max_detections=self.config.max_detections,
        )
        candidates = domain_filter(candidates, self.config.target_class_ids)
        candidates = coordinate_adaptation(candidates, letterbox)
        detections = to_detections(candidates)
        t1 = time.monotonic()

        self.terminal.emit(
            meta=msg.meta,
            detections=detections,
            post_ms=(t1 - t0) * 1e3,
            t_end=t1,
            engine_loaded_t=float(msg.engine_loaded_t),
        )

    # -------------------------------------------------------------- finish
    def _on_stop(self, _msg: Empty) -> None:
        if self.terminal.finish(stopped=True) is None:
            return
        self.terminal.publish_done()

    def close(self) -> None:
        self.terminal.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    rclpy.init(args=argv)
    node = PostprocessingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Expected when the launch shuts the chain down after a run.
        if rclpy.ok():
            node.get_logger().info('shutting down')
    finally:
        node.terminal.finish(stopped=False)
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
