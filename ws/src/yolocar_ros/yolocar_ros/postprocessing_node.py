"""
Stages 4-6 node: NMS -> domain filter -> coordinate adaptation.

Terminal node of the chain: it writes the benchmark's per-frame
``records.jsonl``, acknowledges each frame back to the acquisition node (which
is what paces the graph), and writes ``summary.json`` when the acquisition node
signals that the run is over.

Stage order is the canonical one from ``AGENTS.md`` and the monolith: NMS over
all classes, then the road-class filter, then the inverse letterbox. The three
calls are imported from :mod:`yolocar.postprocessing`; nothing is reimplemented
here.
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

from yolocar_interfaces.msg import CandidatesFrame, FrameAck

from . import qos, topics
from .conversions import candidates_from_frame
from .reporting import build_summary, RecordWriter, write_summary

LOGGER = logging.getLogger('yolocar')

PROGRESS_EVERY = 50


class PostprocessingNode(Node):
    """Run stages 4-6, emit the file contract, pace the acquisition node."""

    def __init__(self) -> None:
        super().__init__('yolocar_postprocessing')
        self._t_start = time.monotonic()

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

        self.config = YoloCarConfig(
            model_path=Path(model),
            device=int(self.declare_parameter('device', 0).value),
            image_size=int(self.declare_parameter('imgsz', 640).value),
            source=source,
            max_frames=None if max_frames <= 0 else max_frames,
            loops=int(self.declare_parameter('loops', 1).value),
            duration_s=None if duration <= 0.0 else duration,
            records_path=Path(records) if records else None,
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

        self.records = RecordWriter(self.config.records_path)

        self.ack_publisher = self.create_publisher(
            FrameAck, topics.ACK_TOPIC, qos.control()
        )
        self.done_publisher = self.create_publisher(
            Empty, topics.DONE_TOPIC, qos.control(depth=1)
        )
        self.create_subscription(
            CandidatesFrame, topics.CANDIDATES_TOPIC, self._on_candidates, qos.latest()
        )
        self.create_subscription(
            Empty, topics.STOP_TOPIC, self._on_stop, qos.control(depth=1)
        )

        self._frames = 0
        self._max_loop = -1
        self._first_t_mono: float | None = None
        self._last_done = 0.0
        self._engine_loaded_t = 0.0
        self._latency_ms: list[float] = []
        self._acquisition_ms: list[float] = []
        self._preprocess_ms: list[float] = []
        self._inference_ms: list[float] = []
        self._postprocess_ms: list[float] = []
        self._transport_ms: list[float] = []
        self._detections_per_frame: list[int] = []
        self._detections_by_class: dict[str, int] = {}
        self._finished = False

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

        meta = msg.meta
        post_ms = (t1 - t0) * 1e3
        latency_ms = (t1 - float(meta.t_mono)) * 1e3 - float(meta.acq_ms)
        transport_ms = latency_ms - (
            float(meta.pre_ms) + float(meta.infer_ms) + post_ms
        )

        if self._first_t_mono is None:
            self._first_t_mono = float(meta.t_mono)
        if msg.engine_loaded_t > 0.0 and self._engine_loaded_t == 0.0:
            self._engine_loaded_t = float(msg.engine_loaded_t)
        self._last_done = t1
        self._max_loop = max(self._max_loop, int(meta.loop))

        self._latency_ms.append(latency_ms)
        self._acquisition_ms.append(float(meta.acq_ms))
        self._preprocess_ms.append(float(meta.pre_ms))
        self._inference_ms.append(float(meta.infer_ms))
        self._postprocess_ms.append(post_ms)
        self._transport_ms.append(transport_ms)
        self._detections_per_frame.append(len(detections))
        for detection in detections:
            self._detections_by_class[detection.class_name] = (
                self._detections_by_class.get(detection.class_name, 0) + 1
            )

        self.records.write(
            {
                'seq': int(meta.seq),
                'loop': int(meta.loop),
                't_mono': round(float(meta.t_mono), 6),
                'acq_ms': round(float(meta.acq_ms), 4),
                'pre_ms': round(float(meta.pre_ms), 4),
                'infer_ms': round(float(meta.infer_ms), 4),
                'post_ms': round(post_ms, 4),
                'present_ms': 0.0,
                'latency_ms': round(latency_ms, 4),
                'n_det': len(detections),
                # ROS-only: serialisation + transport + scheduling, i.e. what
                # end-to-end latency has beyond the three stage timings.
                'transport_ms': round(transport_ms, 4),
            }
        )
        self._frames += 1

        if self._frames % PROGRESS_EVERY == 0:
            self.get_logger().info(
                f'frame {self._frames} (loop {int(meta.loop)}): {latency_ms:.1f} ms '
                f'(acq {float(meta.acq_ms):.1f} / pre {float(meta.pre_ms):.1f} / '
                f'infer {float(meta.infer_ms):.1f} / post {post_ms:.1f} / '
                f'transport {transport_ms:.1f}), {len(detections)} dets'
            )

        # The ack is what releases the acquisition node; send it only after the
        # record is flushed, so one frame is truly in flight.
        self.ack_publisher.publish(
            FrameAck(seq=int(meta.seq), engine_loaded_t=self._engine_loaded_t)
        )

    # -------------------------------------------------------------- finish
    def _on_stop(self, _msg: Empty) -> None:
        if self._finished:
            return
        self._finish(stopped=True)
        if rclpy.ok():
            self.done_publisher.publish(Empty())

    def _finish(self, *, stopped: bool) -> None:
        """Write ``summary.json`` once, whether the run ended or was killed."""
        if self._finished:
            return
        self._finished = True
        t_end = time.monotonic()

        events: dict[str, float] = {'start': self._t_start, 'end': t_end}
        if self._engine_loaded_t > 0.0:
            events['engine_loaded'] = self._engine_loaded_t
        if self._first_t_mono is not None:
            # A frame can be captured before the inference engine is ready;
            # report the moment the pipeline could actually start on it.
            events['first_frame'] = max(self._first_t_mono, self._engine_loaded_t)
        origin = self._engine_loaded_t or self._first_t_mono or self._t_start
        wall_time_s = max(self._last_done - origin, 0.0) if self._frames else 0.0

        summary = build_summary(
            config=self.config,
            source=self.config.source if self.config.source else '',
            frames=self._frames,
            replays=max(self._max_loop + 1, 0),
            events=events,
            interrupted=not stopped,
            wall_time_s=wall_time_s,
            latency_ms=self._latency_ms,
            acquisition_ms=self._acquisition_ms,
            preprocess_ms=self._preprocess_ms,
            inference_ms=self._inference_ms,
            postprocess_ms=self._postprocess_ms,
            transport_ms=self._transport_ms,
            detections_per_frame=self._detections_per_frame,
            detections_by_class=self._detections_by_class,
            records_path=self.config.records_path,
        )
        write_summary(self.config.json_path, summary)

        median = float(summary['latency_ms'].get('median', 0.0))  # type: ignore[union-attr]
        per_frame = float(summary['detections']['per_frame_mean'])  # type: ignore[index]
        status = 'complete' if stopped else 'interrupted'
        fps = float(summary['fps'])
        self.get_logger().info(
            f'run {status}: {self._frames} frames, {fps:.2f} fps, '
            f'p50 {median:.2f} ms, {per_frame:.2f} dets/frame'
        )
        if self.config.json_path is not None:
            self.get_logger().info(f'summary written to {self.config.json_path}')

    def close(self) -> None:
        self.records.close()


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
        node._finish(stopped=False)
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
