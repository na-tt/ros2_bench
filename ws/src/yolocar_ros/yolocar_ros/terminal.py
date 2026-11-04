"""
The tail of every chain: records, summary and the ack.

Whichever node runs the last stage owns the benchmark's file contract, so this
class is about being *last*, not about postprocessing. It is parameterised by
the stage timings and the detections rather than by which stages produced them,
so a node that runs stages 2-6 and a node that runs only stage 6 write the same
record from the same code. That is what lets a chain be cut anywhere without
the contract drifting.

It does hold ROS publishers (the ack paces the acquisition node, and a `done`
signal closes the graph), but no stage logic: every function it calls lives in
``yolocar``.
"""

from __future__ import annotations

import logging
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

from yolocar.config import YoloCarConfig
from yolocar.detections import build_header, DetectionLog
from yolocar.postprocessing import Detection

from yolocar_interfaces.msg import FrameAck, FrameMeta

from . import qos, topics
from .reporting import build_summary, RecordWriter, write_summary

LOGGER = logging.getLogger('yolocar')

PROGRESS_EVERY = 50


class Terminal:
    """Accumulate per-frame results, write the file contract, pace the chain."""

    def __init__(
        self,
        node: Node,
        config: YoloCarConfig,
        *,
        pre_ops: list[str],
        detections_header: dict[str, object],
        engine_loaded_t: float = 0.0,
        shape: str = '',
    ) -> None:
        self.node = node
        self.config = config
        self.pre_ops = pre_ops
        self.shape = shape
        self._t_start = time.monotonic()

        self.records = RecordWriter(config.records_path)
        #: Side artifact, not part of the contract: one line per frame with the
        #: final detections, so this run can be rendered afterwards. A no-op
        #: unless the `detections` parameter was set.
        self.detections_log = DetectionLog(
            config.detections_path, header=detections_header
        ).open_file()

        self.ack_publisher = node.create_publisher(
            FrameAck, topics.ACK_TOPIC, qos.control()
        )
        self.done_publisher = node.create_publisher(
            Empty, topics.DONE_TOPIC, qos.control(depth=1)
        )

        self.frames = 0
        self.engine_loaded_t = engine_loaded_t
        self._max_loop = -1
        self._first_t_mono: float | None = None
        self._last_done = 0.0
        self._latency_ms: list[float] = []
        self._acquisition_ms: list[float] = []
        self._preprocess_ms: list[float] = []
        self._preprocess_ops: dict[str, list[float]] = {}
        self._postprocess_ops: dict[str, list[float]] = {}
        self._inference_ms: list[float] = []
        self._postprocess_ms: list[float] = []
        self._transport_ms: list[float] = []
        self._detections_per_frame: list[int] = []
        self._detections_by_class: dict[str, int] = {}
        self._finished = False

    @staticmethod
    def detections_header(
        config: YoloCarConfig, *, producer: str = 'ros'
    ) -> dict[str, object]:
        """Build the ``detections.jsonl`` header: everything needed to re-render."""
        return build_header(
            source=config.source,
            rotate=config.rotate,
            model=config.model_path,
            image_size=config.image_size,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
            agnostic_nms=config.agnostic_nms,
            class_ids=config.target_class_ids,
            producer=producer,
        )

    # ---------------------------------------------------------------- frames
    def emit(
        self,
        *,
        meta: FrameMeta,
        detections: list[Detection],
        t_end: float,
        engine_loaded_t: float = 0.0,
    ) -> float:
        """
        Write one frame's record, then release the acquisition node.

        Returns the end-to-end latency in ms, which is what a node logs; the
        transport residual is derived here because only the terminal node sees
        all three stage buckets at once.
        """
        # `post_ms` arrives already accumulated: a stage 4-6 split over several
        # nodes adds its share to the frame's metadata, exactly as stage 2 does.
        post_ms = float(meta.post_ms)
        latency_ms = (t_end - float(meta.t_mono)) * 1e3 - float(meta.acq_ms)
        transport_ms = latency_ms - (
            float(meta.pre_ms) + float(meta.infer_ms) + post_ms
        )

        if self._first_t_mono is None:
            self._first_t_mono = float(meta.t_mono)
        # The inference node reports when its engine became ready; the first
        # frame that carries it starts the summary's measurement clock.
        if engine_loaded_t > 0.0 and self.engine_loaded_t == 0.0:
            self.engine_loaded_t = float(engine_loaded_t)
        self._last_done = t_end
        self._max_loop = max(self._max_loop, int(meta.loop))

        self._latency_ms.append(latency_ms)
        self._acquisition_ms.append(float(meta.acq_ms))
        self._preprocess_ms.append(float(meta.pre_ms))
        pre_ops = dict(zip(meta.pre_op_names, meta.pre_op_ms))
        for name, value in pre_ops.items():
            self._preprocess_ops.setdefault(name, []).append(float(value))
        post_ops = dict(zip(meta.post_op_names, meta.post_op_ms))
        for name, value in post_ops.items():
            self._postprocess_ops.setdefault(name, []).append(float(value))
        self._inference_ms.append(float(meta.infer_ms))
        self._postprocess_ms.append(post_ms)
        self._transport_ms.append(transport_ms)
        self._detections_per_frame.append(len(detections))
        for detection in detections:
            self._detections_by_class[detection.class_name] = (
                self._detections_by_class.get(detection.class_name, 0) + 1
            )

        self.detections_log.write(
            seq=int(meta.seq),
            loop=int(meta.loop),
            t_mono=float(meta.t_mono),
            detections=detections,
        )
        self.records.write(
            {
                'seq': int(meta.seq),
                'loop': int(meta.loop),
                't_mono': round(float(meta.t_mono), 6),
                'acq_ms': round(float(meta.acq_ms), 4),
                'pre_ms': round(float(meta.pre_ms), 4),
                # Optional stage-2 breakdown, as in the monolith: absent when
                # the chain is empty, so the record stays comparable.
                **(
                    {'pre_ops': {k: round(v, 4) for k, v in pre_ops.items()}}
                    if pre_ops
                    else {}
                ),
                'infer_ms': round(float(meta.infer_ms), 4),
                'post_ms': round(post_ms, 4),
                **(
                    {'post_ops': {k: round(v, 4) for k, v in post_ops.items()}}
                    if post_ops
                    else {}
                ),
                'present_ms': 0.0,
                'latency_ms': round(latency_ms, 4),
                'n_det': len(detections),
                # ROS-only: serialisation + transport + scheduling, i.e. what
                # end-to-end latency has beyond the three stage timings.
                'transport_ms': round(transport_ms, 4),
            }
        )
        self.frames += 1

        if self.frames % PROGRESS_EVERY == 0:
            LOGGER.info(
                f'frame {self.frames} (loop {int(meta.loop)}): {latency_ms:.1f} ms '
                f'(acq {float(meta.acq_ms):.1f} / pre {float(meta.pre_ms):.1f} / '
                f'infer {float(meta.infer_ms):.1f} / post {post_ms:.1f} / '
                f'transport {transport_ms:.1f}), {len(detections)} dets'
            )

        # The ack is what releases the acquisition node; send it only after the
        # record is flushed, so one frame is truly in flight.
        self.ack_publisher.publish(
            FrameAck(seq=int(meta.seq), engine_loaded_t=self.engine_loaded_t)
        )
        return latency_ms

    # ---------------------------------------------------------------- finish
    def publish_done(self) -> None:
        """Tell the acquisition node the terminal node finished cleaning up."""
        if rclpy.ok():
            self.done_publisher.publish(Empty())

    def finish(self, *, stopped: bool) -> dict[str, object] | None:
        """Write ``summary.json`` once, whether the run ended or was killed."""
        if self._finished:
            return None
        self._finished = True
        t_end = time.monotonic()

        events: dict[str, float] = {'start': self._t_start, 'end': t_end}
        if self.engine_loaded_t > 0.0:
            events['engine_loaded'] = self.engine_loaded_t
        if self._first_t_mono is not None:
            # A frame can be captured before the inference engine is ready;
            # report the moment the pipeline could actually start on it.
            events['first_frame'] = max(self._first_t_mono, self.engine_loaded_t)
        origin = self.engine_loaded_t or self._first_t_mono or self._t_start
        wall_time_s = (
            max(self._last_done - origin, 0.0) if self.frames else 0.0
        )

        summary = build_summary(
            config=self.config,
            source=self.config.source if self.config.source else '',
            frames=self.frames,
            replays=max(self._max_loop + 1, 0),
            events=events,
            interrupted=not stopped,
            wall_time_s=wall_time_s,
            latency_ms=self._latency_ms,
            acquisition_ms=self._acquisition_ms,
            preprocess_ms=self._preprocess_ms,
            preprocess_ops=self._preprocess_ops,
            inference_ms=self._inference_ms,
            postprocess_ms=self._postprocess_ms,
            postprocess_ops=self._postprocess_ops,
            transport_ms=self._transport_ms,
            detections_per_frame=self._detections_per_frame,
            detections_by_class=self._detections_by_class,
            pre_ops=self.pre_ops,
            shape=self.shape,
            records_path=self.config.records_path,
            detections_path=self.config.detections_path,
        )
        write_summary(self.config.json_path, summary)

        latency_ms = float(summary['latency_ms'].get('median', 0.0))  # type: ignore[union-attr]
        per_frame = float(summary['detections']['per_frame_mean'])  # type: ignore[index]
        status = 'complete' if stopped else 'interrupted'
        LOGGER.info(
            f'run {status}: {self.frames} frames, '
            f'{float(summary["fps"]):.2f} fps, '
            f'{latency_ms:.2f} ms p50, {per_frame:.2f} dets/frame'
        )
        if self.config.json_path is not None:
            LOGGER.info(f'summary written to {self.config.json_path}')
        return summary

    def close(self) -> None:
        self.records.close()
        self.detections_log.close()
