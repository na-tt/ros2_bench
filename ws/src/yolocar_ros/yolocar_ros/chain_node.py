"""
One node for any contiguous span of the pipeline's stages.

Every shape in ``docs/proposals/chain-shapes.md`` is a partition of the same
stage sequence::

    pre (the conditioning chain) | infer | post (nms, filter, coord)

so a shape is a list of *spans* and a node only has to know which span it runs.
The span is a launch parameter::

    stages:=pre,infer        # condition and infer in one process (p1c1)
    stages:=infer,post       # infer and postprocess in one process (p2c0)
    stages:=pre,infer,post   # the monolith in one process (p1c0)
    stages:=pre              # today's preprocessing node (p2c1)

and inside the last of them, ``post_stages:=nms`` / ``filter,coord`` /
``filter`` / ``coord`` cuts stage 4-6 further. Nothing is duplicated per shape
and no shape gets its own binary: the launch file is the only thing that knows
the shape.

Three rules keep this honest rather than clever:

* **Only the node that runs ``infer`` loads the engine**, so the "one engine
  owner" rule survives every merge and ``stages:=pre`` still never touches
  TensorRT.
* **Only the node that runs the last post stage is terminal** -- records,
  summary, ack and the detections sidecar are derived from the span, not
  configured, so a badly written shape cannot silently lose the file contract.
* **Timings are additive**, not overwritten: a span that runs two stages adds to
  both buckets, and a stage cut over several nodes accumulates (``pre_ms``,
  ``post_ms`` and their per-op arrays), so the same work reports the same
  numbers at any partition depth. The extra hops land in ``transport_ms``.

The per-stage work itself is imported from ``yolocar``; nothing is
reimplemented here.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Callable

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Empty

from yolocar.config import COCO_CLASS_NAMES, ROAD_CLASS_IDS, YoloCarConfig
from yolocar.inference import decode_predictions, first_output, infer
from yolocar.model_input import LetterboxInfo, prepare
from yolocar.postprocessing import (
    Candidates,
    coordinate_adaptation,
    domain_filter,
    nms,
    select_candidates,
    to_detections,
)
from yolocar.preprocessing import as_bgr_frame, condition, describe, parse
from yolocar.processor import YoloCarProcessor

from yolocar_interfaces.msg import CandidatesFrame, FrameMeta, ImageFrame

from . import qos, topics
from .conversions import (
    candidates_frame,
    candidates_from_frame,
    frame_from_image,
    image_frame,
)
from .terminal import Terminal

LOGGER = logging.getLogger('yolocar')

#: Canonical stage order. A span must be contiguous in this tuple.
STAGES: tuple[str, ...] = ('pre', 'infer', 'post')

#: Stage 4-6 order, likewise contiguous.
POST_STAGES: tuple[str, ...] = ('nms', 'filter', 'coord')


def _split(spec: str) -> tuple[str, ...]:
    """Split a stage list on ``,`` or ``;``, dropping empties and whitespace."""
    parts = str(spec or '').replace(';', ',').split(',')
    return tuple(part.strip() for part in parts if part.strip())


def _contiguous(
    items: tuple[str, ...], order: tuple[str, ...], what: str
) -> tuple[str, ...]:
    unknown = [item for item in items if item not in order]
    if unknown:
        raise ValueError(
            f'unknown {what}(s) {", ".join(unknown)}; accepted: {", ".join(order)}'
        )
    if len(set(items)) != len(items):
        raise ValueError(f'{what} repeated in {", ".join(items)}')
    indices = [order.index(item) for item in items]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError(
            f'{what} {", ".join(items)} are not a contiguous span of '
            f'{", ".join(order)} (and must stay in that order)'
        )
    return items


def parse_stages(spec: str) -> tuple[str, ...]:
    """Turn ``"pre,infer"`` into a validated contiguous span of :data:`STAGES`."""
    items = _split(spec)
    return _contiguous(items, STAGES, 'stage') if items else STAGES[:1]


def parse_post_stages(spec: str) -> tuple[str, ...]:
    """Turn ``"filter,coord"`` into a validated contiguous span of stage 4-6."""
    items = _split(spec)
    return _contiguous(items, POST_STAGES, 'post stage') if items else POST_STAGES


class ChainNode(Node):
    """Run the span of stages this process was told to run."""

    def __init__(
        self, default_stages: str = '', node_name: str = 'yolocar_chain'
    ) -> None:
        super().__init__(node_name)

        # Declared with defaults so every executable sharing this class accepts
        # the same parameter file; a shape only overrides what it uses.
        self.stages = parse_stages(
            str(self.declare_parameter('stages', default_stages).value)
        )
        self.post_stages = parse_post_stages(
            str(self.declare_parameter('post_stages', ','.join(POST_STAGES)).value)
        )
        self.pre_chain = parse(str(self.declare_parameter('pre_ops', '').value))
        self.pre_ops = describe(self.pre_chain)
        #: The shape this process is part of, for provenance. The launch file
        #: sets it; empty means a hand-built graph.
        self.shape = str(self.declare_parameter('shape', '').value)

        self.in_topic = str(
            self.declare_parameter('in_topic', self._default_in_topic()).value
        )
        self.out_topic = str(
            self.declare_parameter('out_topic', self._default_out_topic()).value
        )
        log_level = str(self.declare_parameter('log_level', 'INFO').value)
        LOGGER.setLevel(log_level.upper())

        self.config = self._config(log_level)
        self.pad_value = int(self.declare_parameter('pad_value', 114).value)
        self.terminal = self._terminal()

        # The engine is loaded by whoever runs `infer` and by nobody else.
        self.processor: YoloCarProcessor | None = None
        self.engine_loaded_t = 0.0
        if 'infer' in self.stages:
            self._load_engine()

        self._declare_topics()
        self.get_logger().info(self._describe())

    # ------------------------------------------------------------ parameters
    @property
    def is_terminal(self) -> bool:
        """Report whether this node runs the last stage of the whole pipeline."""
        return (
            self.stages[-1] == 'post'
            and self.post_stages[-1] == POST_STAGES[-1]
        )

    @property
    def last_stage(self) -> str:
        """Return the last atomic stage this node runs (``pre``/``infer``/``nms``...)."""
        last = self.stages[-1]
        return self.post_stages[-1] if last == 'post' else last

    @property
    def output_kind(self) -> str | None:
        """Return what this node publishes, or ``None`` when it writes records."""
        if self.is_terminal:
            return None
        return 'image' if self.last_stage == 'pre' else 'candidates'

    def _default_in_topic(self) -> str:
        """
        Return the topic a span with no explicit ``in_topic`` expects to receive.

        A span that *starts* the pipeline reads what acquisition publishes; one
        that starts mid-pipeline reads what the previous family publishes. The
        split-stage-2 case is not covered here on purpose: those instances are
        wired by the launch file (``yolocar/filter_<i>``), never by default.
        """
        return {
            'pre': topics.FRAME_TOPIC,
            'infer': topics.INPUT_TOPIC,
            'post': topics.CANDIDATES_TOPIC,
        }[self.stages[0]]

    def _default_out_topic(self) -> str:
        return (
            topics.INPUT_TOPIC
            if self.output_kind == 'image'
            else topics.CANDIDATES_TOPIC
        )

    def _describe(self) -> str:
        parts = ['/'.join(self.stages)]
        if 'post' in self.stages:
            parts.append(f'(post: {", ".join(self.post_stages)})')
        parts.append(
            f'{self.in_topic} -> '
            + (self.out_topic if self.output_kind else 'records')
        )
        if 'pre' in self.stages:
            parts.append(f'[{", ".join(self.pre_ops) or "no conditioning ops"}]')
        if self.shape:
            parts.append(f'shape={self.shape}')
        return ' '.join(parts)

    def _load_engine(self) -> None:
        self.processor = YoloCarProcessor(self.config)
        self.get_logger().info(f'loading engine {self.config.model_path}')
        t_load = time.monotonic()
        self.processor.load()
        self.engine_loaded_t = time.monotonic()
        self.get_logger().info(
            f'engine ready after {self.engine_loaded_t - t_load:.2f} s '
            f'(contract: {self.processor.input_spec.dtype} '
            f'{self.processor.input_spec.shape}, pad={self.pad_value}, '
            f'{len(COCO_CLASS_NAMES)} classes)'
        )

    def _config(self, log_level: str) -> YoloCarConfig:
        records = str(self.declare_parameter('records', '').value)
        summary = str(self.declare_parameter('summary', '').value)
        detections = str(self.declare_parameter('detections', '').value)
        source = str(self.declare_parameter('source', '').value)
        duration = float(self.declare_parameter('duration', -1.0).value)
        max_frames = int(self.declare_parameter('max_frames', -1).value)
        classes = str(
            self.declare_parameter(
                'classes', ','.join(str(c) for c in ROAD_CLASS_IDS)
            ).value
        )
        class_ids = tuple(int(value) for value in classes.split(',') if value.strip())
        return YoloCarConfig(
            model_path=Path(
                str(
                    self.declare_parameter(
                        'model', 'assets/models/yolov8n_fp16.engine'
                    ).value
                )
            ),
            device=int(self.declare_parameter('device', 0).value),
            image_size=int(self.declare_parameter('imgsz', 640).value),
            pinned_host_memory=bool(self.declare_parameter('pinned', True).value),
            trt_log_level=str(
                self.declare_parameter('trt_log_level', 'warning').value
            ),
            source=source,
            max_frames=None if max_frames <= 0 else max_frames,
            loops=int(self.declare_parameter('loops', 1).value),
            duration_s=None if duration <= 0.0 else duration,
            records_path=Path(records) if records else None,
            detections_path=Path(detections) if detections else None,
            json_path=Path(summary) if summary else None,
            rotate=int(self.declare_parameter('rotate', 0).value),
            conf_threshold=float(self.declare_parameter('conf', 0.25).value),
            iou_threshold=float(self.declare_parameter('iou', 0.45).value),
            agnostic_nms=bool(self.declare_parameter('agnostic_nms', False).value),
            max_detections=int(self.declare_parameter('max_det', 300).value),
            max_candidates=int(
                self.declare_parameter('max_candidates', 30_000).value
            ),
            target_class_ids=class_ids or ROAD_CLASS_IDS,
            log_level=log_level,
        )

    def _terminal(self) -> Terminal | None:
        if not self.is_terminal:
            return None
        return Terminal(
            self,
            self.config,
            pre_ops=self.pre_ops,
            detections_header=Terminal.detections_header(
                self.config, producer=self.shape or 'ros'
            ),
            shape=self.shape,
        )

    def _declare_topics(self) -> None:
        self.publisher = None
        if self.output_kind == 'candidates':
            self.publisher = self.create_publisher(
                CandidatesFrame, self.out_topic, qos.latest()
            )
        elif self.output_kind == 'image':
            self.publisher = self.create_publisher(
                ImageFrame, self.out_topic, qos.latest()
            )

        if self.stages[0] == 'post':
            self.create_subscription(
                CandidatesFrame, self.in_topic, self._on_candidates, qos.latest()
            )
        else:
            self.create_subscription(
                ImageFrame, self.in_topic, self._on_image, qos.latest()
            )

        if self.is_terminal:
            self.create_subscription(
                Empty, topics.STOP_TOPIC, self._on_stop, qos.control(depth=1)
            )
            self.get_logger().info(
                f'postprocessing: iou={self.config.iou_threshold:.2f} '
                f'agnostic={self.config.agnostic_nms} '
                f'max_det={self.config.max_detections} '
                f'classes={list(self.config.target_class_ids)}'
            )

    # ---------------------------------------------------------------- frames
    def _on_image(self, msg: ImageFrame) -> None:
        """Run the ``pre`` and/or ``infer`` part of this span on a BGR frame."""
        meta = msg.meta
        frame: np.ndarray | None = None

        if 'pre' in self.stages:
            t0 = time.monotonic()
            frame, op_ms = condition(as_bgr_frame(frame_from_image(msg)), self.pre_chain)
            t1 = time.monotonic()
            # Additive, not overwritten: a stage cut over several nodes adds
            # its share here, so `pre_ms` means the whole stage either way.
            meta.pre_ms = float(meta.pre_ms) + (t1 - t0) * 1e3
            meta.pre_op_names = list(meta.pre_op_names) + list(op_ms)
            meta.pre_op_ms = list(meta.pre_op_ms) + list(op_ms.values())
            if 'infer' not in self.stages:
                self.publisher.publish(image_frame(frame, meta))
                return

        if frame is None:
            t0 = time.monotonic()
            frame = frame_from_image(msg)

        candidates, letterbox = self._run_infer(frame, meta)

        if 'post' not in self.stages:
            self.publisher.publish(
                candidates_frame(
                    candidates, meta, letterbox, engine_loaded_t=self.engine_loaded_t
                )
            )
            return

        self._run_post(
            candidates, letterbox, meta, engine_loaded_t=self.engine_loaded_t
        )

    def _on_candidates(self, msg: CandidatesFrame) -> None:
        """Run the ``post`` span on incoming candidates."""
        candidates, letterbox = candidates_from_frame(msg)
        self._run_post(
            candidates,
            letterbox,
            msg.meta,
            engine_loaded_t=float(msg.engine_loaded_t),
        )

    # ---------------------------------------------------------------- stages
    def _run_infer(
        self, frame: np.ndarray, meta: FrameMeta
    ) -> tuple[Candidates, LetterboxInfo]:
        """Model input contract + engine + detect-head decode (stage 3)."""
        assert self.processor is not None  # only reached when 'infer' is in span
        t0 = time.monotonic()
        spec = self.processor.input_spec
        # The contract: letterbox + RGB HWC->NCHW + dtype, all read off the
        # engine this node owns, so there is nothing to disagree with.
        tensor, letterbox = prepare(
            frame, spec.shape, spec.dtype, pad_value=self.pad_value
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
        meta.infer_ms = (time.monotonic() - t0) * 1e3
        return candidates, letterbox

    def _run_post(
        self,
        candidates: Candidates,
        letterbox: LetterboxInfo,
        meta: FrameMeta,
        *,
        engine_loaded_t: float,
    ) -> None:
        """Run the configured slice of stages 4-6, then emit or forward."""
        t0 = time.monotonic()
        timings: dict[str, float] = {}
        for name in self.post_stages:
            step = time.monotonic()
            if name == 'nms':
                candidates = nms(
                    candidates,
                    iou_threshold=self.config.iou_threshold,
                    agnostic=self.config.agnostic_nms,
                    max_detections=self.config.max_detections,
                )
            elif name == 'filter':
                candidates = domain_filter(candidates, self.config.target_class_ids)
            elif name == 'coord':
                candidates = coordinate_adaptation(candidates, letterbox)
            else:  # pragma: no cover - parse_post_stages rejects anything else
                raise AssertionError(name)
            timings[name] = (time.monotonic() - step) * 1e3
        elapsed_ms = (time.monotonic() - t0) * 1e3

        # Same contract as stage 2: the stage total is the sum of the nodes
        # that contributed, and the per-stage breakdown travels alongside it.
        meta.post_ms = float(meta.post_ms) + elapsed_ms
        meta.post_op_names = list(meta.post_op_names) + list(timings)
        meta.post_op_ms = list(meta.post_op_ms) + list(timings.values())

        if not self.is_terminal:
            self.publisher.publish(
                candidates_frame(
                    candidates, meta, letterbox, engine_loaded_t=engine_loaded_t
                )
            )
            return

        assert self.terminal is not None
        self.terminal.emit(
            meta=meta,
            detections=to_detections(candidates),
            t_end=t0 + elapsed_ms * 1e-3,
            engine_loaded_t=engine_loaded_t,
        )

    # ---------------------------------------------------------------- finish
    def _on_stop(self, _msg: Empty) -> None:
        if self.terminal is None:
            return
        if self.terminal.finish(stopped=True) is None:
            return
        self.terminal.publish_done()

    def close(self) -> None:
        if self.terminal is not None:
            self.terminal.close()
        if self.processor is not None:
            self.processor.close()


def entry_point(
    default_stages: str, node_name: str
) -> Callable[[list[str] | None], int]:
    """
    Build a console-script ``main`` that runs a fixed default span.

    The three historic executables (``preprocessing``, ``inference``,
    ``postprocessing``) are this function applied to one stage each, so their
    behaviour and parameter surface are unchanged by the generalisation.
    """

    def main(argv: list[str] | None = None) -> int:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        )
        rclpy.init(args=argv)
        node = ChainNode(default_stages=default_stages, node_name=node_name)
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            # Expected when the launch shuts the chain down after a run.
            if rclpy.ok():
                node.get_logger().info('shutting down')
        finally:
            if node.terminal is not None:
                node.terminal.finish(stopped=False)
            node.close()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        return 0

    return main


#: The general executable: `stages:=` decides what it runs, so every shape in
#: the proposal is this binary with a different span. It defaults to the one
#: stage whose output is another image, because that is the only span that
#: needs an `in_topic`/`out_topic` pair to make sense on its own.
main = entry_point('pre', 'yolocar_chain')
