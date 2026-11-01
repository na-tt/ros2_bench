"""Stages 4-6: raw engine output -> :class:`Detection` objects.

Canonical order (``AGENTS.md``)::

    4. NMS                    -- :func:`nms`
    5. domain filter          -- :func:`domain_filter`
    6. coordinate adaptation  -- :func:`coordinate_adaptation`

Every stage is a pure ``Candidates -> Candidates`` function, so the same code
can run as one callback (short chain) or as three nodes (long chain). The
aggregate :func:`postprocess` is just those three calls in a row, plus the
decode/threshold step that feeds them.

:class:`Candidates` is the wire format between stages: parallel numpy arrays
instead of a list of objects. That keeps slicing vectorised, and maps 1:1 onto
``vision_msgs/Detection2DArray`` should the ROS 2 chain need to serialise it.
Boxes are ``xyxy`` in *engine input* pixels until stage 6, in source-frame
pixels afterwards.

This module needs numpy only: no OpenCV, no TensorRT.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .config import COCO_CLASS_NAMES, YoloCarConfig, class_name
from .inference import decode_predictions, first_output
from .preprocessing import LetterboxInfo


# ------------------------------------------------------------------ value types
@dataclass(frozen=True)
class Candidates:
    """A batch of detections in a fixed coordinate space.

    ``boxes`` are ``(N, 4)`` ``xyxy`` in pixels of whatever frame the producing
    stage worked in; ``class_ids`` are COCO ids (``int32``), not indices into a
    target list.
    """

    boxes: np.ndarray  # (N, 4) float32
    scores: np.ndarray  # (N,) float32
    class_ids: np.ndarray  # (N,) int32

    @classmethod
    def empty(cls) -> Candidates:
        return cls(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
            class_ids=np.empty(0, dtype=np.int32),
        )

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def take(self, indices: np.ndarray) -> Candidates:
        """Subset in the given order (used by every filtering stage)."""
        if len(self) == 0 or np.asarray(indices).size == 0:
            return Candidates.empty()
        return Candidates(
            boxes=self.boxes[indices],
            scores=self.scores[indices],
            class_ids=self.class_ids[indices],
        )


@dataclass(frozen=True)
class Detection:
    """Single detected object in pixel coordinates of the original frame."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def as_dict(self) -> dict[str, Any]:
        """JSON friendly representation (used by the benchmark output)."""
        return {
            "x_min": round(self.x_min, 2),
            "y_min": round(self.y_min, 2),
            "x_max": round(self.x_max, 2),
            "y_max": round(self.y_max, 2),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


# ---------------------------------------------------------------------- geometry
def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """``(N, 4)`` center/width/height -> corner coordinates."""
    converted = np.empty_like(boxes, dtype=np.float32)
    half_w = boxes[:, 2] * 0.5
    half_h = boxes[:, 3] * 0.5
    converted[:, 0] = boxes[:, 0] - half_w
    converted[:, 1] = boxes[:, 1] - half_h
    converted[:, 2] = boxes[:, 0] + half_w
    converted[:, 3] = boxes[:, 1] + half_h
    return converted


# ------------------------------------------------------------------- primitives
def nms_indices(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Greedy non-maximum suppression on ``(N, 4)`` xyxy boxes.

    Returns kept indices ordered by descending score. This is the numpy
    primitive; the pipeline stage is :func:`nms`.
    """
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    ordered = boxes[order]
    x1, y1, x2, y2 = (ordered[:, i] for i in range(4))
    areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    suppressed = np.zeros(len(order), dtype=bool)
    keep: list[int] = []
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(i)
        rest = slice(i + 1, None)
        inter_w = np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])
        inter_h = np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])
        inter = np.maximum(inter_w, 0.0) * np.maximum(inter_h, 0.0)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-9)
        suppressed[rest] |= iou > iou_threshold
    return order[np.asarray(keep, dtype=np.int64)]


def multiclass_nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    agnostic: bool = False,
    max_detections: int = 300,
) -> np.ndarray:
    """Per-class (or class-agnostic) NMS, returns indices sorted by score."""
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    groups = (
        [np.arange(boxes.shape[0])]
        if agnostic
        else [np.flatnonzero(classes == c) for c in np.unique(classes)]
    )
    kept: list[np.ndarray] = []
    for group in groups:
        order = group[np.argsort(-scores[group], kind="stable")]
        keep = nms_indices(boxes[order], scores[order], iou_threshold)
        kept.append(order[keep])
    merged = np.concatenate(kept) if kept else np.empty(0, dtype=np.int64)
    merged = merged[np.argsort(-scores[merged], kind="stable")]
    return merged[:max_detections]


# ------------------------------------------------------------- pre-NMS decoding
def select_candidates(
    boxes_cxcywh: np.ndarray,
    class_scores: np.ndarray,
    conf_threshold: float,
    max_candidates: int,
) -> Candidates:
    """Threshold the decoded head into a :class:`Candidates` batch.

    This is *not* a pipeline stage: it is the candidate selection that feeds
    stage 4, matching the ultralytics reference (best class **over all
    classes**, then the confidence gate, then a top-K cap). Keeping the best
    class over the full head -- rather than over ``target_class_ids`` -- is what
    makes the documented NMS-then-filter order meaningful: a high-scoring
    non-target box can suppress an overlapping target box when NMS is agnostic,
    and only afterwards is it dropped.
    """
    num_anchors = boxes_cxcywh.shape[1]
    if num_anchors == 0:
        return Candidates.empty()

    class_ids = np.argmax(class_scores, axis=0)
    scores = class_scores[class_ids, np.arange(num_anchors)]
    keep = scores >= conf_threshold
    if not keep.any():
        return Candidates.empty()

    columns = np.flatnonzero(keep)
    scores = scores[columns]
    class_ids = class_ids[columns]

    if scores.size > max_candidates:
        top = np.argpartition(-scores, max_candidates)[:max_candidates]
        columns = columns[top]
        scores = scores[top]
        class_ids = class_ids[top]

    boxes = xywh_to_xyxy(
        np.ascontiguousarray(boxes_cxcywh[:, columns].T, dtype=np.float32)
    )
    return Candidates(
        boxes=boxes,
        scores=scores.astype(np.float32, copy=False),
        class_ids=class_ids.astype(np.int32, copy=False),
    )


# ------------------------------------------------------------- pipeline stages
def nms(
    candidates: Candidates,
    iou_threshold: float,
    agnostic: bool = False,
    max_detections: int = 300,
) -> Candidates:
    """Stage 4: non-maximum suppression, per class (or class agnostic)."""
    if len(candidates) == 0:
        return candidates
    keep = multiclass_nms_indices(
        candidates.boxes,
        candidates.scores,
        candidates.class_ids,
        iou_threshold=iou_threshold,
        agnostic=agnostic,
        max_detections=max_detections,
    )
    return candidates.take(keep)


def domain_filter(
    candidates: Candidates, class_ids: Sequence[int] | np.ndarray
) -> Candidates:
    """Stage 5: keep only the COCO ids of interest (road users + infra)."""
    if len(candidates) == 0:
        return candidates
    wanted = np.asarray(class_ids, dtype=np.int32).reshape(-1)
    keep = np.isin(candidates.class_ids, wanted)
    if keep.all():
        return candidates
    return candidates.take(np.flatnonzero(keep))


def coordinate_adaptation(
    candidates: Candidates, letterbox: LetterboxInfo
) -> Candidates:
    """Stage 6: undo letterbox scale/pad, clip to the source frame, xyxy stays."""
    if len(candidates) == 0:
        return candidates
    return replace(candidates, boxes=letterbox.to_source(candidates.boxes))


def to_detections(candidates: Candidates) -> list[Detection]:
    """Turn the final batch into per-object :class:`Detection` values."""
    return [
        Detection(
            x_min=float(box[0]),
            y_min=float(box[1]),
            x_max=float(box[2]),
            y_max=float(box[3]),
            confidence=float(confidence),
            class_id=int(class_id),
            class_name=class_name(int(class_id)),
        )
        for box, confidence, class_id in zip(
            candidates.boxes, candidates.scores, candidates.class_ids
        )
    ]


def postprocess(
    outputs: Sequence[np.ndarray] | Mapping[str, np.ndarray],
    letterbox: LetterboxInfo,
    config: YoloCarConfig,
    num_classes: int = len(COCO_CLASS_NAMES),
) -> list[Detection]:
    """Run stages 4-6 in order on a raw engine output.

    Short chains call this once; long chains call the three stage functions
    individually and pass :class:`Candidates` between nodes.
    """
    raw = first_output(outputs)
    boxes_cxcywh, class_scores = decode_predictions(raw, num_classes)

    candidates = select_candidates(
        boxes_cxcywh,
        class_scores,
        conf_threshold=config.conf_threshold,
        max_candidates=config.max_candidates,
    )
    candidates = nms(
        candidates,
        iou_threshold=config.iou_threshold,
        agnostic=config.agnostic_nms,
        max_detections=config.max_detections,
    )
    candidates = domain_filter(candidates, config.target_class_ids)
    candidates = coordinate_adaptation(candidates, letterbox)
    return to_detections(candidates)
