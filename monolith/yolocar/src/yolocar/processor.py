"""Core yolocar pipeline: frame in -> detections out.

Pipeline stages (see ``AGENTS.md``):

1. acquisition     -- handled by the caller / :mod:`yolocar.main`
2. preprocessing   -- :func:`letterbox` + :func:`to_tensor` (RGB, 1x3xHxW, fp16/32)
3. inference       -- TensorRT engine, see :mod:`yolocar.engine`
4. NMS             -- :func:`nms`, greedy, per class
5. domain filter   -- only ``config.target_class_ids`` (road users + infrastructure)
6. coordinate adaptation -- letterboxed input space -> original frame pixels

Everything here is front-end agnostic: the standalone benchmark and the ROS 2
node both drive :class:`YoloCarProcessor`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .config import COCO_CLASS_NAMES, YoloCarConfig, class_name
from .engine import TensorRTEngine

LOGGER = logging.getLogger(__name__)

#: BGR colours cycled over class ids for annotation.
_CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (147, 69, 52),
    (255, 115, 100),
    (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82),
    (255, 56, 203),
    (200, 149, 255),
    (199, 55, 255),
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


@dataclass(frozen=True)
class LetterboxInfo:
    """Geometry of the letterbox transform, needed to undo it."""

    scale: float
    pad_x: int
    pad_y: int
    original_shape: tuple[int, int]  # (height, width)
    input_shape: tuple[int, int]  # (height, width)

    def to_source(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Map ``(N, 4)`` xyxy boxes from input space back to source pixels."""
        if boxes_xyxy.size == 0:
            return boxes_xyxy
        height, width = self.original_shape
        boxes = boxes_xyxy.astype(np.float32, copy=True)
        boxes[:, [0, 2]] -= self.pad_x
        boxes[:, [1, 3]] -= self.pad_y
        boxes /= self.scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(width))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(height))
        return boxes


@dataclass(frozen=True)
class PreprocessedFrame:
    """Engine-ready tensor plus the metadata needed to undo preprocessing."""

    tensor: np.ndarray  # (1, 3, H, W), dtype matches the engine input
    letterbox: LetterboxInfo


# ---------------------------------------------------------------- preprocessing
def letterbox(
    frame: np.ndarray,
    new_shape: tuple[int, int],
    pad_value: int = 114,
    scaleup: bool = True,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize ``frame`` preserving aspect ratio and pad it to ``new_shape``.

    Returns the padded BGR image and the geometry needed by
    :meth:`LetterboxInfo.to_source`. Padding is split with integer offsets so
    the inverse mapping is exact.
    """
    if frame.ndim != 3:
        raise ValueError(f"expected an HxWxC frame, got shape {frame.shape}")
    height, width = frame.shape[:2]
    new_h, new_w = int(new_shape[0]), int(new_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"empty frame: {frame.shape}")

    scale = min(new_h / height, new_w / width)
    if not scaleup:
        scale = min(scale, 1.0)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    # Recompute from the rounded size so to_source() is exact.
    scale = min(resized_w / width, resized_h / height)

    if (resized_w, resized_h) != (width, height):
        # INTER_LINEAR matches the ultralytics reference letterbox (and is ~2x
        # faster than INTER_AREA on the Jetson CPU).
        resized = cv2.resize(
            frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
    else:
        resized = frame

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    if pad_w or pad_h:
        canvas = np.full((new_h, new_w, frame.shape[2]), pad_value, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    else:
        canvas = resized

    info = LetterboxInfo(
        scale=float(scale),
        pad_x=int(pad_x),
        pad_y=int(pad_y),
        original_shape=(height, width),
        input_shape=(new_h, new_w),
    )
    return canvas, info


def to_tensor(
    bgr_image: np.ndarray, dtype: np.dtype, mean: float = 255.0
) -> np.ndarray:
    """BGR HWC uint8 -> RGB 1x3xHxW tensor with values in ``[0, 1]``."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=dtype)
    if tensor.dtype != np.uint8 and mean:
        np.divide(tensor, mean, out=tensor)
    return tensor


def decode_predictions(
    raw: np.ndarray, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a YOLOv8 detect head output into ``(boxes_cxcywh, class_scores)``.

    Accepts ``(1, 4+nc, N)``, ``(4+nc, N)`` or ``(1, N, 4+nc)`` layouts and
    returns ``((4, N), (nc, N))`` views/copies without transposing the whole
    tensor.
    """
    preds = np.asarray(raw)
    if preds.ndim == 3:
        if preds.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {preds.shape}")
        preds = preds[0]
    if preds.ndim != 2:
        raise ValueError(f"unexpected YOLO output shape {preds.shape}")
    expected = 4 + num_classes
    if preds.shape[0] == expected:
        channel_first = preds
    elif preds.shape[1] == expected:
        channel_first = preds.T
    else:
        raise ValueError(
            f"output shape {preds.shape} does not match 4 + {num_classes} classes"
        )
    return channel_first[:4], channel_first[4:]


# ------------------------------------------------------------------------ NMS
def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Greedy non-maximum suppression on ``(N, 4)`` xyxy boxes.

    Returns kept indices ordered by descending score.
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


def multiclass_nms(
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
        keep = nms(boxes[order], scores[order], iou_threshold)
        kept.append(order[keep])
    merged = np.concatenate(kept) if kept else np.empty(0, dtype=np.int64)
    merged = merged[np.argsort(-scores[merged], kind="stable")]
    return merged[:max_detections]


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


# ------------------------------------------------------------------ processor
class YoloCarProcessor:
    """Loads a TensorRT YOLO engine once and runs the full pipeline per frame."""

    def __init__(self, config: YoloCarConfig | None = None) -> None:
        self.config = config or YoloCarConfig()
        self._engine: TensorRTEngine | None = None
        self._num_classes = len(COCO_CLASS_NAMES)
        self._class_ids = np.asarray(
            sorted(set(self.config.target_class_ids)), dtype=np.int64
        )
        if self._class_ids.size == 0:
            raise ValueError("config.target_class_ids must contain at least one class id")
        if self._class_ids.min() < 0 or self._class_ids.max() >= self._num_classes:
            raise ValueError(
                f"target_class_ids must be in [0, {self._num_classes - 1}]"
            )
        self._input_h = int(self.config.image_size)
        self._input_w = int(self.config.image_size)

    # ------------------------------------------------------------- lifecycle
    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    @property
    def engine(self) -> TensorRTEngine:
        if self._engine is None:
            raise RuntimeError("call YoloCarProcessor.load() first")
        return self._engine

    @property
    def input_dtype(self) -> np.dtype:
        return self.engine.single_input.dtype

    @property
    def input_shape(self) -> tuple[int, int]:
        """Engine input size as ``(height, width)``."""
        return (self._input_h, self._input_w)

    def load(self) -> None:
        """Deserialize the engine and allocate its buffers."""
        if self._engine is not None:
            return
        config = self.config
        engine = TensorRTEngine(
            config.model_path,
            device_index=config.device,
            default_input_shape=(1, 3, int(config.image_size), int(config.image_size)),
            pinned_host_memory=config.pinned_host_memory,
            log_level=config.trt_log_level,
        )
        try:
            info = engine.single_input
            if len(info.shape) != 4:
                raise ValueError(
                    f"expected a 4D NCHW input, engine exposes {info.name!r}{info.shape}"
                )
            if info.shape[1] != 3:
                raise ValueError(
                    f"expected a 3-channel input, engine exposes {info.name!r}{info.shape}"
                )
            self._input_h = int(info.shape[2])
            self._input_w = int(info.shape[3])
        except Exception:
            engine.close()
            raise
        self._engine = engine
        LOGGER.info(
            "yolocar ready: %s input=%s%s output=%s",
            config.model_path,
            info.dtype,
            info.shape,
            {name: (t.dtype, t.shape) for name, t in engine.outputs.items()},
        )

    def close(self) -> None:
        """Release model resources."""
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def warmup(self, iterations: int | None = None) -> None:
        """Run dummy inference so the first real frame is not penalised.

        A 720p grey frame is used so the resize path is exercised as well.
        """
        self.load()
        count = self.config.warmup_iterations if iterations is None else int(iterations)
        if count <= 0:
            return
        dummy = np.full((720, 1280, 3), 114, dtype=np.uint8)
        for _ in range(count):
            self.process_frame(dummy)
        LOGGER.debug("warmup finished (%d iterations)", count)

    def __enter__(self) -> "YoloCarProcessor":
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -------------------------------------------------------------- pipeline
    def preprocess(self, frame: np.ndarray) -> PreprocessedFrame:
        """Stage 2: BGR frame -> letterboxed RGB ``1x3xHxW`` tensor."""
        if frame is None:
            raise ValueError("frame is None")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR frame, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        padded, info = letterbox(
            frame,
            (self._input_h, self._input_w),
            pad_value=self.config.pad_value,
        )
        return PreprocessedFrame(
            tensor=to_tensor(padded, self.input_dtype), letterbox=info
        )

    def infer(self, preprocessed: PreprocessedFrame) -> list[np.ndarray]:
        """Stage 3: run the engine (outputs are views, valid until next call)."""
        return self.engine.infer(preprocessed.tensor)

    def postprocess(
        self, outputs: Sequence[np.ndarray] | Mapping[str, np.ndarray],
        preprocessed: PreprocessedFrame,
    ) -> list[Detection]:
        """Stages 4-6: NMS, domain filter, coordinate adaptation."""
        if isinstance(outputs, Mapping):
            if not outputs:
                raise ValueError("engine returned no outputs")
            raw = next(iter(outputs.values()))
        else:
            if len(outputs) == 0:
                raise ValueError("engine returned no outputs")
            raw = outputs[0]

        boxes_cxcywh, class_scores = decode_predictions(raw, self._num_classes)
        if boxes_cxcywh.shape[1] == 0:
            return []

        # Stage 5 (early): only score the classes we care about, which keeps the
        # candidate set small before any sorting/NMS work.
        target_scores = class_scores[self._class_ids]  # (K, N)
        best_scores = target_scores.max(axis=0)
        candidate_mask = best_scores >= self.config.conf_threshold
        if not candidate_mask.any():
            return []

        best_scores = best_scores[candidate_mask]
        target_scores = target_scores[:, candidate_mask]
        classes = self._class_ids[np.argmax(target_scores, axis=0)]

        if best_scores.size > self.config.max_candidates:
            top = np.argpartition(-best_scores, self.config.max_candidates)[
                : self.config.max_candidates
            ]
            best_scores = best_scores[top]
            classes = classes[top]
            candidate_mask = np.flatnonzero(candidate_mask)[top]

        boxes = xywh_to_xyxy(
            np.ascontiguousarray(boxes_cxcywh[:, candidate_mask].T, dtype=np.float32)
        )

        # Stage 4: NMS.
        keep = multiclass_nms(
            boxes,
            best_scores,
            classes,
            iou_threshold=self.config.iou_threshold,
            agnostic=self.config.agnostic_nms,
            max_detections=self.config.max_detections,
        )
        if keep.size == 0:
            return []

        # Stage 6: back to source pixels.
        boxes = preprocessed.letterbox.to_source(boxes[keep])
        confidences = best_scores[keep]
        class_ids = classes[keep]

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
            for box, confidence, class_id in zip(boxes, confidences, class_ids)
        ]

    def process_frame(self, frame: np.ndarray) -> list[Detection]:
        """Full pipeline for a single BGR frame."""
        preprocessed = self.preprocess(frame)
        outputs = self.infer(preprocessed)
        return self.postprocess(outputs, preprocessed)

    __call__ = process_frame


def process_frame(frame: np.ndarray, processor: YoloCarProcessor) -> list[Detection]:
    """Module-level entry point mirroring the other pipeline packages."""
    return processor.process_frame(frame)


# ------------------------------------------------------------------- rendering
def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    thickness: int | None = None,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Return a copy of ``frame`` with detection boxes and labels annotated."""
    canvas = frame.copy()
    if not detections:
        return canvas
    if thickness is None:
        thickness = max(1, int(round(min(canvas.shape[:2]) / 480)))

    for detection in detections:
        color = _CLASS_COLORS[detection.class_id % len(_CLASS_COLORS)]
        x1, y1, x2, y2 = (int(round(v)) for v in detection.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        top = max(0, y1 - text_h - baseline - 2)
        cv2.rectangle(
            canvas,
            (x1, top),
            (x1 + text_w + 2, top + text_h + baseline + 2),
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 1, top + text_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return canvas
