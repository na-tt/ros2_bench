"""Optional presentation layer: draw detections, write annotated frames.

Everything here is a sink for pixels, never part of the measured pipeline: the
benchmark can run headless and the ROS 2 short chain can skip it entirely. It
depends on :mod:`yolocar.postprocessing` only for the :class:`Detection` type;
no CV logic lives here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from .config import VIDEO_SUFFIXES
from .postprocessing import Detection

LOGGER = logging.getLogger(__name__)

#: BGR colours cycled over class ids for annotation.
CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
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
        color = CLASS_COLORS[detection.class_id % len(CLASS_COLORS)]
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


# ---------------------------------------------------------------------- sinks
class FrameSink(Protocol):
    """Anything that can absorb annotated frames."""

    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


class VideoSink:
    """Encode annotated frames into a video file."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer for {path}")
        self.path = path

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


class ImageSink:
    """Write annotated frames as a numbered JPEG sequence."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.count = 0

    def write(self, frame: np.ndarray) -> None:
        target = self.directory / f"frame_{self.count:06d}.jpg"
        cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        self.count += 1

    def close(self) -> None:
        return None


def open_sink(
    target: Path,
    fps: float,
    size: tuple[int, int] | None,
    image_source: bool = False,
) -> FrameSink:
    """Pick a sink for ``target`` based on the source kind and frame size.

    A video source wants one encoded file; an image sequence (or a source whose
    frame size the decoder did not report) gets a numbered directory instead.
    """
    if image_source and target.suffix.lower() not in VIDEO_SUFFIXES:
        return ImageSink(target)
    if size is None:
        LOGGER.warning("frame size unknown; writing images instead of a video")
        return ImageSink(target)
    return VideoSink(target, fps, size)
