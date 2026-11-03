"""Stage 3 front half: the engine input contract.

Everything here exists because a *particular* engine wants a particular tensor:
an aspect-preserving resize with pad value 114, BGR->RGB, HWC->NCHW, a dtype
and an optional ``/255`` scale. None of that improves the image -- it makes the
image acceptable to one model. The concerns are therefore split:

===============  ============================================================
module           responsibility
===============  ============================================================
preprocessing    stage 2: image conditioning at source resolution
model_input      this module: frame -> engine-ready tensor + geometry
inference        stage 3 body: engine invocation, decode, ``input_spec()``
===============  ============================================================

This module deliberately imports nothing from :mod:`yolocar.inference`: that
module pulls in the TensorRT engine, while :mod:`yolocar.postprocessing` and the
ROS transport layer need :class:`LetterboxInfo` and must stay engine-free.

Nothing here touches TensorRT or the CUDA bindings, so a ROS 2 inference node
can call :func:`prepare` on its own with the ``InputSpec`` it read from the
engine it already owns.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class InputSpec:
    """What a model input contract requires: a tensor shape and a dtype."""

    height: int
    width: int
    dtype: np.dtype

    @property
    def shape(self) -> tuple[int, int]:
        """Engine input size as ``(height, width)``."""
        return (self.height, self.width)


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
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
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


def prepare(
    frame: np.ndarray,
    input_shape: tuple[int, int],
    dtype: np.dtype,
    pad_value: int = 114,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Frame -> ``(engine tensor, letterbox geometry)`` for one input contract.

    The two return values are exactly the two things the downstream stages need
    (inference wants the tensor, coordinate adaptation wants the geometry),
    which is why they are not bundled into one object.
    """
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR frame, got {frame.shape}")
    if frame.dtype != np.uint8:
        # Coercion is stage 2's job (preprocessing.as_bgr_frame): by the time a
        # frame reaches the contract it is already the wire type, so a wrong
        # dtype here is a caller bug and must not be repaired silently.
        raise ValueError(f"expected uint8 pixels, got {frame.dtype}")
    padded, info = letterbox(frame, input_shape, pad_value=pad_value)
    return to_tensor(padded, dtype), info
