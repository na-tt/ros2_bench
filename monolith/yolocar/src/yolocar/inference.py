"""Stage 3: TensorRT invocation and decode-head handling.

Two deliberately separate steps:

* :func:`infer` pushes the preprocessed tensor through the engine. Outputs are
  numpy views owned by the engine and stay valid only until the next call, so
  they are consumed before returning from the frame (nothing is copied).
* :func:`decode_predictions` splits the raw detect head into ``(4, N)`` boxes
  and ``(nc, N)`` class scores without transposing the whole tensor.

The split is not cosmetic. A 640x640 YOLOv8n head is ~2.8 MB per frame
(``84 x 8400`` fp32), the decoded-and-thresholded candidates are a few kB. A
ROS 2 chain can put its topic boundary on either side of the decode depending
on whether the engine and the postprocessing node share a process; decode
therefore never happens implicitly inside :func:`infer`.

This module knows about the engine but nothing about frames, letterboxing or
detections, so it stays a sibling of :mod:`yolocar.preprocessing`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .engine import TensorRTEngine


@dataclass(frozen=True)
class InputSpec:
    """What a preprocessing stage must produce for this engine."""

    height: int
    width: int
    dtype: np.dtype

    @property
    def shape(self) -> tuple[int, int]:
        """Engine input size as ``(height, width)``."""
        return (self.height, self.width)


def input_spec(engine: TensorRTEngine) -> InputSpec:
    """Read (and validate) the single NCHW 3-channel input of ``engine``."""
    info = engine.single_input
    if len(info.shape) != 4:
        raise ValueError(
            f"expected a 4D NCHW input, engine exposes {info.name!r}{info.shape}"
        )
    if info.shape[1] != 3:
        raise ValueError(
            f"expected a 3-channel input, engine exposes {info.name!r}{info.shape}"
        )
    return InputSpec(
        height=int(info.shape[2]),
        width=int(info.shape[3]),
        dtype=info.dtype,
    )


def infer(engine: TensorRTEngine, tensor: np.ndarray) -> list[np.ndarray]:
    """Stage 3: run the engine on an engine-ready tensor.

    Results are views into engine-owned buffers: use them before the next
    :func:`infer` call.
    """
    return engine.infer(tensor)


def first_output(
    outputs: Sequence[np.ndarray] | Mapping[str, np.ndarray],
) -> np.ndarray:
    """Pick the detect head out of an engine output mapping/sequence."""
    if isinstance(outputs, Mapping):
        if not outputs:
            raise ValueError("engine returned no outputs")
        return next(iter(outputs.values()))
    if len(outputs) == 0:
        raise ValueError("engine returned no outputs")
    return outputs[0]


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
