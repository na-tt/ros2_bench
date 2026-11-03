"""Stage 2: image conditioning at source resolution.

This stage answers "how should the image look", never "what does this model
want". It takes a frame of unknown provenance, normalises it to the pipeline's
wire type (BGR uint8, source geometry preserved) and applies the configured
conditioning operations -- denoising, CLAHE, gamma, histogram equalisation and
friends -- each of which is a general image operation that knows nothing about
the engine.

What does *not* belong here is the model input contract (letterbox, colour
order, NCHW layout, dtype, scaling): that lives in :mod:`yolocar.model_input`
and is invoked by the inference stage, because it is parameterised by one
engine and improves nothing.

Nothing here touches TensorRT or the CUDA bindings, so a ROS 2 node can run this
stage on its own and publish the (still BGR uint8) frame downstream.
"""

from __future__ import annotations

import cv2
import numpy as np


def as_bgr_frame(frame: np.ndarray) -> np.ndarray:
    """Coerce a frame of unknown provenance into a contiguous BGR uint8 image.

    Grayscale is promoted, non-uint8 input is clipped and cast, and anything
    that is not ``HxWx3`` is rejected -- the three cases a caller actually hits
    (camera, video decoder, image file). A frame that already satisfies the
    contract is returned untouched, so this is a no-op on the hot path.
    """
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR frame, got {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame
