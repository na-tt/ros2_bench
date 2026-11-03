"""Stage 2 operations: general image conditioning.

One function per operation, all with the same shape:

``(frame: BGR uint8 HxWx3, **params) -> BGR uint8 HxWx3``

They are *general* image operations -- denoising, local contrast, tone curves,
smoothing -- that know nothing about the model that will consume the result and
never change the frame geometry. A caller that wants a model-ready tensor adds
:func:`yolocar.model_input.prepare` afterwards.

Two rules, both enforced by the tests in ``test_yolocar_preprocessing.py``:

* **Do not mutate the input.** The ROS transport hands out a read-only view of
  the serialised buffer, and the monolith shares the decoded frame with the
  presentation stage. Every op returns a new array (or the input itself when
  the parameters make it a no-op).
* **Stay O(pixels).** These are stand-ins for real-world preprocessing
  workloads and the benchmark measures their cost, so they are used at source
  resolution; a per-op constant that grows with the image is a bug, not a
  feature.

The set deliberately spans different bottlenecks (single-pass LUT, multi-pass
memory work, separable convolution, non-separable O(k^2) arithmetic, a
blur-and-blend sharpening pass) so the resolution axis can show *which* cost
grows how, rather than assuming all filtering behaves alike.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


def gamma(frame: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply the tone curve ``out = 255 * (in / 255) ** gamma``.

    A single LUT pass over the pixels: the cheapest op in the set (~3.8 ms at
    720p / ~0.2 ms at 180p on this board), and the memory-bound end of the
    spectrum.
    """
    value = float(gamma)
    if value <= 0.0:
        raise ValueError(f"gamma must be > 0, got {value}")
    lut = np.clip(
        np.arange(256, dtype=np.float32) / 255.0, 0.0, 1.0
    ) ** np.float32(value)
    return cv2.LUT(frame, np.ascontiguousarray(lut * 255.0, dtype=np.uint8))


def hist_equalization(frame: np.ndarray) -> np.ndarray:
    """Equalise the luma channel's histogram (YCrCb round trip).

    Multi-pass over the frame (histogram, prefix sum, LUT, apply) and therefore
    the multi-pass memory-bound representative: it touches the same bytes more
    times than :func:`gamma` at a similar asymptotic complexity.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


@lru_cache(maxsize=8)
def _clahe(clip_limit: float, tiles: int) -> cv2.CLAHE:
    """Cached CLAHE object: a real pipeline creates it once, not per frame."""
    return cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(int(tiles), int(tiles))
    )


def clahe(
    frame: np.ndarray, clip_limit: float = 2.0, tiles: int = 8
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the luma channel.

    Fixed tile count independent of the image size, so the per-tile histogram
    work is O(pixels) but with a much larger constant than :func:`gamma`, plus
    a bilinear LUT interpolation per pixel.
    """
    tiles = int(tiles)
    if tiles < 1:
        raise ValueError(f"tiles must be >= 1, got {tiles}")
    if float(clip_limit) <= 0.0:
        raise ValueError(f"clip_limit must be > 0, got {clip_limit}")
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _clahe(float(clip_limit), tiles).apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def gaussian(frame: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Separable Gaussian blur: O(pixels * k) thanks to the two 1-D passes."""
    size = int(ksize)
    if size < 1 or size % 2 == 0:
        raise ValueError(f"ksize must be odd and >= 1, got {size}")
    return cv2.GaussianBlur(frame, (size, size), float(sigma))


def median(frame: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Non-separable median filter: O(pixels * k^2) with small constants."""
    size = int(ksize)
    if size < 3 or size % 2 == 0 or size > 5:
        # OpenCV only supports 3 and 5 for the fast uint8 path.
        raise ValueError(f"ksize must be 3 or 5, got {size}")
    return cv2.medianBlur(frame, size)


def bilateral(
    frame: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Edge-preserving smoothing; arithmetic-bound (exp per neighbour)."""
    diameter = int(d)
    if diameter < 1:
        raise ValueError(f"d must be >= 1, got {diameter}")
    return cv2.bilateralFilter(
        frame, diameter, float(sigma_color), float(sigma_space)
    )


def unsharp(
    frame: np.ndarray,
    ksize: int = 5,
    sigma: float = 0.0,
    amount: float = 1.0,
) -> np.ndarray:
    """Unsharp mask: ``out = frame + amount * (frame - blur(frame))``.

    Two passes over the frame -- a separable Gaussian and a weighted blend --
    so the cost is :func:`gaussian` plus a memory-bound ``addWeighted``. It is
    the cheap stand-in for the sharpening stage a real pipeline ends with: it
    measured ~11.8 ms standalone and ~8.5 ms inside the chain at 720p, ~0.7 ms
    at 180p: the second most expensive op, but nowhere near an outlier, which
    is what the seventh slot needs (see the calibration in
    ``docs/proposals/chain-shapes.md``).
    """
    size = int(ksize)
    if size < 3 or size % 2 == 0:
        raise ValueError(f"ksize must be odd and >= 3, got {size}")
    value = float(amount)
    if value < 0.0:
        raise ValueError(f"amount must be >= 0, got {value}")
    if value == 0.0:
        return frame
    blurred = cv2.GaussianBlur(frame, (size, size), float(sigma))
    return cv2.addWeighted(frame, 1.0 + value, blurred, -value, 0.0)
