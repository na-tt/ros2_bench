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

A chain is data, not code: :func:`parse` turns ``("clahe", "bilateral:d=9")``
into a tuple of :class:`Op`, and a cut list turns that tuple into a partitioned
ROS 2 graph. Nothing here touches TensorRT or the CUDA bindings, so a node can
run one op (or the whole chain) on its own and publish the still-BGR frame.

Empty chain = identity
----------------------
``condition(frame)`` with no ops returns the input array itself, so the
measurement of an unconditioned pipeline stays byte-for-byte what it was before
this stage existed (and costs no copy).
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import filters

#: Every operation a chain may reference, by name. One entry per
#: ``yolocar.filters`` function; adding an op is adding a function there.
OPS: dict[str, Callable[..., np.ndarray]] = {
    "clahe": filters.clahe,
    "bilateral": filters.bilateral,
    "gaussian": filters.gaussian,
    "gamma": filters.gamma,
    "hist_equalization": filters.hist_equalization,
    "median": filters.median,
    "denoise": filters.denoise,
}


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


@dataclass(frozen=True)
class Op:
    """One configured conditioning operation: a registry entry plus params."""

    name: str
    params: Mapping[str, float] = field(default_factory=dict)

    @property
    def function(self) -> Callable[..., np.ndarray]:
        return OPS[self.name]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Run the operation; never mutates ``frame``."""
        return self.function(frame, **self.params)

    def __str__(self) -> str:
        if not self.params:
            return self.name
        rendered = ",".join(f"{key}={value:g}" for key, value in self.params.items())
        return f"{self.name}:{rendered}"


def _parse_value(raw: str) -> float | str:
    """``"9"`` -> 9.0, ``"0.5"`` -> 0.5, anything else is left as a string."""
    text = raw.strip()
    try:
        return float(text)
    except ValueError:
        return text


def _validate_params(name: str, params: Mapping[str, object]) -> None:
    accepted = set(inspect.signature(OPS[name]).parameters) - {"frame"}
    unknown = set(params) - accepted
    if unknown:
        raise ValueError(
            f"op {name!r} does not take {', '.join(sorted(unknown))}; "
            f"accepted: {', '.join(sorted(accepted)) or 'none'}"
        )


def parse(spec: str | Sequence[str] | None) -> tuple[Op, ...]:
    """Turn a chain specification into validated, configured ops.

    Accepts ``None``/``""`` (empty chain), a single string, or a sequence of
    strings. Each element is ``name`` or ``name:key=value,key2=value2``, e.g.
    ``("clahe:clip_limit=3.0", "bilateral:d=9")``. Both forms exist so the same
    chain can come from a CLI flag, a launch argument or a scenario file.
    """
    if spec is None:
        return ()
    if isinstance(spec, str):
        items: Sequence[str] = (spec,) if spec.strip() else ()
    else:
        items = tuple(spec)

    chain: list[Op] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        name, _, raw_params = text.partition(":")
        name = name.strip()
        if name not in OPS:
            raise ValueError(
                f"unknown preprocessing op {name!r}; available: "
                f"{', '.join(sorted(OPS))}"
            )
        if name in seen:
            # A dict of per-op timings cannot represent the same op twice, and
            # repeating an op is far more likely a typo than an intention.
            raise ValueError(f"op {name!r} appears twice in the chain")
        seen.add(name)

        params: dict[str, float] = {}
        for piece in filter(None, (p.strip() for p in raw_params.split(","))):
            key, sep, value = piece.partition("=")
            if not sep:
                raise ValueError(
                    f"malformed parameter {piece!r} in op {name!r}; "
                    "expected key=value"
                )
            params[key.strip()] = _parse_value(value)  # type: ignore[assignment]
        _validate_params(name, params)
        chain.append(Op(name=name, params=params))
    return tuple(chain)


def describe(chain: Sequence[Op]) -> list[str]:
    """Printable form of a chain, for provenance blocks and log lines."""
    return [str(op) for op in chain]


def enhance(
    frame: np.ndarray, chain: Sequence[Op] = ()
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply ``chain`` in order; return the result and per-op milliseconds.

    The timings are the stage's own breakdown: their sum is what a single
    ``pre_ms`` would have been had the chain not been split across nodes, which
    is what makes a partitioned chain comparable to a monolith one.
    """
    timings: dict[str, float] = {}
    if not chain:
        return frame, timings
    result = frame
    for op in chain:
        start = time.perf_counter()
        result = op.apply(result)
        timings[op.name] = (time.perf_counter() - start) * 1e3
    return result, timings


def condition(
    frame: np.ndarray, chain: Sequence[Op] = ()
) -> tuple[np.ndarray, dict[str, float]]:
    """Stage 2 entry point: coerce the frame, then apply the chain.

    Coercion must happen here rather than in :mod:`yolocar.model_input` because
    it is an image-domain concern: the stage guarantees its consumers (the
    contract, the transport, another filter node) a BGR uint8 frame.
    """
    return enhance(as_bgr_frame(frame), chain)
