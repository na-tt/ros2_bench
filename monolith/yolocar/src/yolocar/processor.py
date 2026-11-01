"""Single-process facade: owns the engine and wires stages 2-6 together.

The stages themselves live in their own modules and are usable without this
class:

===============  ============================================================
module           stage
===============  ============================================================
acquisition      frame source + rotation correction
preprocessing    letterbox + BGR->RGB NCHW tensor
inference        TensorRT invocation + detect-head decoding
postprocessing   NMS -> domain filter -> coordinate adaptation
render           optional annotation / sinks
===============  ============================================================

A ROS 2 short chain can hold one of these in a single node; a long chain calls
the stage functions directly and passes
:class:`yolocar.postprocessing.Candidates` between nodes. Everything here is
front-end agnostic -- no transport, no file formats.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
from typing_extensions import Self

from . import inference, postprocessing, preprocessing
from .config import YoloCarConfig
from .engine import TensorRTEngine
from .inference import InputSpec
from .postprocessing import Detection
from .preprocessing import LetterboxInfo

LOGGER = logging.getLogger(__name__)


class YoloCarProcessor:
    """Loads a TensorRT YOLO engine once and runs the stage functions per frame."""

    def __init__(self, config: YoloCarConfig | None = None) -> None:
        self.config = config or YoloCarConfig()
        self._engine: TensorRTEngine | None = None
        self._input_spec: InputSpec | None = None

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
    def input_spec(self) -> InputSpec:
        """Engine input contract: size and dtype the preprocessor must produce."""
        if self._input_spec is None:
            raise RuntimeError("call YoloCarProcessor.load() first")
        return self._input_spec

    @property
    def input_dtype(self) -> np.dtype:
        return self.input_spec.dtype

    @property
    def input_shape(self) -> tuple[int, int]:
        """Engine input size as ``(height, width)``."""
        return self.input_spec.shape

    def load(self) -> None:
        """Deserialize the engine, allocate its buffers, read its I/O contract."""
        if self._engine is not None:
            return
        config = self.config
        size = int(config.image_size)
        engine = TensorRTEngine(
            config.model_path,
            device_index=config.device,
            default_input_shape=(1, 3, size, size),
            pinned_host_memory=config.pinned_host_memory,
            log_level=config.trt_log_level,
        )
        try:
            spec = inference.input_spec(engine)
        except Exception:
            engine.close()
            raise
        self._engine = engine
        self._input_spec = spec
        LOGGER.info(
            "yolocar ready: %s input=%s(1, 3, %d, %d) output=%s",
            config.model_path,
            spec.dtype,
            spec.height,
            spec.width,
            {name: (t.dtype, t.shape) for name, t in engine.outputs.items()},
        )

    def close(self) -> None:
        """Release model resources."""
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._input_spec = None

    def warmup(self, iterations: int = 10) -> None:
        """Run dummy inference, e.g. before an interactive session.

        Not used by the benchmark: it records every frame and lets the analysis
        pick the steady-state window, rather than declaring a warmup boundary
        up front. A 720p grey frame exercises the resize path as well.
        """
        self.load()
        count = int(iterations)
        if count <= 0:
            return
        dummy = np.full((720, 1280, 3), 114, dtype=np.uint8)
        for _ in range(count):
            self.process_frame(dummy)
        LOGGER.debug("warmup finished (%d iterations)", count)

    def __enter__(self) -> Self:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -------------------------------------------------------------- pipeline
    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        """Stage 2: BGR frame -> ``(engine tensor, letterbox geometry)``."""
        spec = self.input_spec
        return preprocessing.preprocess(
            frame,
            spec.shape,
            spec.dtype,
            pad_value=self.config.pad_value,
        )

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Stage 3: run the engine (outputs are views, valid until next call)."""
        return inference.infer(self.engine, tensor)

    def postprocess(
        self,
        outputs: Sequence[np.ndarray] | Mapping[str, np.ndarray],
        letterbox: LetterboxInfo,
    ) -> list[Detection]:
        """Stages 4-6: NMS, domain filter, coordinate adaptation."""
        return postprocessing.postprocess(outputs, letterbox, self.config)

    def process_frame(self, frame: np.ndarray) -> list[Detection]:
        """Full pipeline for a single BGR frame."""
        tensor, letterbox = self.preprocess(frame)
        outputs = self.infer(tensor)
        return self.postprocess(outputs, letterbox)

    __call__ = process_frame


def process_frame(frame: np.ndarray, processor: YoloCarProcessor) -> list[Detection]:
    """Module-level entry point mirroring the other pipeline packages."""
    return processor.process_frame(frame)
