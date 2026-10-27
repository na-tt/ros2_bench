"""TensorRT engine wrapper: deserialize once, then run inference per frame.

Design notes
------------
* TensorRT is imported lazily so ``import yolocar`` works on machines without
  the TensorRT Python bindings (e.g. CI linting the monolith package).
* Memory is managed through :mod:`yolocar.cuda` (ctypes -> libcudart), so only
  ``tensorrt`` + ``numpy`` are required.
* Host buffers are page-locked and copies are issued asynchronously on one
  non-blocking stream; :meth:`TensorRTEngine.infer` returns after the stream is
  synchronized, i.e. the outputs are ready.
* ``infer`` returns *views* into the pinned output buffers unless ``copy=True``
  is requested. The data is overwritten by the next call -- consume it
  immediately (which is what the post-processing does).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Optional

import numpy as np

from .cuda import (
    CUDA_MEMCPY_DEVICE_TO_HOST,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    CudaRuntime,
    DeviceBuffer,
    HostBuffer,
    get_runtime,
)

LOGGER = logging.getLogger(__name__)

_TRT_LOG_LEVELS = {"verbose": "VERBOSE", "info": "INFO", "warning": "WARNING", "error": "ERROR"}


@dataclass(frozen=True)
class TensorInfo:
    """Static description of one engine IO tensor."""

    name: str
    is_input: bool
    dtype: np.dtype
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        """Number of elements."""
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        """Size of the tensor in bytes."""
        return self.size * self.dtype.itemsize

    def __repr__(self) -> str:
        kind = "input" if self.is_input else "output"
        return f"TensorInfo({self.name!r}, {kind}, {self.dtype}, {self.shape})"


class TensorRTEngine:
    """A deserialized TensorRT engine plus its execution context and buffers."""

    def __init__(
        self,
        engine_path: str | Path,
        device_index: int = 0,
        *,
        default_input_shape: Sequence[int] | None = None,
        input_shapes: Mapping[str, Sequence[int]] | None = None,
        pinned_host_memory: bool = True,
        log_level: str = "warning",
        cuda: Optional[CudaRuntime] = None,
    ) -> None:
        self.engine_path = Path(engine_path)
        self.device_index = int(device_index)
        self.default_input_shape = (
            tuple(int(d) for d in default_input_shape)
            if default_input_shape is not None
            else None
        )
        self._input_shape_overrides = {
            str(name): tuple(int(d) for d in shape)
            for name, shape in (input_shapes or {}).items()
        }
        self._pinned = bool(pinned_host_memory)
        self._log_level = log_level.lower()
        self._cuda = cuda if cuda is not None else get_runtime()

        # Populated by load()
        self._trt: Any = None
        self._trt_logger: Any = None
        self._trt_runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: int = 0
        self.inputs: dict[str, TensorInfo] = {}
        self.outputs: dict[str, TensorInfo] = {}
        self._device_buffers: dict[str, DeviceBuffer] = {}
        self._host_inputs: dict[str, HostBuffer] = {}
        self._host_outputs: dict[str, HostBuffer] = {}
        self._host_input_views: dict[str, np.ndarray] = {}
        self._host_output_views: dict[str, np.ndarray] = {}
        self._loaded = False

        self.load()

    # ------------------------------------------------------------------ io
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(self.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self.outputs)

    @property
    def single_input(self) -> TensorInfo:
        if len(self.inputs) != 1:
            raise RuntimeError(
                f"engine has {len(self.inputs)} inputs ({sorted(self.inputs)}); "
                "use TensorRTEngine.infer_named"
            )
        return next(iter(self.inputs.values()))

    def describe(self) -> str:
        lines = [f"engine: {self.engine_path} (device {self.device_index})"]
        for info in list(self.inputs.values()) + list(self.outputs.values()):
            lines.append(f"  {info!r}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- load
    def load(self) -> None:
        """Deserialize the engine, create the context, allocate all buffers."""
        if self._loaded:
            return
        if not self.engine_path.exists():
            raise FileNotFoundError(f"engine not found: {self.engine_path}")
        if self.engine_path.suffix.lower() not in {".engine", ".plan", ".trt"}:
            raise ValueError(
                f"{self.engine_path} is not a TensorRT engine. Build one first, e.g.\n"
                f"  trtexec --onnx=assets/models/yolov8n.onnx "
                f"--saveEngine=assets/models/yolov8n_fp16.engine --fp16"
            )

        import tensorrt as trt  # lazy: keeps the package importable without TRT

        self._trt = trt
        level = _TRT_LOG_LEVELS.get(self._log_level, "WARNING")
        self._trt_logger = trt.Logger(getattr(trt.Logger, level))
        serialized = self.engine_path.read_bytes()
        self._trt_runtime = trt.Runtime(self._trt_logger)
        engine = self._trt_runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError(
                f"failed to deserialize {self.engine_path}; the engine was probably "
                "built for a different TensorRT version or GPU architecture"
            )
        self._engine = engine
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"failed to create an execution context for {self.engine_path}")
        self._context = context

        self._cuda.set_device(self.device_index)
        self._stream = self._cuda.stream_create(non_blocking=True)

        self._setup_io()
        self._allocate_buffers()
        self._loaded = True
        LOGGER.debug("%s", self.describe())

    def _setup_io(self) -> None:
        """Read tensor metadata, resolve dynamic shapes, allocate device memory."""
        trt = self._trt
        engine = self._engine
        context = self._context

        resolved_inputs: dict[str, tuple[int, ...]] = {}
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = tuple(int(d) for d in engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))

            if is_input:
                shape = self._resolve_input_shape(name, shape)
                context.set_input_shape(name, shape)
                resolved_inputs[name] = shape
                self.inputs[name] = TensorInfo(name, True, dtype, shape)
            else:
                if any(d < 0 for d in shape):
                    shape = tuple(int(d) for d in context.get_tensor_shape(name))
                if any(d < 0 for d in shape):
                    raise RuntimeError(
                        f"output tensor {name!r} still has dynamic dims {shape}; "
                        "call TensorRTEngine.infer once a valid input shape is set"
                    )
                self.outputs[name] = TensorInfo(name, False, dtype, shape)

        if not self.inputs:
            raise RuntimeError(f"engine {self.engine_path} exposes no input tensors")
        if not self.outputs:
            raise RuntimeError(f"engine {self.engine_path} exposes no output tensors")

    def _resolve_input_shape(
        self, name: str, shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        if name in self._input_shape_overrides:
            return self._input_shape_overrides[name]
        if all(d > 0 for d in shape):
            return shape
        if self.default_input_shape is not None:
            if len(self.default_input_shape) != len(shape):
                raise ValueError(
                    f"cannot resolve dynamic input {name!r} with shape {shape} from "
                    f"default_input_shape={self.default_input_shape}"
                )
            return tuple(
                int(fallback) if dim < 0 else int(dim)
                for dim, fallback in zip(shape, self.default_input_shape)
            )
        raise RuntimeError(
            f"input tensor {name!r} has dynamic shape {shape}; pass "
            "default_input_shape= or input_shapes= when constructing the engine"
        )

    def _allocate_buffers(self) -> None:
        for name, info in self.inputs.items():
            self._device_buffers[name] = DeviceBuffer(self._cuda, info.nbytes)
            host = HostBuffer(self._cuda, info.nbytes, pinned=self._pinned)
            self._host_inputs[name] = host
            self._host_input_views[name] = host.view(info.dtype, info.shape)
        for name, info in self.outputs.items():
            self._device_buffers[name] = DeviceBuffer(self._cuda, info.nbytes)
            host = HostBuffer(self._cuda, info.nbytes, pinned=self._pinned)
            self._host_outputs[name] = host
            self._host_output_views[name] = host.view(info.dtype, info.shape)

        for name, buffer in self._device_buffers.items():
            self._context.set_tensor_address(name, buffer.ptr)

    # --------------------------------------------------------------- infer
    def infer(
        self, input_array: np.ndarray, *, copy: bool = False
    ) -> list[np.ndarray]:
        """Run one inference for a single-input engine.

        Returns the output tensors in engine order. Unless ``copy=True`` they
        are views into the pinned host buffers and are invalidated by the next
        call.
        """
        info = self.single_input
        outputs = self.infer_named({info.name: input_array}, copy=copy)
        return [outputs[name] for name in self.outputs]

    def infer_named(
        self, inputs: Mapping[str, np.ndarray], *, copy: bool = False
    ) -> dict[str, np.ndarray]:
        """Run one inference with explicit input tensors."""
        if not self._loaded:
            raise RuntimeError("engine is not loaded")

        missing = set(self.inputs) - set(inputs)
        if missing:
            raise KeyError(f"missing engine inputs: {sorted(missing)}")
        unexpected = set(inputs) - set(self.inputs)
        if unexpected:
            raise KeyError(f"unknown engine inputs: {sorted(unexpected)}")

        for name, info in self.inputs.items():
            array = inputs[name]
            if array.dtype != info.dtype or tuple(array.shape) != info.shape:
                raise ValueError(
                    f"input {name!r} expects {info.dtype}{info.shape}, "
                    f"got {array.dtype}{tuple(array.shape)}"
                )
            np.copyto(
                self._host_input_views[name].reshape(-1),
                np.ascontiguousarray(array).reshape(-1),
            )
            self._cuda.memcpy_async(
                dst=self._device_buffers[name].ptr,
                src=self._host_inputs[name].ptr,
                nbytes=info.nbytes,
                kind=CUDA_MEMCPY_HOST_TO_DEVICE,
                stream=self._stream,
            )

        self._context.execute_async_v3(stream_handle=self._stream)

        for name, info in self.outputs.items():
            self._cuda.memcpy_async(
                dst=self._host_outputs[name].ptr,
                src=self._device_buffers[name].ptr,
                nbytes=info.nbytes,
                kind=CUDA_MEMCPY_DEVICE_TO_HOST,
                stream=self._stream,
            )

        self._cuda.stream_synchronize(self._stream)

        if copy:
            return {name: view.copy() for name, view in self._host_output_views.items()}
        return dict(self._host_output_views)

    # -------------------------------------------------------------- teardown
    def close(self) -> None:
        """Free every buffer and destroy the context/stream."""
        if not self._loaded:
            return
        self._loaded = False
        for name, buffer in self._device_buffers.items():
            try:
                buffer.free()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                LOGGER.warning("freeing device buffer %s failed: %s", name, exc)
        self._device_buffers.clear()
        for host in list(self._host_inputs.values()) + list(self._host_outputs.values()):
            try:
                host.free()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("freeing host buffer failed: %s", exc)
        self._host_inputs.clear()
        self._host_outputs.clear()
        self._host_input_views.clear()
        self._host_output_views.clear()
        self._context = None
        self._engine = None
        self._trt_runtime = None
        if self._stream:
            try:
                self._cuda.stream_destroy(self._stream)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("destroying the CUDA stream failed: %s", exc)
            self._stream = 0

    def __enter__(self) -> "TensorRTEngine":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown order
            pass

    def __repr__(self) -> str:
        return (
            f"TensorRTEngine({str(self.engine_path)!r}, device={self.device_index}, "
            f"inputs={list(self.inputs)}, outputs={list(self.outputs)})"
        )


def engine_input_names(engine_path: str | Path) -> tuple[str, ...]:
    """Convenience helper: input tensor names of an engine file."""
    engine = TensorRTEngine(engine_path)
    try:
        return engine.input_names
    finally:
        engine.close()
