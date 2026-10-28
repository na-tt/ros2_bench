"""Tests for :mod:`yolocar.engine` driven by a fake TensorRT + fake CUDA runtime.

The engine has two injection points (``TensorRTEngine(cuda=...)`` and the lazy
``import tensorrt``), which is enough to exercise the whole load/infer/close
lifecycle without a GPU. These tests pin down the behaviour that only shows up
on hardware: the device must be bound before the context exists, an
out-of-profile input shape must fail loudly, and closing must release every
buffer exactly once.
"""

from __future__ import annotations

import sys
import types
from typing import Iterator

import numpy as np
import pytest

from yolocar.cuda import CUDA_MEMCPY_DEVICE_TO_HOST, CUDA_MEMCPY_HOST_TO_DEVICE
from yolocar.engine import TensorRTEngine

INPUT_NAME = "images"
OUTPUT_NAME = "output0"
NUM_CHANNELS = 84
ANCHORS = 8


# --------------------------------------------------------------------- fakes
class FakeCudaRuntime:
    """Records CUDA calls and moves bytes between fake allocations."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.allocations: dict[int, np.ndarray] = {}
        self.freed: list[int] = []
        self.executed = 0
        self.stream = 0x5000

    # -- allocation ------------------------------------------------------
    def _allocate(self, nbytes: int) -> int:
        memory = np.zeros(int(nbytes), dtype=np.uint8)
        self.allocations[int(memory.ctypes.data)] = memory
        return int(memory.ctypes.data)

    def malloc(self, nbytes: int) -> int:
        self.events.append("cudaMalloc")
        return self._allocate(nbytes)

    def malloc_host(self, nbytes: int) -> int:
        self.events.append("cudaMallocHost")
        return self._allocate(nbytes)

    def free(self, pointer: int) -> None:
        self.events.append("cudaFree")
        self.freed.append(pointer)
        self.allocations.pop(pointer, None)

    def free_host(self, pointer: int) -> None:
        self.events.append("cudaFreeHost")
        self.freed.append(pointer)
        self.allocations.pop(pointer, None)

    # -- copies ----------------------------------------------------------
    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: int, stream: int) -> None:
        assert stream == self.stream
        assert kind in {CUDA_MEMCPY_HOST_TO_DEVICE, CUDA_MEMCPY_DEVICE_TO_HOST}
        self.events.append("cudaMemcpyAsync")
        self.allocations[dst][:nbytes] = self.allocations[src][:nbytes]

    # -- device / streams -------------------------------------------------
    def set_device(self, index: int) -> None:
        self.events.append(f"set_device:{index}")

    def stream_create(self, non_blocking: bool = True) -> int:
        self.events.append("cudaStreamCreate")
        return self.stream

    def stream_destroy(self, stream: int) -> None:
        self.events.append("cudaStreamDestroy")

    def stream_synchronize(self, stream: int) -> None:
        self.events.append("cudaStreamSynchronize")


class FakeContext:
    def __init__(self, engine: "FakeTRTEngine") -> None:
        self.engine = engine
        self.input_shapes: dict[str, tuple[int, ...]] = {}
        self.addresses: dict[str, int] = {}
        self.set_input_shape_calls: list[tuple[str, tuple[int, ...]]] = []

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> bool:
        self.set_input_shape_calls.append((name, tuple(shape)))
        if not self.engine.accept_input_shape:
            return False
        self.input_shapes[name] = tuple(int(d) for d in shape)
        return True

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        if name in self.input_shapes:
            return self.input_shapes[name]
        batch = self.input_shapes.get(INPUT_NAME, (1,))[0]
        return (int(batch), NUM_CHANNELS, ANCHORS)

    def set_tensor_address(self, name: str, pointer: int) -> None:
        self.addresses[name] = pointer

    def execute_async_v3(self, stream_handle: int) -> bool:
        runtime = self.engine.runtime
        runtime.executed += 1
        runtime.events.append("execute_async_v3")
        # Pretend the network wrote its output: fill the registered buffer.
        output = runtime.allocations[self.addresses[OUTPUT_NAME]]
        output.view(np.float32)[...] = 1.5
        return True


class FakeTRTEngine:
    def __init__(
        self,
        runtime: FakeCudaRuntime,
        *,
        accept_input_shape: bool = True,
        deserializes: bool = True,
    ) -> None:
        self.runtime = runtime
        self.accept_input_shape = accept_input_shape
        self.deserializes = deserializes
        self.context: FakeContext | None = None
        self.num_io_tensors = 2
        self._shapes = {INPUT_NAME: (-1, 3, -1, -1), OUTPUT_NAME: (-1, NUM_CHANNELS, -1)}

    def get_tensor_name(self, index: int) -> str:
        return (INPUT_NAME, OUTPUT_NAME)[index]

    def get_tensor_mode(self, name: str) -> str:
        return "INPUT" if name == INPUT_NAME else "OUTPUT"

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self._shapes[name]

    def get_tensor_dtype(self, name: str) -> object:
        return np.float32

    def create_execution_context(self) -> FakeContext | None:
        self.runtime.events.append("create_execution_context")
        if not self.deserializes:
            return None
        self.context = FakeContext(self)
        return self.context


class FakeRuntime:
    def __init__(self, trt: "FakeTRTModule") -> None:
        self._trt = trt

    def deserialize_cuda_engine(self, blob: bytes) -> FakeTRTEngine | None:
        self._trt.cuda.events.append("deserialize_cuda_engine")
        assert blob, "engine blob must not be empty"
        return self._trt.engine


class FakeTRTModule(types.ModuleType):
    class TensorIOMode:  # noqa: D106 - mirrors tensorrt.TensorIOMode
        INPUT = "INPUT"
        OUTPUT = "OUTPUT"

    def __init__(self, cuda: FakeCudaRuntime) -> None:
        super().__init__("tensorrt")
        self.cuda = cuda
        self.logger_levels: list[str] = []
        self.plugin_calls: list[tuple[object, str]] = []
        self.engine = FakeTRTEngine(cuda)

    # -- API used by TensorRTEngine.load --------------------------------
    class Logger:  # noqa: D106 - mirrors tensorrt.Logger
        VERBOSE = "VERBOSE"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"

        def __init__(self, level: str) -> None:
            self.level = level

    def Runtime(self, logger: object) -> FakeRuntime:  # noqa: N802 - TRT naming
        self.logger_levels.append(getattr(logger, "level", "?"))
        return FakeRuntime(self)

    def init_libnvinfer_plugins(self, logger: object, namespace: str) -> None:
        self.plugin_calls.append((logger, namespace))

    @staticmethod
    def nptype(dtype: object) -> object:
        return np.dtype(dtype)


@pytest.fixture()
def fake_trt(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeTRTModule]:
    holder: dict[str, FakeTRTModule] = {}

    def _make(cuda: FakeCudaRuntime) -> FakeTRTModule:
        module = FakeTRTModule(cuda)
        monkeypatch.setitem(sys.modules, "tensorrt", module)
        holder["module"] = module
        return module

    yield _make  # type: ignore[misc]


def _engine_file(tmp_path, name: str = "model.engine"):
    path = tmp_path / name
    path.write_bytes(b"not-a-real-engine")
    return path


def _make(
    tmp_path,
    fake_trt,
    *,
    accept_input_shape: bool = True,
    deserializes: bool = True,
    **kwargs,
) -> tuple[TensorRTEngine, FakeTRTModule, FakeCudaRuntime]:
    cuda = FakeCudaRuntime()
    module = fake_trt(cuda)
    module.engine = FakeTRTEngine(
        cuda, accept_input_shape=accept_input_shape, deserializes=deserializes
    )
    engine = TensorRTEngine(
        _engine_file(tmp_path),
        default_input_shape=(1, 3, 4, 4),
        cuda=cuda,  # type: ignore[arg-type]
        **kwargs,
    )
    return engine, module, cuda


# ---------------------------------------------------------------------- load
def test_device_is_bound_before_the_context_is_created(tmp_path, fake_trt) -> None:
    engine, _, cuda = _make(tmp_path, fake_trt, device_index=1)
    try:
        events = cuda.events
        assert "set_device:1" in events
        assert events.index("set_device:1") < events.index("create_execution_context")
        assert events.index("set_device:1") < events.index("cudaMalloc")
    finally:
        engine.close()


def test_plugins_are_initialised_before_deserialization(tmp_path, fake_trt) -> None:
    engine, module, _ = _make(tmp_path, fake_trt)
    try:
        assert module.plugin_calls, "init_libnvinfer_plugins must be called"
        assert module.plugin_calls[0][1] == ""
    finally:
        engine.close()


def test_tensor_metadata_and_dynamic_output_shape(tmp_path, fake_trt) -> None:
    engine, _, _ = _make(tmp_path, fake_trt)
    try:
        assert engine.input_names == (INPUT_NAME,)
        assert engine.output_names == (OUTPUT_NAME,)
        info = engine.single_input
        assert info.shape == (1, 3, 4, 4)
        assert info.dtype == np.float32
        assert info.nbytes == 1 * 3 * 4 * 4 * 4
        assert engine.outputs[OUTPUT_NAME].shape == (1, NUM_CHANNELS, ANCHORS)
        assert "output0" in engine.describe()
    finally:
        engine.close()


def test_out_of_profile_input_shape_is_rejected(tmp_path, fake_trt) -> None:
    with pytest.raises(RuntimeError, match="optimization profile"):
        _make(tmp_path, fake_trt, accept_input_shape=False)


def test_unknown_log_level_is_rejected(tmp_path, fake_trt) -> None:
    with pytest.raises(ValueError, match="trt_log_level"):
        _make(tmp_path, fake_trt, log_level="chatty")


def test_negative_device_index_is_rejected(tmp_path, fake_trt) -> None:
    cuda = FakeCudaRuntime()
    fake_trt(cuda)
    with pytest.raises(ValueError, match="device_index"):
        TensorRTEngine(_engine_file(tmp_path), device_index=-1, cuda=cuda)  # type: ignore[arg-type]


def test_dynamic_input_without_a_default_shape_is_rejected(tmp_path, fake_trt) -> None:
    cuda = FakeCudaRuntime()
    module = fake_trt(cuda)
    module.engine = FakeTRTEngine(cuda)
    with pytest.raises(RuntimeError, match="dynamic shape"):
        TensorRTEngine(_engine_file(tmp_path), cuda=cuda)  # type: ignore[arg-type]


def test_missing_engine_file_reports_the_path(tmp_path, fake_trt) -> None:
    fake_trt(FakeCudaRuntime())
    with pytest.raises(FileNotFoundError, match="engine not found"):
        TensorRTEngine(tmp_path / "nope.engine")


def test_non_engine_suffix_is_rejected(tmp_path, fake_trt) -> None:
    fake_trt(FakeCudaRuntime())
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="not a TensorRT engine"):
        TensorRTEngine(onnx)


# --------------------------------------------------------------------- infer
def test_infer_roundtrip_moves_data_through_the_device(tmp_path, fake_trt) -> None:
    engine, _, cuda = _make(tmp_path, fake_trt)
    try:
        tensor = np.arange(1 * 3 * 4 * 4, dtype=np.float32).reshape(1, 3, 4, 4) / 100.0
        outputs = engine.infer(tensor)

        assert len(outputs) == 1
        assert outputs[0].shape == (1, NUM_CHANNELS, ANCHORS)
        # The fake network writes 1.5 into the device output buffer...
        np.testing.assert_allclose(outputs[0], 1.5)
        # ...and the input really went H2D, byte for byte.
        device_input = cuda.allocations[engine._device_buffers[INPUT_NAME].ptr]
        np.testing.assert_array_equal(
            device_input.view(np.float32).reshape(tensor.shape), tensor
        )
        assert cuda.executed == 1
        assert "cudaStreamSynchronize" in cuda.events
    finally:
        engine.close()


def test_infer_named_returns_owned_copies_on_request(tmp_path, fake_trt) -> None:
    engine, _, _ = _make(tmp_path, fake_trt)
    try:
        tensor = np.zeros((1, 3, 4, 4), dtype=np.float32)
        view = engine.infer(tensor, copy=False)[0]
        owned = engine.infer(tensor, copy=True)[0]
        assert view.base is not None  # a view into the host buffer
        assert owned.base is None  # an owned copy
        np.testing.assert_allclose(owned, view)
    finally:
        engine.close()


def test_infer_rejects_wrong_dtype_and_shape(tmp_path, fake_trt) -> None:
    engine, _, _ = _make(tmp_path, fake_trt)
    try:
        with pytest.raises(ValueError, match="expects"):
            engine.infer(np.zeros((1, 3, 4, 4), dtype=np.float16))
        with pytest.raises(ValueError, match="expects"):
            engine.infer(np.zeros((2, 3, 4, 4), dtype=np.float32))
    finally:
        engine.close()


def test_infer_named_rejects_unknown_or_missing_inputs(tmp_path, fake_trt) -> None:
    engine, _, _ = _make(tmp_path, fake_trt)
    try:
        tensor = np.zeros((1, 3, 4, 4), dtype=np.float32)
        with pytest.raises(KeyError, match="missing engine inputs"):
            engine.infer_named({})
        with pytest.raises(KeyError, match="unknown engine inputs"):
            engine.infer_named({INPUT_NAME: tensor, "bogus": tensor})
    finally:
        engine.close()


# ----------------------------------------------------------------- close/lifecycle
def test_close_releases_every_buffer_exactly_once(tmp_path, fake_trt) -> None:
    engine, _, cuda = _make(tmp_path, fake_trt)
    host_buffers = list(engine._host_inputs.values()) + list(engine._host_outputs.values())
    pointers = [buffer.ptr for buffer in engine._device_buffers.values()]
    pointers += [buffer.ptr for buffer in host_buffers]

    engine.close()
    assert not engine.is_loaded
    assert engine._context is None
    assert sorted(cuda.freed) == sorted(pointers)
    assert all(buffer.closed for buffer in host_buffers)
    with pytest.raises(Exception, match="already been released"):
        host_buffers[0].array
    assert "cudaStreamDestroy" in cuda.events

    engine.close()  # idempotent
    assert sorted(cuda.freed) == sorted(pointers)


def test_infer_after_close_raises(tmp_path, fake_trt) -> None:
    engine, _, _ = _make(tmp_path, fake_trt)
    engine.close()
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.infer(np.zeros((1, 3, 4, 4), dtype=np.float32))


def test_context_manager_closes(tmp_path, fake_trt) -> None:
    cuda = FakeCudaRuntime()
    module = fake_trt(cuda)
    module.engine = FakeTRTEngine(cuda)
    with TensorRTEngine(
        _engine_file(tmp_path), default_input_shape=(1, 3, 4, 4), cuda=cuda  # type: ignore[arg-type]
    ) as engine:
        assert engine.is_loaded
    assert not engine.is_loaded


def test_engine_input_names_does_not_touch_the_device(tmp_path, fake_trt) -> None:
    from yolocar.engine import engine_input_names

    cuda = FakeCudaRuntime()
    module = fake_trt(cuda)
    module.engine = FakeTRTEngine(cuda)

    assert engine_input_names(_engine_file(tmp_path)) == (INPUT_NAME,)
    # Metadata only: deserialization but no device, stream or buffers.
    assert cuda.events == ["deserialize_cuda_engine"]
    assert module.plugin_calls, "plugins must be registered before deserializing"


def test_engine_input_names_reports_deserialization_failure(tmp_path, fake_trt) -> None:
    from yolocar.engine import engine_input_names

    cuda = FakeCudaRuntime()
    module = fake_trt(cuda)
    module.engine = None  # simulate an engine built for another TRT version

    with pytest.raises(RuntimeError, match="failed to deserialize"):
        engine_input_names(_engine_file(tmp_path))


def test_unresolvable_dynamic_output_is_reported(tmp_path, fake_trt) -> None:
    cuda = FakeCudaRuntime()
    module = fake_trt(cuda)
    engine = FakeTRTEngine(cuda)
    engine._shapes[OUTPUT_NAME] = (-1, -1, -1)
    engine.create_execution_context = lambda: (  # type: ignore[method-assign]
        _StuckContext(engine)
    )
    module.engine = engine
    with pytest.raises(RuntimeError, match="still has dynamic dims"):
        TensorRTEngine(_engine_file(tmp_path), default_input_shape=(1, 3, 4, 4), cuda=cuda)  # type: ignore[arg-type]
    # The failure happens after the stream exists, so load() must have released
    # it again instead of leaking a stream per attempt.
    assert "cudaStreamCreate" in cuda.events
    assert "cudaStreamDestroy" in cuda.events


class _StuckContext(FakeContext):
    """A context that never resolves the output shape (broken profile)."""

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return (-1, -1, -1)
