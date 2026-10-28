"""End-to-end tests against the real CUDA + TensorRT stack.

These are the counterweight to :mod:`tests.test_yolocar_engine` /
:mod:`tests.test_yolocar_cuda`: those prove the *logic* (ordering, lifetime,
error paths) with fakes, these prove that the logic matches what the driver
actually does. They load a real ``.engine`` file, push real frames through
``cudaMallocHost``/``cudaMemcpyAsync``/``IExecutionContext`` and assert on the
real outputs, so they only run where the runtime can work:

    a built engine (``assets/models/yolov8n_fp16.engine`` by default, override
    with ``$YOLOCAR_ENGINE``), ``tensorrt``, ``cv2`` and a usable CUDA device.

Where any of that is missing the whole module skips with a reason. Forcing a
skip on a machine that *does* have the hardware:

    YOLOCAR_SKIP_HARDWARE=1 pytest tests/
    pytest tests/ -m "not hardware"

Nothing here asserts on timing: latency belongs to the benchmark (``main.py``),
not to a pass/fail suite, and a busy Jetson would make it flaky.

What still belongs to the fake suite: anything that is unobservable on a
single-GPU host. Binding the device *before* ``create_execution_context()`` is
the example -- on one GPU both orders work, so only the event-log assertion in
``test_yolocar_engine.py`` can catch a regression there.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

try:
    import cv2

    from yolocar.config import COCO_CLASS_NAMES, ROAD_CLASS_IDS, YoloCarConfig
    from yolocar.cuda import (
        CUDA_MEMCPY_DEVICE_TO_HOST,
        CUDA_MEMCPY_HOST_TO_DEVICE,
        CudaError,
        DeviceBuffer,
        HostBuffer,
        get_runtime,
    )
    from yolocar.engine import TensorRTEngine, engine_input_names
    from yolocar.main import main as yolocar_main
    from yolocar.processor import YoloCarProcessor
except ImportError as exc:  # pragma: no cover - depends on the host
    pytest.skip(
        f"the hardware suite needs the yolocar runtime deps ({exc.name})",
        allow_module_level=True,
    )

pytestmark = pytest.mark.hardware

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = Path(
    os.environ.get("YOLOCAR_ENGINE", REPO_ROOT / "assets/models/yolov8n_fp16.engine")
).expanduser()
IMAGE_SIZE = 640
TARGET_IDS = frozenset(ROAD_CLASS_IDS)
_DEFAULT_INPUT_SHAPE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)


# --------------------------------------------------------------- availability
def _skip_reason() -> str | None:
    """Return why the real stack cannot be exercised here, or ``None``."""
    if os.environ.get("YOLOCAR_SKIP_HARDWARE"):
        return "YOLOCAR_SKIP_HARDWARE is set"
    if not ENGINE_PATH.is_file():
        return f"no TensorRT engine at {ENGINE_PATH} (build one or set $YOLOCAR_ENGINE)"
    for module in ("cv2", "tensorrt"):
        if importlib.util.find_spec(module) is None:
            return f"{module} is not installed"
    try:
        runtime = get_runtime()
        runtime.set_device(0)
        if runtime.get_device() != 0:
            return "CUDA device 0 is not usable"
    except CudaError as exc:  # no driver, no device, headless container, ...
        return f"CUDA runtime unusable: {exc}"
    return None


_SKIP_REASON = _skip_reason()


@pytest.fixture(scope="module", autouse=True)
def _require_hardware() -> None:
    if _SKIP_REASON:
        pytest.skip(_SKIP_REASON)


# -------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def engine() -> Iterator[TensorRTEngine]:
    """The real engine, loaded once for the whole module."""
    instance = TensorRTEngine(
        ENGINE_PATH, device_index=0, default_input_shape=_DEFAULT_INPUT_SHAPE
    )
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture(scope="module")
def processor() -> Iterator[YoloCarProcessor]:
    """The full pipeline, loaded and warmed up once."""
    instance = YoloCarProcessor(
        YoloCarConfig(model_path=ENGINE_PATH, image_size=IMAGE_SIZE)
    )
    instance.load()
    instance.warmup(2)
    try:
        yield instance
    finally:
        instance.close()


def _frames() -> list[np.ndarray]:
    """Deterministic synthetic frames covering the interesting geometries."""
    rng = np.random.default_rng(20240)
    return [
        rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8),  # 16:9 -> vertical padding
        np.full((1080, 1920, 3), 90, dtype=np.uint8),  # flat frame
        rng.integers(0, 256, (361, 640, 3), dtype=np.uint8),  # odd height, square-ish
    ]


def _zero_input(engine: TensorRTEngine) -> np.ndarray:
    info = engine.single_input
    return np.zeros(info.shape, dtype=info.dtype)


# ------------------------------------------------------------------- metadata
def test_engine_exposes_sane_real_io_metadata(engine: TensorRTEngine) -> None:
    info = engine.single_input
    assert engine.input_names == (info.name,)
    assert len(info.shape) == 4, "expected NCHW input"
    assert info.shape[1] == 3
    assert info.shape[0] == 1
    assert info.dtype in (np.dtype(np.float16), np.dtype(np.float32))
    assert info.nbytes == int(np.prod(info.shape)) * info.dtype.itemsize

    assert engine.outputs, "engine exposes no outputs"
    name, output = next(iter(engine.outputs.items()))
    assert all(dim > 0 for dim in output.shape), f"{name} still has dynamic dims"
    assert output.nbytes == int(np.prod(output.shape)) * output.dtype.itemsize
    # YOLOv8 detect head: 4 box coords + one score per class.
    assert output.shape[1] == 4 + len(COCO_CLASS_NAMES), (
        f"unexpected head layout {output.shape}"
    )
    assert output.shape[2] > 0


def test_metadata_only_helper_matches_the_loaded_engine(engine: TensorRTEngine) -> None:
    # engine_input_names() deserializes without a context or any device buffer;
    # real TensorRT is the only thing that can confirm that stays in sync.
    assert engine_input_names(ENGINE_PATH) == engine.input_names


# ------------------------------------------------------------------ inference
def test_real_inference_produces_finite_output_of_declared_shape(
    engine: TensorRTEngine,
) -> None:
    name, declared = next(iter(engine.outputs.items()))
    outputs = engine.infer(_zero_input(engine))

    assert len(outputs) == len(engine.outputs)
    tensor = outputs[0]
    assert tensor.shape == declared.shape
    assert tensor.dtype == declared.dtype
    assert np.isfinite(tensor).all()
    # YOLOv8 emits class scores; they must be probabilities-ish, not garbage
    # reinterpreted bytes (which is what a wrong D2H copy would look like).
    assert float(tensor.max()) < 1e3
    assert float(np.abs(tensor).max()) > 0.0


def test_real_inference_is_deterministic(engine: TensorRTEngine) -> None:
    tensor = _zero_input(engine)
    first = engine.infer(tensor, copy=True)[0]
    second = engine.infer(tensor, copy=True)[0]
    np.testing.assert_array_equal(first, second)


def test_input_data_actually_reaches_the_device(engine: TensorRTEngine) -> None:
    # Identical buffers would be a classic symptom of a broken H2D copy or of
    # reads happening before the copy is ordered: compare two distinct inputs.
    zeros = _zero_input(engine)
    rng = np.random.default_rng(7)
    noise = rng.random(zeros.shape).astype(zeros.dtype)
    if zeros.dtype == np.uint8:  # pragma: no cover - TRT IO for YOLO is float
        noise = (noise * 255).astype(np.uint8)

    quiet = engine.infer(zeros, copy=True)[0]
    busy = engine.infer(noise, copy=True)[0]
    difference = float(np.abs(quiet.astype(np.float64) - busy.astype(np.float64)).max())
    assert difference > 1e-3, "the output did not react to the input at all"


def test_views_are_reused_while_copies_are_owned(engine: TensorRTEngine) -> None:
    tensor = _zero_input(engine)
    first = engine.infer(tensor)[0]
    second = engine.infer(tensor)[0]
    assert first.__array_interface__["data"][0] == second.__array_interface__["data"][0]

    owned = engine.infer(tensor, copy=True)[0]
    assert owned.base is None
    np.testing.assert_array_equal(owned, second)


def test_wrong_input_shape_is_rejected_by_the_real_engine(
    engine: TensorRTEngine,
) -> None:
    info = engine.single_input
    wrong = np.zeros((1, 3, info.shape[2] * 2, info.shape[3] * 2), dtype=info.dtype)
    with pytest.raises(ValueError, match="expects"):
        engine.infer(wrong)


def test_mismatched_default_shape_is_not_silently_applied() -> None:
    doubled = (1, 3, IMAGE_SIZE * 2, IMAGE_SIZE * 2)
    with TensorRTEngine(
        ENGINE_PATH, device_index=0, default_input_shape=doubled
    ) as engine:
        info = engine.single_input
        if info.shape[2:] == doubled[2:]:
            pytest.skip("engine has a dynamic profile that honours the override")
        # A static engine keeps its built shape (the fallback only resolves -1
        # dims), so the requested size surfaces as a clear error at infer time
        # instead of a wrongly sized buffer.
        assert engine.outputs
        with pytest.raises(ValueError, match="expects"):
            engine.infer(np.zeros(doubled, dtype=info.dtype))


# ----------------------------------------------------- pinned vs pageable path
def test_pinned_and_pageable_host_memory_produce_identical_output(
    engine: TensorRTEngine,
) -> None:
    pageable = TensorRTEngine(
        ENGINE_PATH,
        device_index=0,
        default_input_shape=_DEFAULT_INPUT_SHAPE,
        pinned_host_memory=False,
    )
    try:
        assert all(not host.pinned for host in pageable._host_outputs.values())
        tensor = _zero_input(engine)
        pinned_out = engine.infer(tensor, copy=True)[0]
        pageable_out = pageable.infer(tensor.copy(), copy=True)[0]
        np.testing.assert_array_equal(pinned_out, pageable_out)
    finally:
        pageable.close()


# ------------------------------------------------------------------ lifecycle
def test_repeated_load_and_close_cycles() -> None:
    # The benchmark loads and closes engines repeatedly; the real driver is the
    # only place where a leaked stream / pinned allocation shows up.
    for cycle in range(3):
        engine = TensorRTEngine(
            ENGINE_PATH, device_index=0, default_input_shape=_DEFAULT_INPUT_SHAPE
        )
        try:
            assert engine.is_loaded, f"cycle {cycle}"
            outputs = engine.infer(_zero_input(engine))
            assert np.isfinite(outputs[0]).all()
        finally:
            engine.close()
        assert not engine.is_loaded
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.infer(np.zeros(_DEFAULT_INPUT_SHAPE, dtype=np.float32))


def test_invalid_device_index_fails_with_a_clear_cuda_error() -> None:
    # A bad ordinal must surface as a CudaError naming the operation instead of
    # an opaque TensorRT failure later on. (Whether the bind happens *before*
    # context creation is unobservable with one GPU; that is pinned by
    # tests/test_yolocar_engine.py::test_device_is_bound_before_the_context_is_created.)
    with pytest.raises(CudaError, match="cudaSetDevice"):
        TensorRTEngine(ENGINE_PATH, device_index=99, default_input_shape=_DEFAULT_INPUT_SHAPE)


# --------------------------------------------------------- cuda layer, for real
def test_pinned_host_buffer_round_trip_on_the_real_driver() -> None:
    runtime = get_runtime()
    runtime.set_device(0)

    buffer = HostBuffer(runtime, 4096, pinned=True)
    assert buffer.pinned, "cudaMallocHost failed on a device that should support it"
    view = buffer.view(np.float32, (1024,))
    payload = np.arange(1024, dtype=np.float32)
    view[:] = payload
    np.testing.assert_array_equal(view, payload)

    buffer.free()
    assert buffer.closed and buffer.ptr == 0
    with pytest.raises(CudaError, match="already been released"):
        buffer.array


def test_synchronous_memcpy_round_trip_on_the_real_driver() -> None:
    # Exercises the ctypes bindings themselves (argtypes, size in bytes, copy
    # direction) rather than TensorRT's use of them.
    runtime = get_runtime()
    runtime.set_device(0)

    payload = np.arange(256, dtype=np.uint8)
    host_in = HostBuffer(runtime, payload.nbytes)
    host_out = HostBuffer(runtime, payload.nbytes)
    device = DeviceBuffer(runtime, payload.nbytes)
    try:
        np.copyto(host_in.array, payload)
        runtime.memcpy(
            device.ptr, host_in.ptr, payload.nbytes, CUDA_MEMCPY_HOST_TO_DEVICE
        )
        runtime.memcpy(
            host_out.ptr, device.ptr, payload.nbytes, CUDA_MEMCPY_DEVICE_TO_HOST
        )
        np.testing.assert_array_equal(host_out.array, payload)
    finally:
        device.free()
        host_in.free()
        host_out.free()


# -------------------------------------------------------------- full pipeline
def test_full_pipeline_invariants_on_synthetic_frames(
    processor: YoloCarProcessor,
) -> None:
    config = processor.config
    assert processor.input_dtype in (np.dtype(np.float16), np.dtype(np.float32))

    for frame in _frames():
        detections = processor.process_frame(frame)
        height, width = frame.shape[:2]

        assert len(detections) <= config.max_detections
        confidences = [detection.confidence for detection in detections]
        assert confidences == sorted(confidences, reverse=True)

        for detection in detections:
            assert detection.class_id in TARGET_IDS, detection
            assert config.conf_threshold <= detection.confidence <= 1.0
            assert detection.class_name
            assert 0.0 <= detection.x_min <= detection.x_max <= float(width)
            assert 0.0 <= detection.y_min <= detection.y_max <= float(height)


def test_preprocess_matches_the_engine_contract(processor: YoloCarProcessor) -> None:
    info = processor.engine.single_input
    preprocessed = processor.preprocess(_frames()[0])

    assert preprocessed.tensor.shape == info.shape
    assert preprocessed.tensor.dtype == info.dtype
    assert preprocessed.tensor.flags["C_CONTIGUOUS"]
    if preprocessed.tensor.dtype != np.dtype(np.uint8):
        assert 0.0 <= float(preprocessed.tensor.min())
        assert float(preprocessed.tensor.max()) <= 1.0
    assert preprocessed.letterbox.input_shape == (info.shape[2], info.shape[3])


def test_pipeline_accepts_grayscale_strided_and_fortran_frames(
    processor: YoloCarProcessor,
) -> None:
    frame = _frames()[0]
    for variant in (
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        frame[::2, ::2],
        np.asfortranarray(frame),
        frame.astype(np.float32),  # not uint8: preprocess clips/casts
    ):
        processor.process_frame(variant)  # must not raise


# ----------------------------------------------------------------------- CLI
def test_cli_benchmark_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "frames"
    source.mkdir()
    for index, frame in enumerate(_frames()):
        assert cv2.imwrite(str(source / f"frame_{index:03d}.jpg"), frame)

    json_path = tmp_path / "summary.json"
    annotated = tmp_path / "annotated"

    code = yolocar_main(
        [
            "--model",
            str(ENGINE_PATH),
            "--source",
            str(source),
            "--json",
            str(json_path),
            "--output",
            str(annotated),
            "--max-frames",
            "2",
            "--warmup",
            "1",
            "--max-det",
            "50",
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 0

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["pipeline"] == "yolocar"
    assert summary["frames"] == 2
    assert summary["warmup_iterations"] == 1
    assert summary["config"]["target_class_ids"] == list(ROAD_CLASS_IDS)
    assert summary["config"]["image_size"] == IMAGE_SIZE
    assert summary["config"]["device"] == 0
    assert set(summary["latency_ms"]) == {"mean", "std", "median", "p95", "min", "max"}
    assert set(summary["stage_ms"]) == {"preprocess", "inference", "postprocess"}
    for stage in ("preprocess", "inference", "postprocess"):
        assert summary["stage_ms"][stage]["mean"] > 0.0
    assert summary["detections"]["total"] >= 0
    assert summary["detections"]["per_frame_max"] >= 0
    assert summary["fps"] >= 0.0

    written = sorted(annotated.glob("*.jpg"))
    assert len(written) == 2, f"expected 2 annotated frames, got {written}"
