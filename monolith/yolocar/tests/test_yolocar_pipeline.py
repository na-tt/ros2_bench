"""Tests for stages 1-3 (acquisition helpers, preprocessing, inference decode)
plus the single-process facade, none needing a GPU or an engine."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from yolocar.config import COCO_CLASS_NAMES, YoloCarConfig  # noqa: E402
from yolocar.inference import InputSpec, decode_predictions, first_output  # noqa: E402
from yolocar.preprocessing import (  # noqa: E402
    LetterboxInfo,
    as_bgr_frame,
    letterbox,
    preprocess,
    to_tensor,
)
from yolocar.processor import YoloCarProcessor  # noqa: E402

NUM_CLASSES = len(COCO_CLASS_NAMES)
CAR = 2
AIRPLANE = 4


def _raw_output(
    anchors: list[tuple[float, float, float, float, int, float]], total: int = 16
) -> np.ndarray:
    """Build a (1, 4+nc, N) detect-head tensor from (cx, cy, w, h, cls, score)."""
    head = np.zeros((1, 4 + NUM_CLASSES, total), dtype=np.float32)
    for index, (cx, cy, width, height, class_id, score) in enumerate(anchors):
        head[0, 0, index] = cx
        head[0, 1, index] = cy
        head[0, 2, index] = width
        head[0, 3, index] = height
        head[0, 4 + class_id, index] = score
    return head


# ------------------------------------------------------------------ preprocessing
def test_letterbox_preserves_aspect_ratio() -> None:
    padded, info = letterbox(np.zeros((720, 1280, 3), dtype=np.uint8), (640, 640))
    assert padded.shape == (640, 640, 3)
    assert info.scale == pytest.approx(0.5)
    assert info.pad_x == 0
    assert info.pad_y == 140
    assert info.original_shape == (720, 1280)
    assert int(padded[0, 0, 0]) == 114  # padding value (ultralytics default)


def test_letterbox_inverse_maps_boxes_back() -> None:
    _, info = letterbox(np.zeros((720, 1280, 3), dtype=np.uint8), (640, 640))
    boxes = np.array([[288.0, 304.0, 352.0, 336.0]], dtype=np.float32)
    mapped = info.to_source(boxes)
    np.testing.assert_allclose(mapped, [[576.0, 328.0, 704.0, 392.0]], rtol=0, atol=1e-4)


def test_to_tensor_layout_and_scale() -> None:
    image = np.full((8, 4, 3), 255, dtype=np.uint8)
    tensor = to_tensor(image, np.dtype(np.float16))
    assert tensor.shape == (1, 3, 8, 4)
    assert tensor.dtype == np.float16
    assert float(tensor.max()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("frame", "shape"),
    [
        (np.zeros((64, 64), dtype=np.uint8), (64, 64, 3)),  # grayscale promoted
        (np.zeros((32, 32, 3), dtype=np.float32), (32, 32, 3)),  # cast to uint8
        (np.zeros((8, 12, 3), dtype=np.uint8)[::2, ::2], (4, 6, 3)),  # strided
    ],
)
def test_as_bgr_frame_normalises_inputs(frame: np.ndarray, shape: tuple[int, ...]) -> None:
    converted = as_bgr_frame(frame)
    assert converted.shape == shape
    assert converted.dtype == np.uint8


def test_as_bgr_frame_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        as_bgr_frame(np.zeros((10, 10, 4), dtype=np.uint8))


def test_preprocess_returns_tensor_and_geometry() -> None:
    tensor, info = preprocess(
        np.zeros((720, 1280, 3), dtype=np.uint8), (640, 640), np.dtype(np.float16)
    )
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float16
    assert tensor.flags["C_CONTIGUOUS"]
    assert isinstance(info, LetterboxInfo)
    assert info.original_shape == (720, 1280)


# -------------------------------------------------------------------- inference
@pytest.mark.parametrize("transposed", [False, True])
def test_decode_predictions_accepts_both_layouts(transposed: bool) -> None:
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, CAR, 0.9)])
    payload = np.ascontiguousarray(raw.transpose(0, 2, 1)) if transposed else raw
    boxes, scores = decode_predictions(payload, NUM_CLASSES)
    assert boxes.shape[0] == 4
    assert scores.shape[1] == boxes.shape[1]
    assert scores[CAR, 0] == pytest.approx(0.9)


def test_decode_predictions_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="does not match"):
        decode_predictions(np.zeros((1, 4 + NUM_CLASSES + 1, 8), dtype=np.float32), NUM_CLASSES)


def test_first_output_accepts_sequences_and_mappings() -> None:
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, CAR, 0.9)])
    assert first_output([raw]) is raw
    assert first_output({"output0": raw}) is raw
    with pytest.raises(ValueError, match="no outputs"):
        first_output([])


# -------------------------------------------------------------------- facade
def test_processor_constructs_with_default_config() -> None:
    processor = YoloCarProcessor()
    assert isinstance(processor.config, YoloCarConfig)
    assert not processor.is_loaded


def test_processor_requires_load_before_inference() -> None:
    processor = YoloCarProcessor(YoloCarConfig())
    with pytest.raises(RuntimeError):
        processor.process_frame(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError):
        processor.input_spec


def test_processor_postprocess_only_needs_the_letterbox_geometry() -> None:
    """Stage 6 does not need the tensor, so the facade accepts the geometry alone."""
    processor = YoloCarProcessor(YoloCarConfig())
    _, info = letterbox(np.zeros((720, 1280, 3), dtype=np.uint8), (640, 640))
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, CAR, 0.9)])

    detections = processor.postprocess([raw], info)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_name == "car"
    assert detection.class_id == CAR
    assert detection.confidence == pytest.approx(0.9, abs=1e-6)
    assert (detection.x_min, detection.y_min) == pytest.approx((576.0, 328.0), abs=1e-3)
    assert (detection.x_max, detection.y_max) == pytest.approx((704.0, 392.0), abs=1e-3)


def test_processor_domain_filter_drops_other_classes() -> None:
    processor = YoloCarProcessor(YoloCarConfig(target_class_ids=(CAR,)))
    _, info = letterbox(np.zeros((720, 1280, 3), dtype=np.uint8), (640, 640))
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, AIRPLANE, 0.95)])

    assert processor.postprocess([raw], info) == []


def test_config_rejects_unusable_run_lengths() -> None:
    with pytest.raises(ValueError, match="loops"):
        YoloCarConfig(loops=0)
    with pytest.raises(ValueError, match="duration_s"):
        YoloCarConfig(duration_s=0.0)


def test_config_rejects_unusable_class_ids() -> None:
    with pytest.raises(ValueError, match="at least one"):
        YoloCarConfig(target_class_ids=())
    with pytest.raises(ValueError, match=r"must be in \[0, 79\]"):
        YoloCarConfig(target_class_ids=(0, NUM_CLASSES))


class _FakeEngine:
    """Stands in for TensorRTEngine: returns the synthetic head unchanged."""

    def __init__(self, raw: np.ndarray) -> None:
        self.raw = raw
        self.seen: list[np.ndarray] = []

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        self.seen.append(tensor)
        return [self.raw]

    def close(self) -> None:
        return None


def test_processor_wires_the_stages_without_a_real_engine() -> None:
    """The facade must hand the letterbox geometry from stage 2 to stage 6."""
    processor = YoloCarProcessor(YoloCarConfig())
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, CAR, 0.9)])
    processor._engine = _FakeEngine(raw)
    processor._input_spec = InputSpec(640, 640, np.dtype(np.float16))

    detections = processor.process_frame(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert processor._engine.seen[0].shape == (1, 3, 640, 640)
    assert processor._engine.seen[0].dtype == np.float16
    assert len(detections) == 1
    # 640-space box (288..352, 304..336) un-letterboxed into 1280x720 pixels
    assert detections[0].xyxy == pytest.approx((576.0, 328.0, 704.0, 392.0), abs=1e-3)
