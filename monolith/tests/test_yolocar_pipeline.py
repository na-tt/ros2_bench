"""Unit tests for the yolocar pipeline that do not need a GPU or an engine."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from yolocar.config import COCO_CLASS_NAMES, YoloCarConfig  # noqa: E402
from yolocar.processor import (  # noqa: E402
    LetterboxInfo,
    PreprocessedFrame,
    YoloCarProcessor,
    decode_predictions,
    letterbox,
    multiclass_nms,
    nms,
    to_tensor,
    xywh_to_xyxy,
)

NUM_CLASSES = len(COCO_CLASS_NAMES)
CAR = 2
AIRPLANE = 4
TRUCK = 7


def _letterboxed(height: int = 720, width: int = 1280) -> LetterboxInfo:
    _, info = letterbox(np.zeros((height, width, 3), dtype=np.uint8), (640, 640))
    return info


def _raw_output(anchors: list[tuple[float, float, float, float, int, float]], total: int = 16) -> np.ndarray:
    """Build a (1, 4+nc, N) detect-head tensor from (cx, cy, w, h, cls, score)."""
    head = np.zeros((1, 4 + NUM_CLASSES, total), dtype=np.float32)
    for index, (cx, cy, width, height, class_id, score) in enumerate(anchors):
        head[0, 0, index] = cx
        head[0, 1, index] = cy
        head[0, 2, index] = width
        head[0, 3, index] = height
        head[0, 4 + class_id, index] = score
    return head


def test_letterbox_preserves_aspect_ratio() -> None:
    padded, info = letterbox(np.zeros((720, 1280, 3), dtype=np.uint8), (640, 640))
    assert padded.shape == (640, 640, 3)
    assert info.scale == pytest.approx(0.5)
    assert info.pad_x == 0
    assert info.pad_y == 140
    assert info.original_shape == (720, 1280)
    assert int(padded[0, 0, 0]) == 114  # padding value (ultralytics default)


def test_letterbox_inverse_maps_boxes_back() -> None:
    info = _letterboxed()
    boxes = np.array([[288.0, 304.0, 352.0, 336.0]], dtype=np.float32)
    mapped = info.to_source(boxes)
    np.testing.assert_allclose(mapped, [[576.0, 328.0, 704.0, 392.0]], rtol=0, atol=1e-4)


def test_letterbox_clips_to_frame_bounds() -> None:
    info = _letterboxed()
    boxes = np.array([[-100.0, -100.0, 5000.0, 5000.0]], dtype=np.float32)
    mapped = info.to_source(boxes)
    np.testing.assert_allclose(mapped, [[0.0, 0.0, 1280.0, 720.0]])


def test_to_tensor_layout_and_scale() -> None:
    image = np.full((8, 4, 3), 255, dtype=np.uint8)
    tensor = to_tensor(image, np.dtype(np.float16))
    assert tensor.shape == (1, 3, 8, 4)
    assert tensor.dtype == np.float16
    assert float(tensor.max()) == pytest.approx(1.0)


@pytest.mark.parametrize("transposed", [False, True])
def test_decode_predictions_accepts_both_layouts(transposed: bool) -> None:
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, CAR, 0.9)])
    if transposed:
        raw = np.ascontiguousarray(raw.transpose(0, 2, 1))
    boxes, scores = decode_predictions(raw, NUM_CLASSES)
    assert boxes.shape == (4, raw.shape[-1] if not transposed else raw.shape[1])
    assert scores.shape[1] == boxes.shape[1]
    assert scores[CAR, 0] == pytest.approx(0.9)


def test_nms_suppresses_overlapping_boxes() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # heavy overlap with box 0
            [100.0, 100.0, 110.0, 110.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, 0.45)
    assert keep.tolist() == [0, 2]


def test_multiclass_nms_keeps_boxes_of_different_classes() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],  # identical, other class -> kept
            [1.0, 1.0, 11.0, 11.0],  # overlaps box 0 -> dropped
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.85, 0.8], dtype=np.float32)
    classes = np.array([CAR, TRUCK, CAR], dtype=np.int64)
    keep = multiclass_nms(boxes, scores, classes, 0.45, agnostic=False, max_detections=300)
    assert sorted(keep.tolist()) == [0, 1]


def test_xywh_to_xyxy() -> None:
    converted = xywh_to_xyxy(np.array([[50.0, 60.0, 20.0, 10.0]], dtype=np.float32))
    np.testing.assert_allclose(converted, [[40.0, 55.0, 60.0, 65.0]])


def test_postprocess_maps_to_source_pixels() -> None:
    processor = YoloCarProcessor(YoloCarConfig())
    info = _letterboxed()
    prepped = PreprocessedFrame(tensor=np.zeros((1, 3, 640, 640), np.float16), letterbox=info)
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, CAR, 0.9)])

    detections = processor.postprocess([raw], prepped)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_name == "car"
    assert detection.class_id == CAR
    assert detection.confidence == pytest.approx(0.9, abs=1e-6)
    assert (detection.x_min, detection.y_min) == pytest.approx((576.0, 328.0), abs=1e-3)
    assert (detection.x_max, detection.y_max) == pytest.approx((704.0, 392.0), abs=1e-3)


def test_postprocess_domain_filter_drops_other_classes() -> None:
    processor = YoloCarProcessor(YoloCarConfig(target_class_ids=(CAR,)))
    prepped = PreprocessedFrame(tensor=None, letterbox=_letterboxed())
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, AIRPLANE, 0.95)])

    assert processor.postprocess([raw], prepped) == []


def test_postprocess_respects_confidence_threshold() -> None:
    processor = YoloCarProcessor(YoloCarConfig(conf_threshold=0.5))
    prepped = PreprocessedFrame(tensor=None, letterbox=_letterboxed())
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, CAR, 0.4)])

    assert processor.postprocess([raw], prepped) == []


def test_postprocess_accepts_named_outputs() -> None:
    processor = YoloCarProcessor(YoloCarConfig())
    prepped = PreprocessedFrame(tensor=None, letterbox=_letterboxed())
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, CAR, 0.9)])

    assert len(processor.postprocess({"output0": raw}, prepped)) == 1


def test_processor_requires_load_before_inference() -> None:
    processor = YoloCarProcessor(YoloCarConfig())
    with pytest.raises(RuntimeError):
        processor.process_frame(np.zeros((64, 64, 3), dtype=np.uint8))
