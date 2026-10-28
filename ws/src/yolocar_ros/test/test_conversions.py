"""Round trips through the wire types used by the four-node chain."""

import numpy as np
import pytest

from yolocar.postprocessing import Candidates
from yolocar.preprocessing import LetterboxInfo

from yolocar_interfaces.msg import FrameMeta

from yolocar_ros.conversions import (
    candidates_frame,
    candidates_from_frame,
    frame_from_image,
    frame_meta,
    image_frame,
    letterbox_from_message,
    tensor_frame,
    tensor_from_frame,
)


def _letterbox(original=(720, 1280), input_shape=(640, 640)) -> LetterboxInfo:
    return LetterboxInfo(
        scale=0.5,
        pad_x=0,
        pad_y=140,
        original_shape=original,
        input_shape=input_shape,
    )


def _meta(seq: int = 7) -> FrameMeta:
    return frame_meta(seq=seq, loop=2, t_mono=1234.5, acq_ms=8.25, pre_ms=3.5)


def test_frame_meta_round_trips_identity_and_timings() -> None:
    meta = _meta()
    assert meta.seq == 7
    assert meta.loop == 2
    assert meta.t_mono == pytest.approx(1234.5)
    assert meta.acq_ms == pytest.approx(8.25)
    assert meta.pre_ms == pytest.approx(3.5)
    assert meta.infer_ms == 0.0


def test_image_frame_round_trips_pixels_and_metadata() -> None:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    msg = image_frame(frame, _meta())

    assert (msg.height, msg.width, msg.channels) == (12, 16, 3)
    assert msg.step == 48
    assert msg.encoding == 'bgr8'
    assert msg.meta.seq == 7

    restored = frame_from_image(msg)
    assert restored.shape == frame.shape
    np.testing.assert_array_equal(restored, frame)


def test_image_frame_rejects_non_bgr_input() -> None:
    with pytest.raises(ValueError):
        image_frame(np.zeros((4, 4), dtype=np.uint8), _meta())
    with pytest.raises(ValueError):
        image_frame(np.zeros((4, 4, 3), dtype=np.float32), _meta())


def test_frame_from_image_rejects_truncated_payload() -> None:
    msg = image_frame(np.zeros((4, 4, 3), dtype=np.uint8), _meta())
    msg.data = np.zeros(10, dtype=np.uint8).tobytes()
    with pytest.raises(ValueError):
        frame_from_image(msg)


def test_frame_from_image_rejects_padded_rows() -> None:
    msg = image_frame(np.zeros((4, 4, 3), dtype=np.uint8), _meta())
    msg.step = 16  # rows are padded; this format is not supported
    with pytest.raises(ValueError):
        frame_from_image(msg)


def test_tensor_frame_round_trips_float32_tensor_and_geometry() -> None:
    rng = np.random.default_rng(1)
    tensor = rng.random((1, 3, 640, 640), dtype=np.float32)
    letterbox = _letterbox()

    msg = tensor_frame(tensor, _meta(), letterbox)
    assert (msg.height, msg.width) == (640, 640)
    assert msg.dtype == 'float32'
    assert msg.scale == pytest.approx(0.5)
    assert (msg.pad_x, msg.pad_y) == (0, 140)
    assert (msg.original_height, msg.original_width) == (720, 1280)

    restored = tensor_from_frame(msg)
    assert restored.dtype == np.float32
    assert restored.shape == (1, 3, 640, 640)
    np.testing.assert_array_equal(restored, tensor)

    info = letterbox_from_message(msg)
    assert info == letterbox


def test_tensor_from_frame_rejects_wrong_dtype_payload() -> None:
    msg = tensor_frame(np.zeros((1, 3, 4, 4), dtype=np.float32), _meta(), _letterbox())
    msg.dtype = 'float16'  # payload is still float32-sized
    with pytest.raises(ValueError):
        tensor_from_frame(msg)


def test_candidates_frame_round_trips_arrays() -> None:
    boxes = np.array([[1.0, 2.0, 30.0, 40.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    scores = np.array([0.9, 0.25], dtype=np.float32)
    class_ids = np.array([2, 7], dtype=np.int32)
    candidates = Candidates(boxes=boxes, scores=scores, class_ids=class_ids)

    msg = candidates_frame(candidates, _meta(), _letterbox(), engine_loaded_t=99.5)
    assert len(msg.boxes) == 8
    assert len(msg.scores) == 2
    assert len(msg.class_ids) == 2
    assert msg.engine_loaded_t == pytest.approx(99.5)

    restored, info = candidates_from_frame(msg)
    np.testing.assert_array_equal(restored.boxes, boxes)
    np.testing.assert_array_equal(restored.scores, scores)
    np.testing.assert_array_equal(restored.class_ids, class_ids)
    assert info == _letterbox()


def test_empty_candidates_round_trip() -> None:
    msg = candidates_frame(Candidates.empty(), _meta(), _letterbox())
    assert len(msg.boxes) == 0
    restored, info = candidates_from_frame(msg)
    assert len(restored) == 0
    assert restored.boxes.shape == (0, 4)
    assert info == _letterbox()


def test_candidates_from_frame_rejects_inconsistent_arrays() -> None:
    msg = candidates_frame(Candidates.empty(), _meta(), _letterbox())
    msg.boxes = np.zeros(4, dtype=np.float32)  # one box, but no score/id
    with pytest.raises(ValueError):
        candidates_from_frame(msg)
