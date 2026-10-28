"""Tests for stages 4-6 (NMS, domain filter, coordinate adaptation).

Two kinds of assertions here:

* ``tests/golden/postprocess.json`` freezes the output of the *pre-split*
  implementation on synthetic detect heads. It is the safety net for the
  refactor: the aggregate must still produce byte-identical detections.
* The stage tests pin the properties the ROS 2 chain relies on: each stage is a
  pure ``Candidates -> Candidates`` function, no-ops on empty input, and the
  canonical order is NMS **before** the domain filter (see ``AGENTS.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from yolocar.config import COCO_CLASS_NAMES, ROAD_CLASS_IDS, YoloCarConfig  # noqa: E402
from yolocar.inference import decode_predictions  # noqa: E402
from yolocar.postprocessing import (  # noqa: E402
    Candidates,
    coordinate_adaptation,
    domain_filter,
    nms,
    nms_indices,
    postprocess,
    select_candidates,
    to_detections,
    xywh_to_xyxy,
)
from yolocar.preprocessing import letterbox  # noqa: E402

NUM_CLASSES = len(COCO_CLASS_NAMES)
CAR = 2
TRUCK = 7
PERSON = 0
AIRPLANE = 4

GOLDEN_PATH = Path(__file__).parent / "golden" / "postprocess.json"


def _raw_output(
    anchors: list[tuple[float, float, float, float, int, float]], total: int = 24
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


def _letterboxed(height: int = 720, width: int = 1280):
    _, info = letterbox(np.zeros((height, width, 3), dtype=np.uint8), (640, 640))
    return info


def _candidates(
    boxes: list[list[float]],
    scores: list[float],
    class_ids: list[int],
) -> Candidates:
    return Candidates(
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.asarray(scores, dtype=np.float32),
        class_ids=np.asarray(class_ids, dtype=np.int32),
    )


# --------------------------------------------------------------------- golden
@pytest.mark.parametrize(
    "scenario", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")),
    ids=lambda scenario: scenario["name"],
)
def test_postprocess_matches_frozen_pre_refactor_output(scenario: dict) -> None:
    """The refactor must not change a single detection."""
    config = YoloCarConfig(
        conf_threshold=scenario["conf"],
        iou_threshold=scenario["iou"],
        agnostic_nms=scenario["agnostic"],
        max_detections=scenario["max_det"],
        target_class_ids=ROAD_CLASS_IDS,
    )
    original_shape, input_shape = scenario["letterbox"]
    _, info = letterbox(
        np.zeros((*original_shape, 3), dtype=np.uint8), tuple(input_shape)
    )

    detections = postprocess(
        [_raw_output(scenario["anchors"])], info, config
    )

    assert [detection.as_dict() for detection in detections] == scenario["detections"]


# ------------------------------------------------------------- primitive tests
def test_nms_indices_suppresses_overlapping_boxes() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # heavy overlap with box 0
            [100.0, 100.0, 110.0, 110.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms_indices(boxes, scores, 0.45).tolist() == [0, 2]


def test_xywh_to_xyxy() -> None:
    converted = xywh_to_xyxy(np.array([[50.0, 60.0, 20.0, 10.0]], dtype=np.float32))
    np.testing.assert_allclose(converted, [[40.0, 55.0, 60.0, 65.0]])


# ---------------------------------------------------------------- stage: NMS
def test_nms_keeps_identical_boxes_of_different_classes() -> None:
    candidates = _candidates(
        [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
        [0.9, 0.85, 0.8],
        [CAR, TRUCK, CAR],
    )
    kept = nms(candidates, iou_threshold=0.45)
    np.testing.assert_array_equal(kept.class_ids, [CAR, TRUCK])
    np.testing.assert_allclose(kept.scores, [0.9, 0.85])


def test_nms_agnostic_suppresses_across_classes() -> None:
    candidates = _candidates(
        [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]],
        [0.9, 0.85],
        [CAR, TRUCK],
    )
    assert len(nms(candidates, iou_threshold=0.45, agnostic=True)) == 1


def test_nms_respects_max_detections() -> None:
    candidates = _candidates(
        [[float(i) * 20, 0.0, float(i) * 20 + 10, 10.0] for i in range(5)],
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [CAR] * 5,
    )
    kept = nms(candidates, iou_threshold=0.45, max_detections=2)
    np.testing.assert_allclose(kept.scores, [0.9, 0.8])


# --------------------------------------------------------- stage: domain filter
def test_domain_filter_keeps_requested_ids_in_score_order() -> None:
    candidates = _candidates(
        [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]],
        [0.9, 0.85, 0.8],
        [CAR, AIRPLANE, PERSON],
    )
    kept = domain_filter(candidates, ROAD_CLASS_IDS)
    np.testing.assert_array_equal(kept.class_ids, [CAR, PERSON])
    np.testing.assert_allclose(kept.scores, [0.9, 0.8])


# ---------------------------------------------------- stage: coordinate mapping
def test_coordinate_adaptation_maps_to_source_pixels() -> None:
    info = _letterboxed()
    candidates = _candidates([[288.0, 304.0, 352.0, 336.0]], [0.9], [CAR])
    mapped = coordinate_adaptation(candidates, info)
    np.testing.assert_allclose(
        mapped.boxes, [[576.0, 328.0, 704.0, 392.0]], rtol=0, atol=1e-4
    )
    # scores/classes must survive the transform untouched
    np.testing.assert_allclose(mapped.scores, candidates.scores)
    np.testing.assert_array_equal(mapped.class_ids, candidates.class_ids)


def test_coordinate_adaptation_clips_to_frame_bounds() -> None:
    info = _letterboxed()
    candidates = _candidates([[-100.0, -100.0, 5000.0, 5000.0]], [0.9], [CAR])
    mapped = coordinate_adaptation(candidates, info)
    np.testing.assert_allclose(mapped.boxes, [[0.0, 0.0, 1280.0, 720.0]])


# ------------------------------------------------------------- stage chaining
def test_stages_are_noops_on_empty_candidates() -> None:
    empty = Candidates.empty()
    assert len(nms(empty, iou_threshold=0.45)) == 0
    assert len(domain_filter(empty, ROAD_CLASS_IDS)) == 0
    assert len(coordinate_adaptation(empty, _letterboxed())) == 0
    assert to_detections(empty) == []


def test_candidates_take_preserves_requested_order() -> None:
    candidates = _candidates(
        [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], [0.9, 0.5, 0.7], [CAR, CAR, CAR]
    )
    subset = candidates.take(np.array([2, 0]))
    np.testing.assert_allclose(subset.scores, [0.7, 0.9])


def test_postprocess_equals_the_three_stages_in_sequence() -> None:
    """The aggregate must stay a literal composition of the stage functions."""
    config = YoloCarConfig(conf_threshold=0.25, iou_threshold=0.45)
    info = _letterboxed(1080, 1920)
    raw = _raw_output(
        [
            (320.0, 320.0, 64.0, 32.0, CAR, 0.90),
            (322.0, 321.0, 64.0, 32.0, CAR, 0.80),
            (100.0, 100.0, 40.0, 40.0, TRUCK, 0.70),
            (48.0, 48.0, 32.0, 32.0, AIRPLANE, 0.95),
        ]
    )

    composed = to_detections(
        coordinate_adaptation(
            domain_filter(
                nms(
                    select_candidates(
                        *decode_predictions(raw, NUM_CLASSES),
                        conf_threshold=config.conf_threshold,
                        max_candidates=config.max_candidates,
                    ),
                    iou_threshold=config.iou_threshold,
                    agnostic=config.agnostic_nms,
                    max_detections=config.max_detections,
                ),
                config.target_class_ids,
            ),
            info,
        )
    )

    assert composed == postprocess([raw], info, config)


# ------------------------------------------------------------- canonical order
def test_canonical_order_is_nms_then_domain_filter() -> None:
    """``AGENTS.md`` order: NMS (4) runs on *all* classes, the filter is (5).

    Same geometry, two classes: the higher scoring box belongs to a class the
    domain filter will drop. With agnostic NMS the non-target box must win the
    suppression round and take the car with it -- which can only happen if NMS
    runs before the filter. Per-class NMS leaves both alive, so the car survives.
    """
    anchors = [
        (320.0, 320.0, 64.0, 32.0, CAR, 0.90),
        (320.0, 320.0, 64.0, 32.0, AIRPLANE, 0.95),
    ]
    info = _letterboxed()
    raw = _raw_output(anchors)

    agnostic = postprocess([raw], info, YoloCarConfig(agnostic_nms=True))
    assert agnostic == []

    per_class = postprocess([raw], info, YoloCarConfig(agnostic_nms=False))
    assert [detection.class_id for detection in per_class] == [CAR]


def test_select_candidates_uses_the_best_class_over_all_classes() -> None:
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, AIRPLANE, 0.95)])
    candidates = select_candidates(
        *decode_predictions(raw, NUM_CLASSES), conf_threshold=0.25, max_candidates=100
    )
    np.testing.assert_array_equal(candidates.class_ids, [AIRPLANE])
    np.testing.assert_allclose(candidates.scores, [0.95])


def test_select_candidates_applies_confidence_threshold() -> None:
    raw = _raw_output(
        [
            (320.0, 320.0, 64.0, 32.0, CAR, 0.90),
            (100.0, 100.0, 64.0, 32.0, CAR, 0.24),
        ]
    )
    candidates = select_candidates(
        *decode_predictions(raw, NUM_CLASSES), conf_threshold=0.25, max_candidates=100
    )
    np.testing.assert_allclose(candidates.scores, [0.90])


def test_select_candidates_respects_max_candidates() -> None:
    raw = _raw_output(
        [(float(i) * 10, 100.0, 8.0, 8.0, CAR, 0.9 - 0.05 * i) for i in range(6)]
    )
    candidates = select_candidates(
        *decode_predictions(raw, NUM_CLASSES), conf_threshold=0.25, max_candidates=3
    )
    assert len(candidates) == 3
    # select_candidates is pre-NMS: it keeps the top-K by score, unsorted
    # (NMS sorts). The three highest scores are 0.90, 0.85, 0.80.
    np.testing.assert_allclose(sorted(candidates.scores, reverse=True), [0.9, 0.85, 0.8])


def test_domain_filter_only_and_named_outputs() -> None:
    info = _letterboxed()
    raw = _raw_output([(320.0, 320.0, 64.0, 32.0, CAR, 0.9)])
    assert len(postprocess({"output0": raw}, info, YoloCarConfig())) == 1


def test_postprocess_rejects_empty_outputs() -> None:
    with pytest.raises(ValueError, match="no outputs"):
        postprocess([], _letterboxed(), YoloCarConfig())
