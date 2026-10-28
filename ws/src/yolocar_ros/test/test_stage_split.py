"""
Hardware test: the four-node split must not change what the pipeline finds.

This does not start ROS nodes. It walks the same stage functions and message
round trips the nodes use (image -> tensor -> candidates -> detections) and
compares the result with the monolith facade on a real frame. Skipped when the
engine, the clip or TensorRT is unavailable.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('tensorrt')

from yolocar.acquisition import open_source  # noqa: E402
from yolocar.config import COCO_CLASS_NAMES, YoloCarConfig  # noqa: E402
from yolocar.inference import decode_predictions, first_output, infer  # noqa: E402
from yolocar.postprocessing import (  # noqa: E402
    coordinate_adaptation,
    domain_filter,
    nms,
    select_candidates,
    to_detections,
)
from yolocar.preprocessing import preprocess  # noqa: E402
from yolocar.processor import YoloCarProcessor  # noqa: E402

from yolocar_ros.conversions import (  # noqa: E402
    candidates_frame,
    candidates_from_frame,
    frame_meta,
    letterbox_from_message,
    tensor_frame,
    tensor_from_frame,
)

REPO = Path(__file__).resolve().parents[4]
ENGINE = REPO / 'assets' / 'models' / 'yolov8n_fp16.engine'
VIDEO = REPO / 'assets' / 'video' / '0000f77c-6257be58.mov'


@pytest.fixture(scope='module')
def frame() -> np.ndarray:
    """First frame of the benchmark clip, rotation-corrected."""
    if not ENGINE.is_file() or not VIDEO.is_file():
        pytest.skip('benchmark assets are not present')
    source = open_source(VIDEO, rotate=180, loops=1)
    try:
        return next(iter(source)).image
    finally:
        source.close()


def test_split_stage_boundary_matches_the_monolith_facade(frame: np.ndarray) -> None:
    config = YoloCarConfig(model_path=ENGINE, source=VIDEO, rotate=180)
    processor = YoloCarProcessor(config)
    try:
        processor.load()
        expected = processor.process_frame(frame)

        # Stage 2 -> wire -> stage 3 (the preprocessing node's boundary).
        tensor, letterbox = preprocess(
            frame,
            processor.input_spec.shape,
            processor.input_spec.dtype,
            pad_value=config.pad_value,
        )
        meta = frame_meta(seq=0, loop=0, t_mono=0.0, acq_ms=1.0)
        tensor_msg = tensor_frame(tensor, meta, letterbox)
        tensor_split = tensor_from_frame(tensor_msg)

        # Stage 3 (the inference node's boundary).
        outputs = infer(processor.engine, tensor_split)
        boxes, class_scores = decode_predictions(
            first_output(outputs), len(COCO_CLASS_NAMES)
        )
        candidates = select_candidates(
            boxes,
            class_scores,
            conf_threshold=config.conf_threshold,
            max_candidates=config.max_candidates,
        )
        candidates_msg = candidates_frame(
            candidates,
            tensor_msg.meta,
            letterbox_from_message(tensor_msg),
            engine_loaded_t=0.0,
        )

        # Stages 4-6 (the postprocessing node's boundary).
        restored, restored_letterbox = candidates_from_frame(candidates_msg)
        restored = nms(
            restored,
            iou_threshold=config.iou_threshold,
            agnostic=config.agnostic_nms,
            max_detections=config.max_detections,
        )
        restored = domain_filter(restored, config.target_class_ids)
        restored = coordinate_adaptation(restored, restored_letterbox)
        actual = to_detections(restored)

        assert [d.as_dict() for d in actual] == [d.as_dict() for d in expected]
    finally:
        processor.close()
