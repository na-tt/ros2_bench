"""
Hardware test: every shape must find exactly the same detections.

A shape is a partition of the same stages, so cutting it differently may change
the *timings* and must change nothing else. This walks each shape's process
list from the launch policy, runs the stages the way the node does, and
serialises through the message types at every process boundary -- which is the
only thing a cut actually does to the data.

It does not start processes: it composes the same ``yolocar`` functions the node
composes, with the same wire round trips. That makes it a test of the *split*
(payload types, stage order, no double application, nothing dropped), not of the
transport. Skipped when the engine, the clip or TensorRT are unavailable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('tensorrt')

from yolocar.acquisition import open_source  # noqa: E402
from yolocar.config import COCO_CLASS_NAMES, YoloCarConfig  # noqa: E402
from yolocar.inference import decode_predictions, first_output, infer  # noqa: E402
from yolocar.model_input import prepare  # noqa: E402
from yolocar.postprocessing import (  # noqa: E402
    coordinate_adaptation,
    domain_filter,
    nms,
    select_candidates,
    to_detections,
)
from yolocar.preprocessing import as_bgr_frame, condition, parse  # noqa: E402
from yolocar.processor import YoloCarProcessor  # noqa: E402

from yolocar_ros.conversions import (  # noqa: E402
    candidates_frame,
    candidates_from_frame,
    frame_from_image,
    frame_meta,
    image_frame,
)

REPO = Path(__file__).resolve().parents[4]
CLIP = REPO / 'assets/video/0000f77c-6257be58_180p.mov'
ENGINE = REPO / 'assets/models/yolov8n_fp16.engine'
LAUNCH_FILE = Path(__file__).resolve().parents[1] / 'launch' / 'bench.launch.py'

#: The whole conditioning chain: the shape axis is measured with all of it on.
CHAIN = ';'.join(
    ('clahe', 'bilateral', 'gaussian', 'gamma', 'hist_equalization', 'median',
     'unsharp')
)

#: Every shape the launch file builds for this project, by label. Spelled out
#: rather than derived from the scenario file, so a shape cannot quietly drop
#: out of the invariant by being removed from the design.
SHAPES = [
    ('p1c0', 'acq|pre+infer+post', {}),
    ('p1c1', 'acq|pre+infer|post', {}),
    ('p2c0', 'acq|pre|infer+post', {}),
    ('p2c1', 'acq|pre|infer|post', {}),
    ('p2c2', 'acq|pre|infer|nms|filter+coord', {}),
    ('p2c3', 'acq|pre|infer|nms|filter|coord', {}),
    ('p3c1', 'acq|pre|pre|infer|post', {'pre_groups': '2,5'}),
    (
        'p8c3',
        'acq|pre|pre|pre|pre|pre|pre|pre|infer|nms|filter|coord',
        {'pre_groups': '1,1,1,1,1,1,1'},
    ),
]


@pytest.fixture(scope='module')
def launch():
    spec = importlib.util.spec_from_file_location('bench_launch', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def rig():
    if not ENGINE.exists() or not CLIP.exists():
        pytest.skip('engine or clip missing')
    config = YoloCarConfig(model_path=ENGINE, source=str(CLIP), rotate=0)
    processor = YoloCarProcessor(config)
    try:
        processor.load()
    except Exception as error:  # noqa: BLE001 - any driver failure means skip
        pytest.skip(f'TensorRT unavailable: {error}')
    source = open_source(str(CLIP), rotate=0, loops=1)
    try:
        frame = next(iter(source)).image
    finally:
        source.close()
    yield processor, frame
    processor.close()


def _reference(processor, frame):
    """
    Compute the monolith answer: every stage once, no message boundary.

    Includes the conditioning chain, because that is work the shape also does --
    a shape diff must not be a diff in what was computed, only in where it ran.
    """
    frame, _ = condition(as_bgr_frame(frame), parse(CHAIN))
    spec = processor.input_spec
    tensor, letterbox = prepare(frame, spec.shape, spec.dtype)
    outputs = infer(processor.engine, tensor)
    boxes, scores = decode_predictions(first_output(outputs), len(COCO_CLASS_NAMES))
    candidates = select_candidates(
        boxes,
        scores,
        conf_threshold=processor.config.conf_threshold,
        max_candidates=processor.config.max_candidates,
    )
    candidates = nms(candidates, iou_threshold=processor.config.iou_threshold)
    candidates = domain_filter(candidates, processor.config.target_class_ids)
    return to_detections(coordinate_adaptation(candidates, letterbox))


def _post(candidates, letterbox, post_stages, processor):
    for name in [part for part in post_stages.split(',') if part]:
        if name == 'nms':
            candidates = nms(
                candidates, iou_threshold=processor.config.iou_threshold
            )
        elif name == 'filter':
            candidates = domain_filter(
                candidates, processor.config.target_class_ids
            )
        elif name == 'coord':
            candidates = coordinate_adaptation(candidates, letterbox)
    return candidates


def _run_group(processor, group, payload):
    """Run one process's span, taking and returning what crosses a topic."""
    stages = [part for part in str(group['stages']).split(',') if part]
    post_stages = str(group['post_stages'])

    if stages[0] == 'post':
        candidates, letterbox = candidates_from_frame(payload)
        candidates = _post(candidates, letterbox, post_stages, processor)
        if group['out_topic'] is None:
            return to_detections(candidates)
        return candidates_frame(candidates, payload.meta, letterbox)

    frame = frame_from_image(payload)
    if 'pre' in stages:
        frame, _ = condition(as_bgr_frame(frame), parse(str(group['pre_ops'])))
        if 'infer' not in stages:
            return image_frame(frame, payload.meta)

    spec = processor.input_spec
    tensor, letterbox = prepare(frame, spec.shape, spec.dtype)
    outputs = infer(processor.engine, tensor)
    boxes, scores = decode_predictions(first_output(outputs), len(COCO_CLASS_NAMES))
    candidates = select_candidates(
        boxes,
        scores,
        conf_threshold=processor.config.conf_threshold,
        max_candidates=processor.config.max_candidates,
    )

    if 'post' not in stages or not post_stages:
        return candidates_frame(candidates, payload.meta, letterbox)
    candidates = _post(candidates, letterbox, post_stages, processor)
    if group['out_topic'] is None:
        return to_detections(candidates)
    return candidates_frame(candidates, payload.meta, letterbox)


def _key(detections):
    """Reduce detections to a comparable form: geometry, score and class."""
    return [
        (
            round(float(d.x_min), 2),
            round(float(d.y_min), 2),
            round(float(d.x_max), 2),
            round(float(d.y_max), 2),
            round(float(d.confidence), 4),
            int(d.class_id),
        )
        for d in detections
    ]


def test_the_reference_frame_is_not_empty(rig):
    """A comparison over zero detections would pass for the wrong reason."""
    processor, frame = rig
    assert _reference(processor, frame), 'the reference frame produced no detections'


@pytest.mark.parametrize(('label', 'shape', 'knobs'), SHAPES)
def test_every_shape_finds_the_same_detections(rig, launch, label, shape, knobs):
    processor, frame = rig
    plan = launch.instances(shape, pre_ops=CHAIN, **knobs)
    payload = image_frame(frame, frame_meta(seq=0, loop=0, t_mono=0.0))
    for group in plan[1:]:  # plan[0] is acquisition; it produced `payload`
        payload = _run_group(processor, group, payload)
    assert _key(payload) == _key(_reference(processor, frame)), label


def test_one_stage_split_and_merged_agree(rig, launch):
    """A stage-2 cut changes where the ops run, not what they compute."""
    processor, frame = rig
    meta = frame_meta(seq=0, loop=0, t_mono=0.0)

    split = image_frame(frame, meta)
    for group in launch.instances(
        'acq|pre|pre|pre|infer|post', pre_ops=CHAIN, pre_groups='1,1,5'
    )[1:]:
        split = _run_group(processor, group, split)

    whole = _run_group(
        processor,
        {
            'stages': 'pre,infer,post',
            'post_stages': 'nms,filter,coord',
            'pre_ops': CHAIN,
            'out_topic': None,
        },
        image_frame(frame, frame_meta(seq=0, loop=0, t_mono=0.0)),
    )
    assert _key(split) == _key(whole)
    assert split


def test_the_same_shape_twice_is_stable(rig, launch):
    """Guards the test itself: a nondeterministic pipeline would pass above."""
    processor, frame = rig
    results = []
    for _ in range(2):
        payload = image_frame(frame, frame_meta(seq=0, loop=0, t_mono=0.0))
        for group in launch.instances(
            'acq|pre|pre|infer|post', pre_ops='clahe;gamma', pre_groups='1,1'
        )[1:]:
            payload = _run_group(processor, group, payload)
        results.append(_key(payload))
    assert results[0] == results[1]
