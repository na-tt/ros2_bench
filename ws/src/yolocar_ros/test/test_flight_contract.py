"""
The frames-in-flight window as data: one number, four places it must land.

The window is not a stage and not a shape, so it is not a pipeline change: it is
one launch argument that (a) paces the acquisition node, (b) sets the depth of
every data topic, (c) travels in every message so the record can say how many
frames were outstanding, and (d) is written into the summary's provenance.
These tests pin the three that can be checked without starting a graph; the
record itself is checked end-to-end in ``test_no_drops.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('rclpy')

from yolocar.model_input import LetterboxInfo  # noqa: E402
from yolocar.postprocessing import Candidates  # noqa: E402

from yolocar_ros import qos  # noqa: E402
from yolocar_ros.conversions import (  # noqa: E402
    candidates_frame,
    frame_meta,
    image_frame,
)

LAUNCH_FILE = Path(__file__).resolve().parents[1] / 'launch' / 'bench.launch.py'


@pytest.fixture(scope='module')
def launch():
    spec = importlib.util.spec_from_file_location('bench_launch', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------- qos
@pytest.mark.parametrize('inflight', [1, 2, 3, 4, 8])
def test_every_data_topic_is_exactly_as_deep_as_the_window(inflight: int) -> None:
    """A shorter history evicts an unacked sample; a longer one hides a mistake."""
    profile = qos.latest(inflight)
    assert profile.depth == inflight
    assert profile.history.name == 'KEEP_LAST'
    assert profile.reliability.name == 'RELIABLE'


def test_a_window_of_zero_is_clamped_to_one() -> None:
    assert qos.latest(0).depth == 1
    assert qos.control(0).depth == 1


# ------------------------------------------------------------------- launch
def test_inflight_is_a_launch_argument(launch) -> None:
    declared = {
        action.name: action
        for action in launch.generate_launch_description().entities
        if hasattr(action, 'name')
    }
    assert 'inflight' in declared, 'inflight must be a launch argument, not a constant'


def test_every_node_receives_the_window(launch) -> None:
    """One dict for every node: acquisition and the chain cannot disagree."""
    assert 'inflight' in launch._parameters()


def test_the_window_is_not_a_shape(launch) -> None:
    """Panic guard: adding frames in flight must not change the process plan."""
    plan = launch.instances('acq|pre|infer|post', pre_ops='clahe;gamma')
    assert [instance['name'] for instance in plan] == [
        'yolocar_acquisition',
        'yolocar_preprocessing',
        'yolocar_inference',
        'yolocar_postprocessing',
    ]
    assert all('inflight' not in instance for instance in plan)


# ---------------------------------------------------------------- the frame
def test_the_window_travels_with_the_frame() -> None:
    """Both hops carry FrameMeta, so the per-frame count survives either payload."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    meta = frame_meta(seq=7, loop=2, t_mono=12.5, acq_ms=1.5, inflight=4)
    meta.t_pub = 12.75
    meta.hop_ms = 0.5

    image = image_frame(frame, meta)
    assert image.meta.inflight == 4
    assert image.meta.t_pub == pytest.approx(12.75)
    assert image.meta.hop_ms == pytest.approx(0.5)

    candidates = candidates_frame(
        Candidates(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int32),
        ),
        meta,
        LetterboxInfo(
            scale=1.0,
            pad_x=0,
            pad_y=0,
            original_shape=(4, 4),
            input_shape=(4, 4),
        ),
    )
    assert candidates.meta.inflight == 4
    assert candidates.meta.hop_ms == pytest.approx(0.5)
