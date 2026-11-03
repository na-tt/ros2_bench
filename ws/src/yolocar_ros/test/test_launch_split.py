"""
The shape -> processes policy, tested as data (no launch, no ROS graph).

Every configuration in ``docs/proposals/chain-shapes.md`` is a partition of the
six atomic stages, so the interesting questions are all pure functions of the
``chain`` string: how many processes, what each one runs, which topic carries
each hop, and which one is terminal. If these answers are right, launching is
just plumbing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('launch')

LAUNCH_FILE = Path(__file__).resolve().parents[1] / 'launch' / 'bench.launch.py'


@pytest.fixture(scope='module')
def launch():
    spec = importlib.util.spec_from_file_location('bench_launch', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _names(plan):
    return [instance['name'] for instance in plan]


def _hops(plan, launch):
    image = {launch.topics.FRAME_TOPIC, launch.topics.INPUT_TOPIC}
    pixel = [
        instance['out_topic']
        for instance in plan
        if instance['out_topic'] in image
        or str(instance['out_topic']).startswith(launch.topics.FILTER_TOPIC_PREFIX)
    ]
    candidate = [
        instance['out_topic']
        for instance in plan
        if instance['out_topic'] == launch.topics.CANDIDATES_TOPIC
        or str(instance['out_topic']).startswith(launch.topics.POST_TOPIC_PREFIX)
    ]
    return pixel, candidate


# ------------------------------------------------------------------- shapes
@pytest.mark.parametrize(
    ('shape', 'expected'),
    [
        ('acq|pre|infer|post', (('acq',), ('pre',), ('infer',), ('nms', 'filter', 'coord'))),
        ('acq|pre+infer|post', (('acq',), ('pre', 'infer'), ('nms', 'filter', 'coord'))),
        ('acq|pre+infer+post', (('acq',), ('pre', 'infer', 'nms', 'filter', 'coord'))),
        (
            'acq|pre|infer|nms|filter|coord',
            tuple((stage,) for stage in ('acq', 'pre', 'infer', 'nms', 'filter', 'coord')),
        ),
    ],
)
def test_parse_shape_normalises_alias_and_keeps_order(launch, shape, expected):
    assert launch.parse_shape(shape) == expected


@pytest.mark.parametrize(
    ('shape', 'message'),
    [
        ('acq|pre|infer', 'not a partition'),                  # coord missing
        ('acq|pre|infer|post|coord', 'not a partition'),       # coord twice
        ('acq|infer|pre|post', 'not a partition'),             # transposed
        ('pre|infer|acq|post', 'not a partition'),             # acquisition moved
        ('acq||pre|infer|post', 'empty group'),
        ('acq|pre|infer|sharpen', 'not a partition'),          # typo
    ],
)
def test_parse_shape_rejects_anything_that_is_not_a_partition(launch, shape, message):
    with pytest.raises(ValueError, match=message):
        launch.parse_shape(shape)


# --------------------------------------------------------------- slicing
def test_pre_slices_defaults_to_one_node(launch):
    assert launch.pre_slices('clahe;bilateral') == ['clahe;bilateral']
    assert launch.pre_slices('') == ['']


def test_pre_split_is_one_op_per_node(launch):
    assert launch.pre_slices('a;b;c', pre_split='true') == ['a', 'b', 'c']
    # A single op has nothing to split.
    assert launch.pre_slices('a', pre_split='true') == ['a']


def test_pre_groups_is_explicit(launch):
    assert launch.pre_slices('a;b;c;d', pre_groups='3,1') == ['a;b;c', 'd']


def test_pre_groups_must_account_for_every_op(launch):
    with pytest.raises(ValueError, match='sums to 2 but pre_ops has 3'):
        launch.pre_slices('a;b;c', pre_groups='2')


# ------------------------------------------------------------------ p2c1
def test_default_shape_is_the_four_node_chain(launch):
    plan = launch.instances('acq|pre|infer|post')
    assert _names(plan) == [
        'yolocar_acquisition',
        'yolocar_preprocessing',
        'yolocar_inference',
        'yolocar_postprocessing',
    ]
    assert [instance['out_topic'] for instance in plan] == [
        launch.topics.FRAME_TOPIC,
        launch.topics.INPUT_TOPIC,
        launch.topics.CANDIDATES_TOPIC,
        None,
    ]
    assert plan[0]['executable'] == 'acquisition'
    assert {instance['executable'] for instance in plan[1:]} == {'chain'}


def test_only_the_node_running_coord_is_terminal(launch):
    for shape in (
        'acq|pre|infer|post',
        'acq|pre|infer|nms|filter+coord',
        'acq|pre+infer|post',
        'acq|pre+infer+post',
        'acq|pre|infer|nms|filter|coord',
    ):
        plan = launch.instances(shape)
        terminal = [instance for instance in plan if instance['out_topic'] is None]
        assert len(terminal) == 1, shape
        assert terminal[0]['post_stages'].endswith('coord'), shape


# ------------------------------------------------------- merged families
def test_merged_spans_carry_the_right_payload(launch):
    p1c0 = launch.instances('acq|pre+infer+post')
    assert len(p1c0) == 2
    assert p1c0[1]['stages'] == 'pre,infer,post'
    assert p1c0[1]['in_topic'] == launch.topics.FRAME_TOPIC
    assert p1c0[1]['out_topic'] is None  # it is terminal
    assert p1c0[1]['pre_ops'] == ''  # the whole chain, not a slice

    p1c1 = launch.instances('acq|pre+infer|post')
    assert [instance['stages'] for instance in p1c1[1:]] == ['pre,infer', 'post']
    assert p1c1[1]['out_topic'] == launch.topics.CANDIDATES_TOPIC
    assert p1c1[2]['in_topic'] == launch.topics.CANDIDATES_TOPIC

    p2c0 = launch.instances('acq|pre|infer+post')
    assert [instance['stages'] for instance in p2c0[1:]] == ['pre', 'infer,post']
    assert p2c0[2]['post_stages'] == 'nms,filter,coord'
    assert p2c0[1]['out_topic'] == launch.topics.INPUT_TOPIC
    assert p2c0[2]['in_topic'] == launch.topics.INPUT_TOPIC


def test_merged_pre_cannot_carry_a_slice(launch):
    with pytest.raises(ValueError, match='merged node can only run the whole chain'):
        launch.instances('acq|pre+infer|post', pre_ops='a;b', pre_groups='1,1')


# --------------------------------------------------------- split stage 4-6
def test_post_split_numbers_its_hops_after_candidates(launch):
    plan = launch.instances('acq|pre|infer|nms|filter|coord')
    assert [instance['out_topic'] for instance in plan] == [
        launch.topics.FRAME_TOPIC,
        launch.topics.INPUT_TOPIC,
        launch.topics.CANDIDATES_TOPIC,
        f'{launch.topics.POST_TOPIC_PREFIX}_0',
        f'{launch.topics.POST_TOPIC_PREFIX}_1',
        None,
    ]
    assert [instance['post_stages'] for instance in plan[1:]] == [
        '', '', 'nms', 'filter', 'coord',
    ]
    assert _names(plan)[-3:] == [
        'yolocar_postprocessing_0',
        'yolocar_postprocessing_1',
        'yolocar_postprocessing_2',
    ]


# --------------------------------------------------------- split stage 2
def test_pre_split_numbers_its_hops_between_frame_and_input(launch):
    plan = launch.instances(
        'acq|pre|pre|pre|infer|post', pre_ops='a;b;c', pre_groups='1,1,1'
    )
    assert [instance['out_topic'] for instance in plan] == [
        launch.topics.FRAME_TOPIC,
        f'{launch.topics.FILTER_TOPIC_PREFIX}_0',
        f'{launch.topics.FILTER_TOPIC_PREFIX}_1',
        launch.topics.INPUT_TOPIC,
        launch.topics.CANDIDATES_TOPIC,
        None,
    ]
    assert [instance['pre_ops'] for instance in plan[1:4]] == ['a', 'b', 'c']


def test_pre_group_count_must_match_the_shape(launch):
    # Three stage-2 nodes in the shape, two slices in the chain.
    with pytest.raises(ValueError, match='cut into 2 slice'):
        launch.instances(
            'acq|pre|pre|pre|infer|post', pre_ops='a;b', pre_groups='1,1'
        )


# ---------------------------------------------- the label is the hop count
@pytest.mark.parametrize(
    ('label', 'shape', 'pre_groups'),
    [
        ('p1c0', 'acq|pre+infer+post', ''),
        ('p2c0', 'acq|pre|infer+post', ''),
        ('p1c1', 'acq|pre+infer|post', ''),
        ('p2c1', 'acq|pre|infer|post', ''),
        ('p2c2', 'acq|pre|infer|nms|filter+coord', ''),
        ('p2c3', 'acq|pre|infer|nms|filter|coord', ''),
        ('p3c1', 'acq|pre|pre|infer|post', '4,3'),
    ],
)
def test_the_label_agrees_with_the_graph_it_names(launch, label, shape, pre_groups):
    """`ros-N{x}-p{p}c{c}`: N processes, p pixel hops, c candidate hops."""
    plan = launch.instances(
        shape, pre_ops='a;b;c;d;e;f;g', pre_groups=pre_groups
    )
    pixel, candidate = _hops(plan, launch)
    expected_p, expected_c = int(label[1]), int(label[3])
    assert (len(pixel), len(candidate)) == (expected_p, expected_c), label
    expected_n = expected_p + expected_c + 1
    assert len(plan) == expected_n, label


def test_the_twelve_process_shape(launch):
    ops = 'clahe;bilateral;gaussian;gamma;hist_equalization;median;unsharp'
    plan = launch.instances(
        'acq|pre|pre|pre|pre|pre|pre|pre|infer|nms|filter|coord',
        pre_ops=ops,
        pre_groups='1,1,1,1,1,1,1',
    )
    pixel, candidate = _hops(plan, launch)
    assert (len(plan), len(pixel), len(candidate)) == (12, 8, 3)
    assert [instance['pre_ops'] for instance in plan[1:8]] == ops.split(';')
