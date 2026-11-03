"""
The launch file's chain composition is data, so test it as data.

``pre_split`` decides how many stage-2 processes run and how they are wired,
and it is resolved at generation time -- a typo in it would produce a silently
different graph rather than an error. The partition policy is a pure function
(``stage_two_instances``) precisely so it can be pinned down here without
starting a single ROS process.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('launch')

LAUNCH_FILE = Path(__file__).resolve().parents[1] / 'launch' / 'bench.launch.py'


@pytest.fixture(scope='module')
def launch_module():
    spec = importlib.util.spec_from_file_location('bench_launch', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_launch_description_builds(launch_module) -> None:
    """Catches import/typo regressions in the file itself."""
    assert launch_module.generate_launch_description() is not None


def test_default_chain_is_one_stage_two_node(launch_module) -> None:
    instances = launch_module.stage_two_instances('', 'false')
    assert instances == [
        {
            'name': 'yolocar_preprocessing',
            'op': '',
            'in_topic': 'yolocar/frame',
            'out_topic': 'yolocar/input',
        }
    ]


def test_unsplit_chain_stays_one_process(launch_module) -> None:
    """A chain without pre_split is one process however many ops it runs."""
    instances = launch_module.stage_two_instances('clahe;bilateral:d=9;gamma', 'false')
    assert len(instances) == 1
    assert instances[0]['op'] == 'clahe;bilateral:d=9;gamma'
    assert instances[0]['in_topic'] == 'yolocar/frame'
    assert instances[0]['out_topic'] == 'yolocar/input'


@pytest.mark.parametrize('flag', ['true', 'True', '1', 'yes', 'on'])
def test_truthy_spellings_all_split(launch_module, flag: str) -> None:
    assert len(launch_module.stage_two_instances('clahe;gamma', flag)) == 2


def test_split_chain_is_one_process_per_op_wired_head_to_tail(launch_module) -> None:
    instances = launch_module.stage_two_instances('clahe;bilateral:d=9;gamma', 'true')
    assert [instance['name'] for instance in instances] == [
        'yolocar_preprocessing_0',
        'yolocar_preprocessing_1',
        'yolocar_preprocessing_2',
    ]
    # Order preserved, ends on the shared topics: the acquisition and inference
    # nodes cannot tell the stage was split.
    assert [
        (instance['in_topic'], instance['out_topic']) for instance in instances
    ] == [
        ('yolocar/frame', 'yolocar/filter_0'),
        ('yolocar/filter_0', 'yolocar/filter_1'),
        ('yolocar/filter_1', 'yolocar/input'),
    ]
    assert [instance['op'] for instance in instances] == [
        'clahe',
        'bilateral:d=9',
        'gamma',
    ]


def test_split_with_a_single_op_falls_back_to_one_process(launch_module) -> None:
    """Splitting one op into one process is the unsplit configuration."""
    instances = launch_module.stage_two_instances('clahe', 'true')
    assert [instance['name'] for instance in instances] == ['yolocar_preprocessing']
    assert instances[0]['op'] == 'clahe'


def test_stage_two_names_and_topics_are_unique_per_split(launch_module) -> None:
    instances = launch_module.stage_two_instances(
        'gamma;clahe;bilateral;median;gaussian', 'true'
    )
    names = [instance['name'] for instance in instances]
    topics = [instance['out_topic'] for instance in instances]
    assert len(set(names)) == len(names)
    assert len(set(topics)) == len(topics)
