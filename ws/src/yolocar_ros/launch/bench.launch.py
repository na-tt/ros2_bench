r"""
Launch the yolocar chain, with the preprocessing stage optionally split.

The default is four stages, one process each::

    acquisition -> preprocessing -> inference -> postprocessing

With ``pre_split:=true`` every conditioned op gets its own stage-2 node, wired
head to tail, so the same chain can be measured at any partition depth::

    acquisition -> filter(clahe) -> filter(bilateral) -> ...
                -> inference -> postprocessing

``pre_ops`` stays the single definition of *what* runs; ``pre_split`` only
decides *where the cuts are*. Nothing else in the graph changes: the inference
node still applies the model contract, and the postprocessing node is still the
terminal node that writes ``summary.json``.

Each node is a separate process connected by DDS topics. The acquisition node
paces the chain with an ack from the terminal node, so exactly one frame is in
flight. When the acquisition node exits (source exhausted, --duration reached,
or a failure), the whole launch is shut down; the postprocessing node writes
``summary.json`` before that happens.

Run it from the repository root (the default paths are relative to it), with
the pipeline package importable:

    PYTHONPATH=monolith/yolocar/src \\
      ros2 launch yolocar_ros bench.launch.py duration:=120 \\
        records:=results/ros/records.jsonl summary:=results/ros/summary.json

The benchmark harness uses exactly this entry point through
``bench/scenarios.json``.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from yolocar_ros import topics


def _argument(name: str, default: object, description: str) -> DeclareLaunchArgument:
    return DeclareLaunchArgument(name, default_value=str(default), description=description)


def _typed(name: str, kind: type) -> ParameterValue:
    """Wrap a launch argument as a typed ROS parameter (launch passes strings)."""
    return ParameterValue(LaunchConfiguration(name), value_type=kind)


def generate_launch_description() -> LaunchDescription:
    declared = [
        _argument('source', 'assets/video/0000f77c-6257be58.mov',
                  'video file, image, image directory or camera index'),
        _argument(
            'rotate',
            0,
            'clockwise rotation applied at acquisition; the shipped _720p/_360p/_180p '
            'files are already upright (0), the original clip is stored upside down (180)',
        ),
        _argument('loops', 1, 'replay the source this many times'),
        _argument('duration', -1.0, 'measurement seconds after engine load (-1 = source length)'),
        _argument('max_frames', -1, 'stop after this many frames (-1 = no cap)'),
        _argument('model', 'assets/models/yolov8n_fp16.engine',
                  'TensorRT engine deserialized by the inference node'),
        _argument('conf', 0.25, 'candidate confidence threshold'),
        _argument('iou', 0.45, 'NMS IoU threshold'),
        _argument('imgsz', 640,
                  'dynamic-shape hint for a dynamic engine; a static engine wins, '
                  'and the input dtype is always read from the engine'),
        _argument('pre_ops', '',
                  'stage-2 conditioning chain, ";" separated '
                  '(e.g. "clahe;bilateral:d=9"); empty = identity'),
        _argument('pre_split', 'false',
                  'run one stage-2 node per op instead of one node for the '
                  'whole chain (the chain itself still comes from pre_ops)'),
        _argument('device', 0, 'CUDA device ordinal'),
        _argument('classes', '0,1,2,3,5,6,7,9,11,12', 'comma separated COCO ids kept'),
        _argument('agnostic_nms', 'false', 'merge boxes across classes'),
        _argument('max_det', 300, 'max detections kept after NMS'),
        _argument('max_candidates', 30000, 'cap candidates before NMS'),
        _argument('pad_value', 114, 'letterbox padding value'),
        _argument('pinned', 'true', 'page-locked host buffers in TensorRT'),
        _argument('trt_log_level', 'warning', 'TensorRT logger verbosity'),
        _argument('records', 'results/ros/records.jsonl', 'per-frame JSONL output'),
        _argument('detections', '',
                  'optional per-frame detections JSONL (rendered offline by `yolocar.demo`)'),
        _argument('summary', 'results/ros/summary.json', 'run summary JSON output'),
        _argument('log_level', 'INFO', 'python logger level for the pipeline modules'),
    ]

    return LaunchDescription([*declared, OpaqueFunction(function=_chain_nodes)])


def _parameters() -> dict[str, object]:
    """
    One dict for every node: each declares only what it uses.

    The terminal node gets the run configuration it needs for ``summary.json``
    from the same place the other nodes get theirs, so the summary describes the
    launch and not one node's guess at it.
    """
    return {
        'source': _typed('source', str),
        'rotate': _typed('rotate', int),
        'loops': _typed('loops', int),
        'duration': _typed('duration', float),
        'max_frames': _typed('max_frames', int),
        'model': _typed('model', str),
        'conf': _typed('conf', float),
        'iou': _typed('iou', float),
        'imgsz': _typed('imgsz', int),
        'pre_ops': _typed('pre_ops', str),
        'device': _typed('device', int),
        'classes': _typed('classes', str),
        'agnostic_nms': _typed('agnostic_nms', bool),
        'max_det': _typed('max_det', int),
        'max_candidates': _typed('max_candidates', int),
        'pad_value': _typed('pad_value', int),
        'pinned': _typed('pinned', bool),
        'trt_log_level': _typed('trt_log_level', str),
        'records': _typed('records', str),
        'detections': _typed('detections', str),
        'summary': _typed('summary', str),
        'log_level': _typed('log_level', str),
    }


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def stage_two_instances(pre_ops: str, pre_split: str) -> list[dict[str, str]]:
    """
    Decide the stage-2 partition: which nodes run, in which order, on what.

    This is the whole partition policy, and it is pure so it can be tested
    without a launch context. ``pre_ops`` says *what* runs; ``pre_split`` says
    whether the chain is cut into one process per op. Either way the first node
    reads ``FRAME_TOPIC`` and the last writes ``INPUT_TOPIC``, so the rest of
    the graph cannot tell how deep the stage was split.
    """
    ops = [op for op in str(pre_ops).split(';') if op.strip()]
    if not _truthy(pre_split) or len(ops) < 2:
        return [
            {
                'name': 'yolocar_preprocessing',
                'op': ';'.join(ops),
                'in_topic': topics.FRAME_TOPIC,
                'out_topic': topics.INPUT_TOPIC,
            }
        ]
    instances: list[dict[str, str]] = []
    upstream = topics.FRAME_TOPIC
    for index, op in enumerate(ops):
        downstream = (
            topics.INPUT_TOPIC
            if index == len(ops) - 1
            else f'{topics.FILTER_TOPIC_PREFIX}_{index}'
        )
        instances.append(
            {
                'name': f'yolocar_preprocessing_{index}',
                'op': op.strip(),
                'in_topic': upstream,
                'out_topic': downstream,
            }
        )
        upstream = downstream
    return instances


def _chain_nodes(context) -> list[Node]:
    """
    Build the node list from the *resolved* arguments.

    ``pre_split`` splits the chain at generation time, which is why this cannot
    be a static description: the number of nodes depends on the value the
    operator passed, and launch only resolves it here.
    """
    nodes = [
        Node(
            package='yolocar_ros',
            executable='acquisition',
            name='yolocar_acquisition',
            output='screen',
            parameters=[_parameters()],
        )
    ]

    for instance in stage_two_instances(
        LaunchConfiguration('pre_ops').perform(context),
        str(LaunchConfiguration('pre_split').perform(context)),
    ):
        nodes.append(
            Node(
                package='yolocar_ros',
                executable='preprocessing',
                name=instance['name'],
                output='screen',
                parameters=[
                    {
                        **_parameters(),
                        'pre_ops': instance['op'],
                        'in_topic': instance['in_topic'],
                        'out_topic': instance['out_topic'],
                    }
                ],
            )
        )

    nodes.extend(
        [
            Node(
                package='yolocar_ros',
                executable='inference',
                name='yolocar_inference',
                output='screen',
                parameters=[_parameters()],
            ),
            Node(
                package='yolocar_ros',
                executable='postprocessing',
                name='yolocar_postprocessing',
                output='screen',
                parameters=[_parameters()],
            ),
        ]
    )

    # Any node exiting ends the run: the acquisition node is the normal case
    # (it exits after the terminal node acknowledged the summary), a crash is
    # the abnormal one and should not leave a trial hanging.
    handlers = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=node,
                on_exit=[
                    EmitEvent(
                        event=Shutdown(
                            reason=f'{node.node_executable} exited; shutting down the chain'
                        )
                    )
                ],
            )
        )
        for node in nodes
    ]
    return [*nodes, *handlers]
