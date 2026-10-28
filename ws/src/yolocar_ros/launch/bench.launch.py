r"""
Launch the four-node yolocar chain.

    acquisition -> preprocessing -> inference -> postprocessing

Each stage is a separate process and a separate ROS 2 node, connected by DDS
topics. The acquisition node paces the chain with an ack from the terminal
node, so exactly one frame is in flight. When the acquisition node exits (source
exhausted, --duration reached, or a failure), the whole launch is shut down;
the postprocessing node writes ``summary.json`` before that happens.

Run it from the repository root (the default paths are relative to it), with
the pipeline package importable:

    PYTHONPATH=monolith/yolocar/src \\
      ros2 launch yolocar_ros bench.launch.py duration:=120 \\
        records:=results/ros/records.jsonl summary:=results/ros/summary.json

The benchmark harness uses exactly this entry point through
``bench/scenarios.json``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _argument(name: str, default: object, description: str) -> DeclareLaunchArgument:
    return DeclareLaunchArgument(name, default_value=str(default), description=description)


def _typed(name: str, kind: type) -> ParameterValue:
    """Wrap a launch argument as a typed ROS parameter (launch passes strings)."""
    return ParameterValue(LaunchConfiguration(name), value_type=kind)


def generate_launch_description() -> LaunchDescription:
    declared = [
        _argument('source', 'assets/video/0000f77c-6257be58.mov',
                  'video file, image, image directory or camera index'),
        _argument('rotate', 180, 'clockwise rotation applied at acquisition'),
        _argument('loops', 1, 'replay the source this many times'),
        _argument('duration', -1.0, 'measurement seconds after engine load (-1 = source length)'),
        _argument('max_frames', -1, 'stop after this many frames (-1 = no cap)'),
        _argument('model', 'assets/models/yolov8n_fp16.engine',
                  'TensorRT engine deserialized by the inference node'),
        _argument('conf', 0.25, 'candidate confidence threshold'),
        _argument('iou', 0.45, 'NMS IoU threshold'),
        _argument('imgsz', 640, 'engine input size (square)'),
        _argument('input_dtype', 'float32', 'engine input dtype: float32/float16/uint8'),
        _argument('device', 0, 'CUDA device ordinal'),
        _argument('classes', '0,1,2,3,5,6,7,9,11,12', 'comma separated COCO ids kept'),
        _argument('agnostic_nms', 'false', 'merge boxes across classes'),
        _argument('max_det', 300, 'max detections kept after NMS'),
        _argument('max_candidates', 30000, 'cap candidates before NMS'),
        _argument('pad_value', 114, 'letterbox padding value'),
        _argument('pinned', 'true', 'page-locked host buffers in TensorRT'),
        _argument('trt_log_level', 'warning', 'TensorRT logger verbosity'),
        _argument('records', 'results/ros/records.jsonl', 'per-frame JSONL output'),
        _argument('summary', 'results/ros/summary.json', 'run summary JSON output'),
        _argument('log_level', 'INFO', 'python logger level for the pipeline modules'),
    ]

    # One dict for all four nodes: each node declares only what it uses, and
    # the terminal node gets the run configuration it needs for summary.json
    # from the same place the other nodes get theirs.
    parameters = {
        'source': _typed('source', str),
        'rotate': _typed('rotate', int),
        'loops': _typed('loops', int),
        'duration': _typed('duration', float),
        'max_frames': _typed('max_frames', int),
        'model': _typed('model', str),
        'conf': _typed('conf', float),
        'iou': _typed('iou', float),
        'imgsz': _typed('imgsz', int),
        'input_dtype': _typed('input_dtype', str),
        'device': _typed('device', int),
        'classes': _typed('classes', str),
        'agnostic_nms': _typed('agnostic_nms', bool),
        'max_det': _typed('max_det', int),
        'max_candidates': _typed('max_candidates', int),
        'pad_value': _typed('pad_value', int),
        'pinned': _typed('pinned', bool),
        'trt_log_level': _typed('trt_log_level', str),
        'records': _typed('records', str),
        'summary': _typed('summary', str),
        'log_level': _typed('log_level', str),
    }

    nodes = [
        Node(
            package='yolocar_ros',
            executable='acquisition',
            name='yolocar_acquisition',
            output='screen',
            parameters=[parameters],
        ),
        Node(
            package='yolocar_ros',
            executable='preprocessing',
            name='yolocar_preprocessing',
            output='screen',
            parameters=[parameters],
        ),
        Node(
            package='yolocar_ros',
            executable='inference',
            name='yolocar_inference',
            output='screen',
            parameters=[parameters],
        ),
        Node(
            package='yolocar_ros',
            executable='postprocessing',
            name='yolocar_postprocessing',
            output='screen',
            parameters=[parameters],
        ),
    ]

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

    return LaunchDescription([*declared, *nodes, *handlers])
