r"""
Launch the yolocar chain in any shape.

A shape is a partition of the pipeline's six stages into processes::

    acq | pre | infer | nms | filter | coord        # the atomic stages
    acq | pre | infer | post                        # = the last three merged

``|`` separates processes, ``+`` merges stages inside one (``post`` is sugar for
``nms+filter+coord``). The ``chain`` argument *is* the shape:

    chain:=acq|pre|infer|post              p2c1  four processes (default)
    chain:=acq|pre|infer|nms|filter|coord  p2c3  six
    chain:=acq|pre+infer|post              p1c1  pixels cross one boundary
    chain:=acq|pre,infer,post              p1c0  the monolith in one process

Every process is the same executable (``chain``) with a different ``stages``
parameter; only acquisition is its own binary, because it owns the source and
paces the graph. The stage-2 chain and stage 4-6 can be cut further inside
their span with ``pre_groups`` and by naming the post stages separately::

    pre_ops:=clahe;bilateral;gamma pre_groups:=2,1
    chain:=acq|pre|pre|infer|post          # three stage-2 nodes, ops 2+1

The terminal node -- the one that runs ``coord`` -- is derived from the shape,
not configured, and it is the only node that writes ``records.jsonl`` /
``summary.json`` and acknowledges frames. Because the ack paces acquisition,
exactly one frame is in flight and the graph is demand-driven like the
monolith's loop, at any depth.

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

#: The atomic stages, in canonical order. A shape is a contiguous partition of
#: exactly this sequence: nothing missing, nothing repeated, nothing reordered.
STAGE_ORDER: tuple[str, ...] = ('acq', 'pre', 'infer', 'nms', 'filter', 'coord')

#: Sugar: ``post`` means the whole of stage 4-6.
STAGE_ALIASES: dict[str, tuple[str, ...]] = {'post': ('nms', 'filter', 'coord')}

#: Stage 4-6 in canonical order.
POST_STAGE_ORDER: tuple[str, ...] = ('nms', 'filter', 'coord')

#: A group that is exactly this runs the conditioning chain alone, so it may
#: carry one *slice* of it instead of the whole thing.
PRE_ONLY = ('pre',)


#: Which *node* family a stage belongs to. The node's ``stages`` parameter
#: speaks families (pre/infer/post); the ``post_stages`` parameter says which
#: of stage 4-6 it runs. Keeping the two vocabularies apart means a node never
#: has to know that "nms" is a stage rather than a family.
FAMILY_OF: dict[str, str] = {
    'pre': 'pre',
    'infer': 'infer',
    'nms': 'post',
    'filter': 'post',
    'coord': 'post',
}


def stage_span(stages: tuple[str, ...]) -> str:
    """``('infer','nms','filter')`` -> ``'infer,post'`` (families, in order)."""
    families = list(dict.fromkeys(FAMILY_OF[stage] for stage in stages))
    return ','.join(families)


def _family(group: tuple[str, ...]) -> str:
    """Which family a group belongs to: ``pre``, ``infer``, ``post`` or ``mixed``."""
    if group == PRE_ONLY:
        return 'pre'
    if group == ('infer',):
        return 'infer'
    if all(stage in POST_STAGE_ORDER for stage in group):
        return 'post'
    return 'mixed'


def _argument(name: str, default: object, description: str) -> DeclareLaunchArgument:
    return DeclareLaunchArgument(name, default_value=str(default), description=description)


def _typed(name: str, kind: type) -> ParameterValue:
    """Wrap a launch argument as a typed ROS parameter (launch passes strings)."""
    return ParameterValue(LaunchConfiguration(name), value_type=kind)


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def parse_shape(shape: str) -> tuple[tuple[str, ...], ...]:
    """
    Turn ``"acq|pre+infer|post"`` into groups of atomic stages.

    Validating that the flattened groups equal :data:`STAGE_ORDER` is the whole
    check: it catches a missing stage, a duplicate, a transposed pair and a typo
    in one comparison, and it makes "a shape is a partition" a fact rather than
    a convention.
    """
    groups: list[tuple[str, ...]] = []
    for chunk in str(shape).split('|'):
        stages: list[str] = []
        for token in chunk.split('+'):
            name = token.strip()
            if not name:
                continue
            stages.extend(STAGE_ALIASES.get(name, (name,)))
        if not stages:
            raise ValueError(f'empty group in shape {shape!r}')
        groups.append(tuple(stages))
    flat = tuple(stage for group in groups for stage in group)
    # `pre` may repeat -- one node per slice of the conditioning chain -- but
    # only consecutively, because the slices run in order. Collapsing adjacent
    # duplicates turns the check into "is this a partition of the six stages"
    # and keeps a transposed pair or a duplicate `coord` rejected.
    collapsed: list[str] = []
    for stage in flat:
        if stage == 'pre' and collapsed and collapsed[-1] == 'pre':
            continue
        collapsed.append(stage)
    if tuple(collapsed) != STAGE_ORDER:
        raise ValueError(
            f'shape {shape!r} is {"/".join("+".join(g) for g in groups)}, which is '
            f'not a partition of {"|".join(STAGE_ORDER)} (got {", ".join(flat)})'
        )
    return tuple(groups)


def pre_slices(
    pre_ops: str, pre_groups: str = '', pre_split: str = 'false'
) -> list[str]:
    """
    Split the stage-2 chain into the ``;``-joined slices its nodes run.

    ``pre_groups`` is the explicit form (``"2,5"`` = two ops then five);
    ``pre_split`` is the shorthand for one op per node. An empty chain is one
    node running nothing, which is the identity and therefore the baseline.
    """
    ops = [op.strip() for op in str(pre_ops).split(';') if op.strip()]
    if str(pre_groups).strip():
        sizes = [
            int(part) for part in str(pre_groups).replace(';', ',').split(',')
            if part.strip()
        ]
        if any(size < 0 for size in sizes):
            raise ValueError(f'pre_groups must be non-negative, got {pre_groups!r}')
        if sum(sizes) != len(ops):
            raise ValueError(
                f'pre_groups {pre_groups!r} sums to {sum(sizes)} but '
                f'pre_ops has {len(ops)} op(s)'
            )
    elif _truthy(pre_split) and len(ops) > 1:
        sizes = [1] * len(ops)
    else:
        sizes = [len(ops)]

    slices: list[str] = []
    index = 0
    for size in sizes:
        slices.append(';'.join(ops[index:index + size]))
        index += size
    return slices


def instances(
    shape: str, *, pre_ops: str = '', pre_groups: str = '', pre_split: str = 'false'
) -> list[dict[str, object]]:
    """
    Decide which processes run, on what, wired how.

    This is the whole partition policy, and it is pure, so it is tested without
    a launch context and without ROS. Each entry is one process:

    ``name``/``executable`` -- what to run;
    ``in_topic``/``out_topic`` -- the hop it consumes and the hop it produces
    (``None`` for the terminal node, which produces records);
    ``stages``/``post_stages``/``pre_ops`` -- the span it runs.

    Hop names follow the canonical edges where they exist (``yolocar/frame``,
    ``yolocar/input``, ``yolocar/candidates``) and are numbered otherwise
    (``yolocar/filter_<i>``, ``yolocar/post_<i>``), so a deep shape never has
    two producers on one topic.
    """
    groups = parse_shape(shape)
    slices = pre_slices(pre_ops, pre_groups, pre_split)
    pre_nodes = sum(1 for group in groups if group == PRE_ONLY)
    merged_pre = sum(1 for group in groups if 'pre' in group and group != PRE_ONLY)

    if pre_nodes + merged_pre == 0:
        raise ValueError(f'shape {shape!r} has no stage 2')
    if merged_pre and len(slices) > 1:
        raise ValueError(
            f'shape {shape!r} merges stage 2 with another stage but the chain is '
            'cut into slices; a merged node can only run the whole chain'
        )
    if pre_nodes + merged_pre != len(slices):
        raise ValueError(
            f'shape {shape!r} has {pre_nodes} stage-2 node(s) but the chain is '
            f'cut into {len(slices)} slice(s); use pre_groups:=... to match'
        )

    post_nodes = sum(1 for group in groups if _family(group) == 'post')
    plan: list[dict[str, object]] = []
    filter_hops = 0
    post_hops = 0
    candidate_hops = 0
    pre_index = 0
    post_index = 0

    for position, group in enumerate(groups):
        if position == 0:
            if group != ('acq',):
                raise ValueError(
                    f'shape {shape!r} must start with acquisition as its own group'
                )
            plan.append(
                {
                    'name': 'yolocar_acquisition',
                    'executable': 'acquisition',
                    'in_topic': None,
                    'out_topic': topics.FRAME_TOPIC,
                    'stages': '',
                    'post_stages': '',
                    'pre_ops': str(pre_ops),
                }
            )
            continue

        stages = group
        previous = groups[position - 1]
        family = _family(group)

        # What this node consumes: whatever the previous node produced. The
        # canonical topics are used where they exist (frame, input, candidates)
        # and numbered otherwise, so a deep shape never puts two producers on
        # one topic.
        first = stages[0]
        if first == 'pre':
            in_topic = (
                topics.FRAME_TOPIC
                if previous == ('acq',)
                else f'{topics.FILTER_TOPIC_PREFIX}_{filter_hops}'
            )
            if previous != ('acq',):
                filter_hops += 1
        elif first == 'infer':
            in_topic = topics.INPUT_TOPIC
        else:
            in_topic = (
                topics.CANDIDATES_TOPIC
                if post_hops == 0
                else f'{topics.POST_TOPIC_PREFIX}_{post_hops - 1}'
            )
            post_hops += 1

        # What it produces: a frame, candidates, or records. Only the node
        # that ends on `coord` is terminal; `nms` and `filter` nodes pass
        # candidates on, numbered like the consumers expect them.
        last = stages[-1]
        if last == 'coord':
            out_topic: str | None = None  # terminal: records, summary, ack
        elif last == 'pre':
            # Same numbering the consumers use: the last stage-2 node hands the
            # frame to inference, the ones before it hand it to the next node.
            out_topic = (
                topics.INPUT_TOPIC
                if pre_index == pre_nodes - 1
                else f'{topics.FILTER_TOPIC_PREFIX}_{pre_index}'
            )
        else:
            out_topic = (
                topics.CANDIDATES_TOPIC
                if candidate_hops == 0
                else f'{topics.POST_TOPIC_PREFIX}_{candidate_hops - 1}'
            )
            candidate_hops += 1

        # The historic node names are kept for single-family nodes so log lines
        # stay recognisable; anything merged gets a positional name, which is
        # also the only honest one (it is not any single stage).
        if family == 'pre':
            suffix = '' if pre_nodes == 1 else f'_{pre_index}'
            name = f'yolocar_preprocessing{suffix}'
            op_slice = slices[pre_index]
            pre_index += 1
        elif family == 'infer':
            name, op_slice = 'yolocar_inference', ''
        elif family == 'post':
            suffix = '' if post_nodes == 1 else f'_{post_index}'
            name = f'yolocar_postprocessing{suffix}'
            op_slice = ''
            post_index += 1
        else:
            name, op_slice = f'yolocar_chain_{position}', ''

        plan.append(
            {
                'name': name,
                'executable': 'chain',
                'in_topic': in_topic,
                'out_topic': out_topic,
                'stages': stage_span(group),
                'post_stages': ','.join(
                    stage for stage in stages if stage in POST_STAGE_ORDER
                ),
                'pre_ops': str(pre_ops) if 'pre' in stages and not op_slice else op_slice,
            }
        )
    return plan


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
        'chain': _typed('chain', str),
        #: Provenance, not behaviour: the node writes it into the summary and
        #: the detections sidecar so an artefact says which graph made it.
        'shape': _typed('chain', str),
    }


def _chain_nodes(context) -> list[Node]:
    """
    Build the node list from the *resolved* arguments.

    The shape is resolved here because the number of nodes depends on the value
    the operator passed, and launch only knows it at generation time.
    """
    plan = instances(
        LaunchConfiguration('chain').perform(context),
        pre_ops=LaunchConfiguration('pre_ops').perform(context),
        pre_groups=LaunchConfiguration('pre_groups').perform(context),
        pre_split=LaunchConfiguration('pre_split').perform(context),
    )
    shared = _parameters()
    nodes = [
        Node(
            package='yolocar_ros',
            executable=instance['executable'],
            name=str(instance['name']),
            output='screen',
            parameters=[
                {
                    **shared,
                    'stages': instance['stages'],
                    'post_stages': instance['post_stages'],
                    'pre_ops': instance['pre_ops'],
                    'in_topic': instance['in_topic'] or '',
                    'out_topic': instance['out_topic'] or '',
                }
            ],
        )
        for instance in plan
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
    return [*nodes, *handlers]


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
        _argument('chain', 'acq|pre|infer|post',
                  'the shape: "|" separates processes, "+" merges stages; stages are '
                  'acq, pre, infer, nms, filter, coord, and "post" means the last three'),
        _argument('pre_ops', '',
                  'stage-2 conditioning chain, ";" separated '
                  '(e.g. "clahe;bilateral:d=9"); empty = identity'),
        _argument('pre_groups', '',
                  'how to cut pre_ops across the stage-2 nodes named in `chain`, '
                  'e.g. "2,5" for two ops then five; empty = one node'),
        _argument('pre_split', 'false',
                  'shorthand for one stage-2 node per op (same as pre_groups:=1,1,...); '
                  'the chain itself still comes from pre_ops'),
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
