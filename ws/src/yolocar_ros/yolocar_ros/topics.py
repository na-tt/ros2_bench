"""
Topic names shared by the chain nodes.

Both hops out of the preprocessing stage carry an ``ImageFrame``: conditioning
keeps source geometry, and the model contract (which the inference node owns)
is what changes the frame into a tensor.

Names are relative (no leading slash) so the whole chain can be pushed into a
namespace, e.g. ``ros2 launch ... --ros-args -r __ns:=/bench``. Every node uses
the same strings, so the graph stays connected under any namespace.
"""

FRAME_TOPIC = 'yolocar/frame'            # acquisition -> preprocessing
INPUT_TOPIC = 'yolocar/input'            # preprocessing -> inference
#: Stage-2 hops when ``pre_split`` cuts the chain: ``yolocar/filter_0``, ...
#: (a ``/`` -separated token may not start with a digit, an ``_`` suffix may).
#: The first node still reads FRAME_TOPIC and the last still writes INPUT_TOPIC,
#: so the rest of the graph cannot tell how deep the stage was split.
FILTER_TOPIC_PREFIX = 'yolocar/filter'
CANDIDATES_TOPIC = 'yolocar/candidates'  # inference -> postprocessing
ACK_TOPIC = 'yolocar/ack'                # postprocessing -> acquisition
STOP_TOPIC = 'yolocar/stop'              # acquisition -> postprocessing
DONE_TOPIC = 'yolocar/done'              # postprocessing -> acquisition
