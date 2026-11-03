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
INPUT_TOPIC = 'yolocar/input'             # preprocessing -> inference
CANDIDATES_TOPIC = 'yolocar/candidates'  # inference -> postprocessing
ACK_TOPIC = 'yolocar/ack'                # postprocessing -> acquisition
STOP_TOPIC = 'yolocar/stop'              # acquisition -> postprocessing
DONE_TOPIC = 'yolocar/done'              # postprocessing -> acquisition
