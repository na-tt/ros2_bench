"""
Topic names shared by the four nodes.

Names are relative (no leading slash) so the whole chain can be pushed into a
namespace, e.g. ``ros2 launch ... --ros-args -r __ns:=/bench``. Every node uses
the same strings, so the graph stays connected under any namespace.
"""

FRAME_TOPIC = 'yolocar/frame'            # acquisition -> preprocessing
TENSOR_TOPIC = 'yolocar/tensor'          # preprocessing -> inference
CANDIDATES_TOPIC = 'yolocar/candidates'  # inference -> postprocessing
ACK_TOPIC = 'yolocar/ack'                # postprocessing -> acquisition
STOP_TOPIC = 'yolocar/stop'              # acquisition -> postprocessing
DONE_TOPIC = 'yolocar/done'              # postprocessing -> acquisition
