# One acquired BGR frame plus the identity/timing metadata of stage 1.
#
# Deliberately not a bare sensor_msgs/Image: the per-frame metadata has to
# travel in the same message as the pixels, otherwise the acquisition
# timestamp and the image could be reordered relative to each other (they
# would live on two topics with independent delivery).

FrameMeta meta

uint32 height
uint32 width
uint32 channels   # 3
uint32 step       # bytes per row (width * channels)
string encoding   # "bgr8"
uint8[] data
