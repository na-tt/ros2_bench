# Engine-ready NCHW tensor plus the letterbox geometry needed to undo it.
#
# Produced by the preprocessing node with yolocar.preprocessing.preprocess().
# The payload is the raw host-order (little-endian) tensor buffer; `dtype` is a
# numpy dtype name ("float32", "float16" or "uint8") and the shape is fixed to
# (1, 3, height, width). Raw bytes are used because there is no standard ROS
# message for an NCHW float tensor, and a per-element sequence would make the
# serialisation cost depend on the field type rather than on the bytes.

FrameMeta meta

uint32 height
uint32 width
string dtype

# Letterbox geometry (yolocar.preprocessing.LetterboxInfo).
float64 scale
int32 pad_x
int32 pad_y
uint32 original_height
uint32 original_width

uint8[] data
