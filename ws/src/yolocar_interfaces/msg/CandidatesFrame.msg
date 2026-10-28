# Post-decode detection candidates, in engine-input pixels (xyxy).
#
# This is the wire form of yolocar.postprocessing.Candidates: parallel arrays
# instead of a list of objects, so NMS/filter/coordinate-adaptation can slice
# them without rebuilding Python objects. The letterbox geometry is echoed so
# the postprocessing node can run stage 6 (coordinate adaptation) without
# having seen the preprocessing node's message.
#
# Producing this message (rather than the raw (1, 84, 8400) detect head) moves
# the engine-output decode/threshold into the inference node: a raw head is
# ~2.8 MB/frame and its serialisation would dominate the measurement.

FrameMeta meta

uint32 height       # engine input height (letterbox target)
uint32 width        # engine input width
float64 scale
int32 pad_x
int32 pad_y
uint32 original_height
uint32 original_width

float32[] boxes     # 4 * N, xyxy
float32[] scores    # N
uint16[] class_ids  # N, COCO ids

# time.monotonic() when the inference node finished loading its engine, or 0.0.
# Carried upwards so the terminal node can put it in the summary timeline.
float64 engine_loaded_t
