# Per-frame identity and accumulated stage timings.
#
# The acquisition node fills in seq/loop/t_mono/acq_ms; each downstream stage
# overwrites the field it owns (pre_ms, infer_ms) and forwards the message
# unchanged otherwise. That is what lets the terminal node report end-to-end
# latency across processes instead of timing around its own callback.

uint64 seq        # monotonic frame counter, across replays
uint32 loop       # 0-based replay index of the source
float64 t_mono    # time.monotonic() at capture, seconds (CLOCK_MONOTONIC)
float64 acq_ms    # stage 1: decode + rotation correction
float64 pre_ms    # stage 2: conditioning chain at source resolution
float64 infer_ms  # stage 3: model contract + engine + decode/threshold
float64 post_ms   # stages 4-6: NMS + domain filter + coordinate adaptation

# Per-stage breakdowns as parallel arrays in canonical stage order (empty when
# the stage ran nothing). The terminal node rebuilds the optional `pre_ops` /
# `post_ops` maps of the records contract from them; carried in the message so
# a stage split across several nodes still reports one breakdown instead of one
# node's share. `post_ms`/`pre_ms` are sums over the nodes that contributed.
string[] pre_op_names
float64[] pre_op_ms
string[] post_op_names
float64[] post_op_ms
