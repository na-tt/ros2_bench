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
float64 pre_ms    # stage 2: letterbox + BGR->RGB NCHW tensor
float64 infer_ms  # stage 3: engine invocation + detect-head decode/threshold
