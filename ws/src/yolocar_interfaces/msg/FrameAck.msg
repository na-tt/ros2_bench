# Flow control: the terminal node tells the acquisition node it may acquire and
# publish the next frame. One frame is in flight at a time, which keeps the
# chain demand-driven like the single-process loop (no queue depth to hide
# latency, no silently dropped frames).
#
# engine_loaded_t is passed along so the acquisition node can start its
# --duration clock at engine load, exactly like the monolith, rather than when
# the first frame happens to finish.

uint64 seq
float64 engine_loaded_t
