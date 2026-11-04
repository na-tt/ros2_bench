# Flow control: the terminal node releases one credit, letting the acquisition
# node publish another frame. The acquisition owns a window of `inflight`
# credits and publishes while its outstanding frames are below it, so the graph
# holds at most `inflight` frames and none is ever dropped: a window of 1 is
# the old demand-driven chain, a window of k is k frames resident.
#
# The credit is identified by `seq`, not counted: the acquisition tracks one
# monotonic watermark, so a duplicated or reordered ack is harmless and only
# the latest one has to arrive. `seq` is what makes the window exact.
#
# engine_loaded_t is passed along so the acquisition node can start its
# --duration clock at engine load, exactly like the monolith, rather than when
# the first frame happens to finish.

uint64 seq
float64 engine_loaded_t
