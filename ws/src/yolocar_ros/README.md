# yolocar_ros — ROS 2 chain, with a splittable preprocessing stage

ROS 2 wrapper for the [`yolocar`](../../monolith/yolocar) pipeline. It contains
no computer-vision logic: every stage calls the same function the monolith
calls, and this package only adds nodes, topics and the benchmark's file
contract.

```
acquisition ──frame──> preprocessing ──frame──> inference ──candidates──> postprocessing
     ^                                                                          │
     └────────────────── ack (one credit; inflight:=D frames) ──────────────────┘

with pre_split:=true, the middle stage may be cut into one node per op:

acquisition ──frame──> filter(clahe) ──frame──> filter(bilateral) ──frame──> inference ...
```

| node | stage function | input → output |
| --- | --- | --- |
| `acquisition` | `yolocar.acquisition.open_source` | source → `ImageFrame` (BGR + metadata) |
| `preprocessing` | `yolocar.preprocessing.condition` | `ImageFrame` → `ImageFrame` (same geometry, conditioned) |
| `inference` | `yolocar.model_input.prepare` then `yolocar.inference.infer` / `decode_predictions` / `select_candidates` | `ImageFrame` → `CandidatesFrame` |
| `postprocessing` | `yolocar.postprocessing.nms` → `domain_filter` → `coordinate_adaptation` | `CandidatesFrame` → `records.jsonl` + `summary.json` |

Only two message types carry pixels, and both are the *same* type: every hop out
of stage 2 carries a BGR `ImageFrame` at source resolution. The model input
contract (letterbox, RGB, NCHW, dtype) never crosses a topic -- it is applied by
the node that owns the engine -- so there is no payload of fixed size on the
wire and the transport cost scales with the source resolution.

Messages live in `../yolocar_interfaces`; the `FrameMeta` sub-message carries
`seq`, `loop`, the capture timestamp (`time.monotonic()`) and the timings each
stage contributes, so latency is measured **across** processes instead of
around a callback.

## Why the ack, and what `inflight` does with it

The ack is a **credit**. The acquisition node holds a window of `inflight`
frames (default 1): it publishes while fewer than that many of its frames are
unacknowledged, and each ack from the terminal node releases one. Without any
pacing a fast publisher would either fill queues with work the rest of the chain
has not reached (adding queueing latency that is not the pipeline's) or drop
frames on a shallow subscription (silently shrinking the sample).

* `inflight:=1` is the demand-driven chain: exactly like the monolith's blocking
  loop, one frame resident, nothing queued, `p50` is the pipeline's residency.
  Every cell measured before this parameter existed ran at this setting.
* `inflight:=k` keeps `k` frames resident. Throughput rises until the slowest
  node saturates and latency grows by that node's service time per extra frame
  after that; the curve and its model are in
  [`docs/proposals/frames-in-flight.md`](../../docs/proposals/frames-in-flight.md).

The credit is identified by `seq`, not counted: the acquisition keeps one
monotonic watermark, so a duplicated or reordered ack changes nothing. The ack
is only sent after the frame's record has been flushed to disk. The same number
is the history depth of every data topic (`depth < inflight` would evict an
unacked sample and never release its credit), and the acquisition drains the
window before the terminal node writes `summary.json`, so a stop never truncates
the records.

Frame 0 is always published synchronously: its ack is also the discovery gate
(every downstream node has matched) and the engine-ready gate (it carries
`engine_loaded_t`, which starts the `duration` clock).

## Stage-boundary caveat

The inference node publishes **post-decode** candidates, not the raw
`(1, 84, 8400)` detect head: the raw head is ~2.8 MB per frame and serialising
it would dominate the measurement. `decode_predictions()` and
`select_candidates()` therefore run inside the inference node, so its
`infer_ms` includes the decode/threshold step.

When comparing with the monolith, compare **`infer + post` as one block**.
`pre_ms`, `acq_ms` and the detection output are directly comparable.

The ROS front-end additionally reports `transport_ms` per record (and a
`transport` bucket in `summary.json`): capture-to-detections minus the three
stage timings, i.e. serialisation, DDS transport and scheduling. The monolith
has no such cost, which is why the two front-ends' `latency_ms` are not expected
to be equal.

## Build and run

The workspace must be built once, and the pipeline project must be importable.
There is no `pip` on the Jetson's `/usr/bin/python3`, so use `PYTHONPATH`:

```bash
cd ~/Memoria/ros2_bench/ws
colcon build --symlink-install
source install/setup.bash

cd ~/Memoria/ros2_bench            # defaults are relative to the repo root
PYTHONPATH=monolith/yolocar/src:$PYTHONPATH \
  ros2 launch yolocar_ros bench.launch.py duration:=120 \
    records:=results/ros/records.jsonl \
    summary:=results/ros/summary.json
```

`duration:=-1` (the launch default) runs the source to its end. The clip in
`assets/video/` is stored upside down, hence `rotate:=180` is the default.
There is no `input_dtype` launch argument: the inference node reads the input
spec off the engine it owns, so a dtype that disagrees with the engine is not
expressible. `imgsz` survives only as a dynamic-shape hint for a dynamic engine
(a static engine's shape wins).

### Conditioning chain and partition depth

`pre_ops` is the single definition of what stage 2 runs (`;` separated);
`pre_split` decides where the cuts are:

```bash
# the whole chain in one node (the default shape of the graph)
ros2 launch yolocar_ros bench.launch.py "pre_ops:=clahe;bilateral:d=9"

# one process per op, wired head to tail
ros2 launch yolocar_ros bench.launch.py "pre_ops:=clahe;bilateral:d=9" pre_split:=true
```

`pre_ms` and the per-op arrays accumulate along the chain, so the terminal node
writes the same records either way and the extra hops show up in `transport_ms`.
That is what makes partition depth a measurable axis rather than a different
experiment.

The benchmark harness uses exactly this entry point through
`bench/scenarios.json` (`ros-N4-p2c1`). A smoke comparison:

```bash
bench/run.py --conditions monolith,ros-N4-p2c1 --repeats 1 --duration 30
bench/analyze.py --from 10 --to 30
```

## Nodes, topics, parameters

| topic | type | direction |
| --- | --- | --- |
| `yolocar/frame` | `yolocar_interfaces/ImageFrame` | acquisition → preprocessing |
| `yolocar/input` | `yolocar_interfaces/ImageFrame` | preprocessing → inference |
| `yolocar/filter_<i>` | `yolocar_interfaces/ImageFrame` | between stage-2 nodes, only with `pre_split:=true` |
| `yolocar/candidates` | `yolocar_interfaces/CandidatesFrame` | inference → postprocessing |
| `yolocar/ack` | `yolocar_interfaces/FrameAck` | postprocessing → acquisition |
| `yolocar/stop` | `std_msgs/Empty` | acquisition → postprocessing |
| `yolocar/done` | `std_msgs/Empty` | postprocessing → acquisition |

All data topics use reliable, keep-last-`inflight`, `TRANSIENT_LOCAL` QoS: the
history is exactly the window, and a node that is still discovering does not
lose the first message. A history shallower than the window would silently
evict an unacked sample, which is why the depth is derived from the one
`inflight` number instead of being set per topic. Topics are relative, so the
whole chain can be put in a namespace.

When the source ends (or `duration`/`max_frames` is reached) the acquisition
node publishes `stop`, waits for the terminal node to write `summary.json` and
reply `done`, and then exits. The launch file shuts the other nodes down when
any node exits (the normal case is the acquisition node; a crash should not
leave a trial hanging).

Each node can be run on its own (`ros2 run yolocar_ros acquisition`, etc.) and
declares its own parameters. The launch file passes one parameter dictionary to
all four nodes; each reads what it needs.

## Tests

```bash
cd ws && colcon test --packages-select yolocar_interfaces yolocar_ros
colcon test-result --verbose
```

* `test_conversions.py` — message ↔ numpy round trips.
* `test_reporting.py` — records/summary writer and the contract keys.
* `test_stage_split.py` — **hardware**: walks the same message round trips and
  stage calls the nodes use on a real frame and asserts the detections are
  bit-for-bit identical to `YoloCarProcessor.process_frame()`.
* `test_flight_window.py` — the credit window as a pure state machine: capacity,
  out-of-order and duplicate acks, and the fact that a lost frame is released by
  a later ack (so contiguity is checked on the records, not here).
* `test_flight_contract.py` — the window as data: topic depth equals `inflight`,
  the launch argument reaches every node, and the count travels with the frame.
* `test_no_drops.py` — **hardware, real graph**: launches the chain at
  `inflight:=3` and asserts the `seq` set is exactly `0..n-1` with no short tail
  (a topic shallower than the window would surface here and nowhere else), and
  that the detections are identical to the same run at `inflight:=1`.
* The hardware tests skip when the engine, the clip, `ros2` or TensorRT is
  absent. The stock `ament_flake8` / `ament_pep257` tests also run.

The tests set `PYTHONPATH` for `monolith/yolocar/src` themselves (see
`test/conftest.py`), because the pipeline project is not installed into the ROS
workspace.
