# Frames in flight: trading latency for throughput

Status: **mechanism implemented, harness and analyser wired, axis not yet
measured.** The numbers in *First light* are a smoke run (2 passes, 200-frame
windows, 180p) whose only job was to prove the mechanism end to end; they are
not a result. Nothing below may be quoted as one until the matrix in
*Measurement plan* has been run.

Related: `chain-shapes.md` (which shape, i.e. how many processes), and
`preprocessing-chain.md` (why the model contract lives in the inference node).

## The question

Every ROS 2 cell so far was measured with **one frame in flight**: the terminal
node acknowledges a frame, the acquisition node decodes the next. That is the
right instrument for *latency* -- nothing queues, so `p50` is the pipeline's
residency -- but it caps the graph at one frame resident, which means at most
one node is doing work at any instant. On a six-core board that is a deliberate
handicap, and it is the reason the four-node chain reads ~20 fps at 720p while
its own CPU-per-frame (70 ms) would allow far more.

The question this axis adds:

> **What does a process boundary buy once the graph may hold more than one
> frame? Where does that buying stop, and what does the window cost?**

Partitioning a pipeline is normally justified by throughput, not by latency:
independent processes can overlap, and the usual price is that a frame's
residency grows by the queueing that overlap creates. The measurement is
therefore a trade, and the deliverable is the curve, not a winner.

## What is being varied, and what is deliberately not

`inflight:=D` is the size of a credit window held by the acquisition node:

* it publishes while fewer than `D` of its frames are unacknowledged;
* every ack from the terminal node releases exactly one;
* frame 0 is published synchronously (its ack is the discovery gate and the
  engine-ready gate), everything after it fills the window greedily;
* the same number is the history depth of every data topic, and the sequence is
  drained before the terminal node writes the summary.

Constraints, stated so a later change cannot quietly invalidate the axis:

* **all concurrency is inter-process.** Every node stays single-threaded with a
  single mutually exclusive callback group, batch size 1. No reentrant groups,
  no `MultiThreadedExecutor`, no batching, no deferred frames.
* **the work is unchanged.** The window changes how many frames are resident,
  never what is computed: the detections are bit-identical across windows
  (`test_no_drops.py`), and each node's own service time is invariant in `D`.
* **nothing is dropped or retried.** A frame lost in the graph invalidates the
  cell rather than shrinking the sample; `depth = D` is derived from the same
  number so the two cannot disagree.

Those constraints are what make the model below exact instead of approximate:
one callback at a time in a linear chain means completion order equals
emission order at every `D` (`records.jsonl` stays `seq`-ordered and `t_mono`
non-decreasing), and a station's service time does not move when the window
grows.

## The model

Closed queueing network with population `D`. Station `i` is a node; its service
time `s_i` is its own stage work plus the inbound hop (serialisation + DDS +
scheduling). Then Little's law on the loop and the bottleneck constraint give

```
X(D) = min( D / Σ s_i , 1 / max s_i )        L(D) = D / X(D) = max( Σ s_i , D · max s_i )
```

with one knee, at

```
D* = Σ s_i / max s_i
```

* `D < D*` -- **latency-bound**: throughput grows linearly in `D`, latency is
  flat at `Σ s_i`. Pipelining is free here in latency terms: nothing queues,
  the work simply overlaps.
* `D = D*` -- every station busy. Throughput and energy per frame have peaked.
* `D > D*` -- **throughput-bound**: `fps` is pinned at `1 / max s_i`, and each
  extra frame costs exactly one bottleneck service time of latency and buys
  nothing.

`D*` is *not* the node count. It equals `N` only for a perfectly balanced chain;
this one is not, because inference plus its inbound hop is roughly half the
residency. From the measured `ros-4node` cells (`Σ ≈ 34-50 ms`,
`max ≈ 17-20 ms`) the prediction is **`D* ≈ 2`**, i.e. the whole throughput
dividend arrives at a window of two and everything beyond it is pure cost.

## Predictions (written before the matrix, so they can fail)

| # | claim | how it fails |
| --- | --- | --- |
| H1 | `fps` at `D=2` is already within noise of `D=8` at both resolutions | `fps` keeps rising -> `D*` is larger than the per-stage estimate, so a hop is much more expensive than `xport` says |
| H2 | `p50(D) ≈ p50(1) + (D-1) · max s` for `D ≥ D*` | the slope is not the bottleneck residency -> the added latency is not queueing at the bottleneck |
| H3 | `stage_ms` medians and per-op medians are flat in `D` | `infer_ms` grows with `D` -> GPU/clock contention, and the knee is a *machine* knee rather than a pipeline one |
| H4 | `fps(D*)` approaches but stays below the monolith's `59.5 fps` | it overtakes -> the ROS CPU-per-frame is for some reason not additive with concurrency |
| H5 | `J/f` improves from `D=1` to `D*` and is flat after | it worsens -> a window buys throughput only by burning more energy per frame |

H5 is the one that decides the narrative. `D=1` spends 8.1 W to deliver
0.408 J/frame; at `D*` the board is doing 2-3x the frames, so even at higher
power the energy per frame should fall by a third or more. That is the real
argument for partitioning, because latency only gets worse.

## Mechanism notes that a reader of the code will ask about

* **The credit is `seq`-identified, not counted.** The acquisition node keeps
  one monotonic watermark and releases everything at or below it, so a
  duplicated or reordered ack is harmless. An added `+1` counter would drift on
  both.
* **`depth = D`, exactly.** A `KEEP_LAST(1)` writer is allowed to evict an
  unacked sample even under `RELIABLE`, and the window would not notice: a later
  ack releases the missing frame too. The only trace is a hole in the `seq` set,
  which is why `test_no_drops.py` asserts contiguity and `analyze.py` counts
  `drops`.
* **`hop_ms` is measured, not inferred.** Each node stamps `t_pub` before it
  publishes; the next node adds `entry - t_pub` at callback entry. At `D=1` that
  is transport (the whole residual is delivery); its growth with `D` *is* the
  queueing delay. `transport_ms` keeps its old definition, so old cells still
  compare -- it just means "everything the stages do not account for", which now
  includes queueing.
* **`inflight` is recorded per frame.** A window that never filled must not be
  mistaken for one that filled and bought nothing: the record carries the
  realized count and the report prints it as `fill`.

## First light (smoke, not a result)

`ros-4node` at 180p, no conditioning chain, 2 passes, 200-frame windows,
`settle_s=5` (the ladder's real cells use 45 s):

| `i` | fill | fps | p50 ms | p99 ms | hop ms | cores | J/f |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | 29.5 ± 0.5 | 29.5 ± 1.1 | 49.2 | 8.8 | 0.84 | 0.406 ± 0.005 |
| 2 | 1.00 | 44.2 ± 1.5 | 39.8 ± 1.9 | 61.5 | 18.5 | 1.27 | 0.303 ± 0.008 |
| 4 | 1.00 | 44.4 ± 1.4 | 85.8 ± 3.3 | 123.0 | 65.0 | 1.26 | 0.303 ± 0.007 |

Read it as a mechanism check, and note how exactly it matches the model:
throughput doubles by `i=2` and is flat from there (H1), `p50` grows by ~46 ms
between `i=2` and `i=4` while `fps` does not move (H2, with a bottleneck
residency of ~23 ms), energy per frame falls 25 % then stops (H5). `fill = 1.00`,
`drops = 0`, and `L·X/i` ≈ 0.9 throughout (the closed-loop consistency check).

## Measurement plan

The ladder is `{1, 2, 4, 8}`; the `i1` cell of each family is the local zero and
must be in the same invocation (the global `anchor` stays the monolith).

| tier | cells | why | est. |
| --- | --- | --- | --- |
| 1 | `ros-4node-720p-n` × {1,2,4,8} + `monolith-720p-n` | the knee, at the resolution where hops cost bytes | ~40 min |
| 2 | `ros-4node-180p-n` × {1,2,4,8} | H4: cheaper hops should lower `D*` and raise the peak | ~35 min |
| 3 | `ros-N12-p8c3-720p-n` × {1,2,4} | 12 nodes, 6 cores: the *machine* saturates before the pipeline | ~60-70 min |

```bash
bench/run.py --out results/fif1 --conditions \
  monolith-720p-n,ros-4node-720p-n,ros-4node-720p-n-i2,ros-4node-720p-n-i4,ros-4node-720p-n-i8
bench/analyze.py --dir results/fif1 --md results/fif1/report.md --json results/fif1/report.json
```

Acceptance before quoting a number: `fill ≈ 1` and `drops = 0` in every windowed
trial; `seq` contiguous; detections identical across windows for a fixed cell;
`latency - acq - Σ stages - hop_ms` in `[0, ~2] ms` and flat in `D`; and no
change to the machine policy (no affinity, governor, GC or clock pinning).

## Why `FIF` is not the name

`FiF` collides with FIFO in every sentence it appears in. The code says
`inflight` (parameter, message field, record key, provenance field) and the cell
token is `-i{D}`, appended only when `D > 1` so the demand-driven cells keep the
names they were measured under.
