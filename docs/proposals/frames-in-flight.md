# Frames in flight: trading latency for throughput

Status: **mechanism implemented, harness and analyser wired, tier 1 measured
once** (3 passes, 720p, the cell now called `ros-N4-p2c1`) -- on a board running
at 94-97 % load, so the ladder's *shape* is informative and its absolute numbers
are not. That run predates the conditioning chain becoming a constant of the
experiment: its `i1` cell ran stage 2 empty, so every measured number here is
about 100 ms/frame cheaper at 720p than the ladder the harness now launches.
Tiers 2-3 are unmeasured.

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
residency. From the measured op-free cells (`Σ ≈ 34-50 ms`, `max ≈ 17-20 ms`)
the prediction was **`D* ≈ 2`**, i.e. the whole throughput dividend arrives at a
window of two and everything beyond it is pure cost.

**The conditioning chain moves that knee, because it unbalances the chain.**
Every cell now runs the seven ops, and they cost 103.5 ms/frame at 720p against a
~20 ms inference node: `pre` becomes the bottleneck, `Σs / max s` falls to
≈1.3, so the whole dividend arrives at `D=2` and every extra frame beyond it is
pure latency. The ceiling there is flat at `1/103.5 ms ≈ 9.7 fps` whatever `D`
is. At 180p the ops are 13.6 ms and the op-free estimate roughly holds. The model
is unchanged; the per-stage residencies it consumes are not.

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

## Mechanism check (180p smoke, not a result)

`ros-N4-p2c1` at 180p, **no conditioning chain** (historic: this was the chain
before stage 2 became a constant, so the ladder now runs ~13.6 ms/frame more),
2 passes, 200-frame windows, `settle_s=5` (the ladder's real cells use 45 s):

| `i` | fill | fps | p50 ms | p99 ms | hop ms | cores | J/f |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | 29.5 ± 0.5 | 29.5 ± 1.1 | 49.2 | 8.8 | 0.84 | 0.406 ± 0.005 |
| 2 | 1.00 | 44.2 ± 1.5 | 39.8 ± 1.9 | 61.5 | 18.5 | 1.27 | 0.303 ± 0.008 |
| 4 | 1.00 | 44.4 ± 1.4 | 85.8 ± 3.3 | 123.0 | 65.0 | 1.26 | 0.303 ± 0.007 |

Read it as a mechanism check, and note how closely it matched the model on a
ladder whose hops were cheap:
throughput doubles by `i=2` and is flat from there (H1), `p50` grows by ~46 ms
between `i=2` and `i=4` while `fps` does not move (H2, with a bottleneck
residency of ~23 ms), energy per frame falls 25 % then stops (H5). `fill = 1.00`,
`drops = 0`, and `L·X/i` ≈ 0.9 throughout (the closed-loop consistency check).

## Measurement plan

The ladder is `{1, 2, 4, 8}`; the `i1` cell of each family is the local zero and
must be in the same invocation (the global `anchor` stays the monolith).

| tier | cells | why | est. | state |
| --- | --- | --- | --- | --- |
| 1 | `ros-N4-p2c1-720p-n` x {1,2,4,8} + `monolith-720p-n` | the knee, at the resolution where hops cost bytes | ~2 h | measured once, loaded board, op-free |
| 2 | `ros-N4-p2c1-180p-n` x {1,2,4,8} | H4: cheaper hops should lower `D*` and raise the peak | ~1 h | to do |
| 3 | `ros-N12-p8c3-720p-n` x {1,2,4} | 12 nodes, 6 cores: the *machine* saturates before the pipeline | ~60-70 min | to do |

```bash
bench/run.py --out results/fif1 --conditions \
  monolith-720p-n,ros-N4-p2c1-720p-n,ros-N4-p2c1-720p-n-i2,ros-N4-p2c1-720p-n-i4,ros-N4-p2c1-720p-n-i8
bench/analyze.py --dir results/fif1 --md results/fif1/report.md --json results/fif1/report.json
```

Acceptance before quoting a number: `fill ~ 1` and `drops = 0` in every windowed
trial; `seq` contiguous; detections identical across windows for a fixed cell;
`latency - acq - sum stages - hop_ms` in `[0, ~2]` ms and flat in `D`; and no
change to the machine policy (no affinity, governor, GC or clock pinning).

## Tier 1, first pass (720p, 3 passes, loaded board)

Same session throughout, cells interleaved in one block, so the within-pass
deltas hold even though the board did not (see the caveats). `i1` is the same
graph as `ros-N4-p2c1-720p-n` was under the old, op-free scenario, so this run
also re-measures it -- but **it is not the cell the harness now launches**: add
~103.5 ms/frame for the conditioning chain before comparing a fresh `i1` to it.

| `i` | fill | frames | fps | p50 ms | p95 ms | p99 ms | hop ms | cpu ms/f | cores | W | J/f |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | 1766 | 8.97 +/- 0.15 | 101.1 | 157.6 | 178.8 | 74.7 | 64.1 | 0.58 | 9.9 | 1.107 |
| 2 | 1.00 | 1732 | 14.24 +/- 0.16 | 126.4 | 202.3 | 235.0 | 98.6 | 61.0 | 0.87 | 10.2 | 0.721 |
| 4 | 1.00 | 2181 | 20.45 +/- 0.24 | 180.9 | 259.5 | 303.4 | 154.5 | 58.9 | 1.20 | 10.6 | 0.521 |
| 8 | 1.00 | 2500 | 28.65 +/- 4.48 | 268.0 | 352.6 | 394.3 | 215.7 | 54.8 | 1.56 | 11.0 | 0.394 |

Anchor in the same run: `monolith-720p-n` 41.55 +/- 0.71 fps, p50 18.38 ms,
J/f 0.274. Within-pass deltas against the family's own `i1`: `fps` +5.3 / +11.5
/ +19.7, `p50` +25 / +80 / +167 ms, `J/f` -0.386 / -0.586 / -0.713 (all CIs
exclude zero).

What it establishes:

* **the window is real and full** -- `fill = 1.00`, `drops = 0`, and `L*X/i`
  between 0.91 and 0.93 in every row. So the added latency is queueing (p99
  179 -> 394 ms) and not slower work.
* **the dividend is large and it is energy**, not just throughput: 3.2x the
  frames at 64 % less energy per frame. At `i=8` the ROS chain is at 28.7 fps
  and 0.394 J/f, against the monolith's 41.6 fps and 0.274 J/f.

What it contradicts, on this board:

* **H1 fails at 720p.** Throughput is still rising at `D=8`, no plateau. The
  knee is `sum s / max s`, and here `sum s` is dominated by the hop (74.7 ms of
  ~111 ms at `i1`), so a big window keeps paying. This is why `D*` must be
  computed per board/session instead of assumed: the 180p smoke reached its
  knee at 2 because its hops were 15x cheaper.
* **H3 fails, in the favourable direction.** `cpu ms/f` *falls* with the window
  (64.1 -> 54.8) and the effective bottleneck residency shrinks, so extra frames
  keep buying throughput rather than only adding latency. The plausible reading
  is that a per-hop cost at `D=1` is partly a wakeup/idle tax that amortizes once
  frames are resident -- i.e. the partitioning overhead is a function of
  concurrency, not only of structure. That is a hypothesis, not a result: on a
  loaded board CPU starvation and amortization are confounded.

Caveats, attached rather than buried:

1. **The board was at 94-97 % load** (load average 9-10 on 6 cores; the desktop
   session, mostly the terminal emulator). Absolute values are therefore not
   comparable to the baseline table -- the anchor confirms it (`monolith-720p-n`
   41.6 fps here against 59.5 fps on a quiet board). Only the within-pass deltas
   and the internal consistency checks mean anything.
2. **Three of four ROS cells under-filled their 2500-frame window** (1766, 1732,
   2181) because their `fps_hint` predates this load: the percentiles are over
    different `n`. `i8` reached 2500.
3. **`i8` `r1` is an outlier** (23.5 fps against 30.9 and 31.5), which is why
   that row's std is 4.5 fps.

Before tier 2 and any comparison across sessions: re-derive the ladder's
`fps_hint` from a quiet board, add an `i16` point at 720p (the curve had not
turned), and re-run tier 1.

## Why `FIF` is not the name

`FiF` collides with FIFO in every sentence it appears in. The code says
`inflight` (parameter, message field, record key, provenance field) and the cell
token is `-i{D}`, appended only when `D > 1` so the demand-driven cells keep the
names they were measured under.
