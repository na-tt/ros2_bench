# Proposal: an independent preprocessing stage, and a chain-length axis

Status: **implemented** on branch `feature/preprocessing-ops` (commits below).
Not yet measured: the matrix has to be re-run before any cell number is quoted.

This document records four decisions, the scaling model they serve, and the
commit sequence that lands them. It is written *before* the code so the trail
shows intent first, implementation second.

## Problem

`yolocar.preprocessing.preprocess()` currently fuses three unrelated things:

```python
padded, info = letterbox(as_bgr_frame(frame), input_shape, pad_value=pad_value)
return to_tensor(padded, dtype), info
```

1. **Coercion** -- `as_bgr_frame`: gray->BGR, dtype clamp, shape check.
2. **Photometric/structural conditioning** -- *absent today*; this is the stage
   we want to grow (CLAHE, bilateral, gamma, histogram equalisation, ...).
3. **Geometric + layout adaptation** -- `letterbox` + `to_tensor`: pad value
   114, aspect-preserving resize, BGR->RGB, HWC->NCHW, `/255`. Parameterised
   *only* by `inference.InputSpec`, i.e. purely the engine's input contract.

(3) is not image conditioning at all: it is the model contract. (2) is the real
preprocessing stage, and it has to be splittable so chain length becomes a
measurable axis.

Scope note: detection *quality* is explicitly out of scope for this benchmark.
The conditioning ops are stand-ins for realistic real-world preprocessing
workloads; what is being measured is their computational implication, not their
effect on accuracy. `n_det` is reported, not optimised.

## Decision 1 -- the model contract belongs to the inference stage

`letterbox`, `to_tensor`, `LetterboxInfo` and `InputSpec` move to a new,
**engine-free** module `yolocar.model_input`:

| module | responsibility | imports |
| --- | --- | --- |
| `preprocessing` | stage 2: coercion + conditioning at **source resolution** | cv2, numpy |
| `model_input` | the engine input contract: `InputSpec`, `LetterboxInfo`, `letterbox`, `to_tensor`, `model_input()` | cv2, numpy |
| `inference` | stage 3: engine invocation, decode, `input_spec(engine)` | `engine`, `model_input` |

`model_input` must not import `inference`: `inference` imports `engine`
(TensorRT/CUDA), while `postprocessing` and the ROS transport layer need
`LetterboxInfo` and must stay engine-free.

Python-side, `inference` re-exports the moved names so `inference.InputSpec`
keeps working. ROS-side, the inference node calls `model_input()` itself and
publishes `CandidatesFrame` as before -- **no new node, no new topic, and
`TensorFrame` disappears entirely**: the preprocessing node emits an
`ImageFrame`, exactly the type acquisition already publishes.

Why fuse the contract into the inference node rather than giving it its own
node: a 640x640 fp32 tensor is a **fixed 4.9 MB payload, invariant with
resolution**. A hop carrying it would add a resolution-invariant term (~11 ms at
the measured `2.2 ms/MB`) that is confounded with the per-hop fixed cost the
experiment is trying to fit. Fusing keeps the only resolution-scaling payload a
BGR frame -- so the resolution axis measures what it claims to measure.

## Decision 2 -- `imgsz` stays fixed at 640

`engine.py` already supports dynamic shapes (`default_input_shape`,
`_resolve_input_shape`, `set_input_shape` validated against the optimization
profile), so a single engine built with
`--minShapes=1x3x128x160 --optShapes=1x3x640x640 --maxShapes=1x3x768x1344`
could serve every resolution without per-resolution exports.

Deliberately **not** used for the main matrix. If `imgsz` tracks resolution,
YOLO cost becomes ~O(A) and swamps the filters (1280x720 input is ~2.25x the
flops of 640x640). Fixed `imgsz=640` makes inference a *constant* that cancels
in resolution deltas, so the preprocessing slope can be read off directly. The
dynamic-profile build stays documented as a separate experiment (inference
cost vs `imgsz`), not part of this matrix.

## Decision 3 -- one uniform op chain, so any partition is expressible

Every conditioning op is `ImageFrame -> ImageFrame`: BGR uint8 in, BGR uint8
out, **source geometry preserved**. Then:

* a chain is a *list of op names*, a partition is a *list of cut points*, and
  both are data (launch/config), not code;
* one generic `filter_node` binary instantiates any op (`op:=clahe`) and any
  depth, so "preprocessing split across N processes" needs no new node type,
  no new message type, no per-chain Python;
* each op is a pure function `(frame, **params) -> frame` in `yolocar.filters`;
  the ROS node holds zero logic (same rule as every other stage).

Op selection is part of the design, not an afterthought. The set spans
deliberately different bottlenecks, because that is what lets the resolution
axis *show* the difference rather than assume it:

| op | bottleneck | measured, 720p / 180p |
| --- | --- | --- |
| `gamma` (LUT) | memory, 1 pass | 2.4 / 0.2 ms |
| `unsharp` | blur + weighted blend | 7.2 / 0.6 ms |
| `hist_equalization` | memory, multi-pass | 9.1 / 0.6 ms |
| `clahe` | mixed, fixed tile grid | 12.0 / 1.5 ms |
| `gaussian` | separable convolution | 3.5 / 0.4 ms |
| `median` | non-separable, O(k^2) | 16.6 / 1.3 ms |
| `bilateral` | arithmetic, O(k^2) with exp | 50.0 / 8.3 ms |

These are the guesses this table started with, replaced by measurement; the
current numbers and the method live in
`chain-shapes.md#calibration-measured`, which is the single source. A
non-local-means `denoise` was in the original set and was removed after it
measured 1.36 s/frame at 720p -- see that same section for why.

An empty chain must be a **bit-exact, zero-copy no-op** (`return frame`), so
existing cells and the frozen behaviour remain comparable.

## Decision 4 -- keep the contract, add a breakdown next to it

`records.jsonl` keeps `pre_ms` as the **sum** over the whole conditioning
chain (so `analyze.py` and every existing number keep their meaning) and gains
an optional `pre_ops: {"clahe": 4.1, "bilateral": 38.2}` map -- the same
pattern `transport_ms` already uses: present only when there is something to
report, ignored by consumers that do not know it. `summary.json` gets
`stage_ms.preprocess_ops` alongside `stage_ms.preprocess`.

## The scaling model this enables

Chain length `k` and source area `A` (pixels) are two independent axes:

```
T(k, A) = C(A) + k * (a + beta * bytes(A))
```

* `C(A)` = conditioning compute, O(A), invariant in `k`;
* `a` = fixed per-hop cost (serialise/dispatch/scheduling);
* `beta * bytes(A)` = per-hop payload cost.

Resolution alone cannot separate these: every term is O(A), so the *ratio*
transport/compute is constant in the asymptote. **`k` is the axis that
separates `a` from `beta`**, because compute does not move with `k` while
transport is linear in it. From the existing baseline
(`xport` 24.2 ms at 720p over two hops + 4.9 MB tensor):

| resolution | hop cost `a + beta*3A` | an op deserves a node if it costs more than |
| --- | --- | --- |
| 720p (0.92 MP) | ~3.5 + 6.1 = 9.6 ms | ~10 ms |
| 360p (0.23 MP) | ~3.5 + 1.5 = 5.0 ms | ~5 ms |
| 180p (0.058 MP) | ~3.5 + 0.4 = 3.9 ms | ~4 ms |

The fixed term *dominates as resolution falls*, which inverts the usual
intuition: partitioning is most hostile where the payload is smallest. That is
a falsifiable prediction of the implementation below, and the reason the op set
includes both sub-millisecond and 50 ms ops.

One more design consequence, recorded but **not implemented here**: with one
frame in flight a serial chain's latency is the sum of hop latencies, so more
nodes can only lose. Partitioning pays only when stages overlap on different
frames, i.e. in-flight depth > 1, which decouples `latency_ms` from `fps`.
Depth is therefore a factor for a later run, not a constant to hard-code now.

## Commit sequence (as landed)

| # | commit | left the tree |
| --- | --- | --- |
| 1 | `docs: proposal for an independent preprocessing stage and a chain-length axis` | unchanged code |
| 2 | `refactor(yolocar)!: move the model input contract out of preprocessing` | behaviour identical, 147 tests green |
| 3 | `feat(yolocar): a conditioning op registry for the preprocessing stage` | ops exist, unused; 43 new tests |
| 4 | `feat(yolocar): run the conditioning chain in the monolith and record it` | `--pre` works, `pre_ops` recorded |
| 5 | `feat(bench): report the stage-2 chain breakdown when the records carry it` | analyser prints it |
| 6 | `refactor(yolocar_ros)!: stage 2 conditions at source resolution, the inference node owns the contract` | `TensorFrame` deleted; 4 messages, contract off the wire |
| 7 | `feat(yolocar_ros): a data-driven stage-2 partition, one process per op` | `pre_split` — any partition from launch data |

## What is now true

* `preprocessing.condition(frame, chain)` is stage 2: coercion plus ops, source
  geometry, BGR uint8. Empty chain returns the input array itself.
* `model_input.prepare(frame, shape, dtype, pad_value)` is the contract, invoked
  by the inference stage on both front-ends with the engine's own `InputSpec`.
* Ops are `yolocar.filters` functions, registered in `preprocessing.OPS`, and
  named by a chain string (`"clahe;bilateral:d=9"`, `;` for the ROS/shell form,
  or a sequence for the CLI's repeatable `--pre`).
* Both front-ends record `pre_ops` per frame (optional key) and
  `stage_ms.preprocess_ops` in the summary; `analyze.py` reports a per-op table
  for the cells that have it.
* `pre_split:=true` runs one `preprocessing` process per op, head to tail,
  through `yolocar/filter_<i>` topics, with `pre_ms` and the per-op arrays
  accumulated additively so the records are identical in both shapes.

## Smoke evidence (not a measurement)

180p, `pre_ops:=clahe;gamma:gamma=0.8`, model n, single short runs:

| shape | p50 | pre_ms | xport |
| --- | --- | --- | --- |
| one stage-2 node | 28.4 ms | 3.7 ms | 10.6 ms |
| one node per op | 32.9 ms | 3.1 ms | 12.4 ms |

Consistent with the per-hop model above (`+1.8 ms` observed for the extra hop at
180p, `+4.0 ms` predicted, single non-interleaved pair), and with "splitting two
cheap ops does not pay". That is a smoke test, not a result: it is one pass, no
interleaving, no anchor delta. The k x A grid below is what would make it one.

## Not implemented, on purpose

* In-flight depth > 1 (the only configuration in which partitioning can win).
* The dynamic-shape engine / per-resolution `imgsz` run (Decision 2).
* `bench/scenarios.json` cells for the k x A grid: they are `pre_ops`/
  `pre_split` values on the existing `ros-4node` template, so they can be added
  without touching the harness -- but the budget question (3 ops x 3 chain
  depths x 3 resolutions x 3 passes) is an operator decision, not a code one.
