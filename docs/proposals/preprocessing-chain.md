# Proposal: an independent preprocessing stage, and a chain-length axis

Status: **implementing** (branch `feature/preprocessing-ops`).

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
   we want to grow (denoise, CLAHE, gamma, histogram equalisation, ...).
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

| op | bottleneck | cost at 720p (order of magnitude) |
| --- | --- | --- |
| `gamma` (LUT) | memory, 1 pass | ~0.5 ms |
| `hist_equalization` | memory, multi-pass | ~1-2 ms |
| `clahe` | mixed, tile histograms | ~4 ms |
| `gaussian` | separable convolution | ~3 ms |
| `median` | non-separable, O(k^2) | ~10-20 ms |
| `bilateral` | arithmetic, O(k^2) with exp | ~30-60 ms |
| `denoise` (fastNlMeans) | arithmetic + search window | ~100 ms+ |

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

## Commit sequence

| # | commit | leaves the tree |
| --- | --- | --- |
| 1 | this document | unchanged code |
| 2 | `refactor: move the model contract into model_input` | behaviour identical, tests green |
| 3 | `feat: conditioning op registry + filters` | new stage exists, unused |
| 4 | `feat: wire ops into the monolith and records` | `--pre` works, `pre_ops` recorded |
| 5 | `feat(bench): report the per-op breakdown` | analyser reads it |
| 6 | `refactor(ros): inference node owns the contract` | `TensorFrame` gone, 3 nodes + 1 hop removed from the wire |
| 7 | `feat(ros): generic filter node` | any chain partition expressible |
| 8 | `docs: stage map, contract and scenario notes` | documentation matches code |
