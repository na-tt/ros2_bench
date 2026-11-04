# Chain shapes: how many processes, and where the cuts go

Status: **implemented, not yet measured** on branch
`feature/preprocessing-ops`. All eight shapes launch from one `chain:=`
argument, the identity invariant is a test, and the benchmark has cells for the
axis. What is missing is the measurement: nothing below may be quoted as a
result until the matrix has been re-run. Branch `feature/preprocessing-ops` already ships
the mechanism for one axis (stage 2, `pre_ops` + `pre_split`); this document is
about generalising it to the whole pipeline and about what each shape is
allowed to claim.

Related: `preprocessing-chain.md` (why the model contract moved into the
inference node -- that decision is a precondition for the numbers here).

## The question, stated so it can fail

The existing axis is *resolution*. This proposal adds *chain length*, and the
point of it is not "which shape is fastest" -- with one frame in flight that
answer is known in advance and is boring (fewer hops, always). The point is:

> **What does a process boundary cost, where is it cheapest, and does that
> change with resolution?**

Everything below is measured at one frame in flight. A shape's verdict can look
different once the graph may hold more than one: a deep chain of cheap nodes is
exactly what pays off when the pipeline can be filled, and that is a separate
axis with its own instrument (`frames-in-flight.md`). No result here is a
statement about it.

A shape is a decomposition of fixed work. Because the work is identical, the
only thing a shape changes is (a) how many times pixels or candidates are
serialised across a process boundary and (b) how many times the scheduler is
woken per frame. Separating those two is the entire value of the axis, and it
is what makes the k-sweep complementary to the resolution sweep.

## Three things "chain length" is used to mean

| quantity | symbol | what it actually costs |
| --- | --- | --- |
| number of processes | `N` | per-node wakeup, RSS, DDS participants |
| number of hops | `H = N-1` | serialisation + copy + scheduling, per hop |
| **which payload crosses each hop** | -- | bytes; the only term that scales with resolution |

`N` and `H` are locked together in a linear chain, so they are one axis. The
third is not a number at all, and it dominates: the payload drops by three
orders of magnitude at exactly one boundary.

### Payload classes

| class | what crosses | bytes | resolution-dependent |
| --- | --- | --- | --- |
| **pixel** | `ImageFrame` (BGR uint8, source geometry) | `3*A` where `A` = 0.92 / 0.23 / 0.058 MP | yes, linearly |
| **candidate** | `CandidatesFrame` (`22*N_c` B: 4+1 floats + 1 uint16 per box) | ~0.5-7 kB typical (`max_candidates=30000` -> 660 kB worst case) | **no** |
| **control** | `FrameAck` / `Empty` | ~40 B | no |

At 720p: 2.76 MB / ~2 kB / 40 B. The interesting consequence is that a hop's
cost is `a + beta*bytes` with `a ~= 3.6 ms`, `beta ~= 2.2 ms/MB`, so
**a candidate hop costs `a` and nothing else** -- it is a fixed 3.6 ms
irrespective of the frame, and the same 3.6 ms at 180p and at 720p.

One caveat that must be recorded with the data: the candidate payload is
*content-dependent*, unlike the pixel payload which is exactly `3*A`. Splitting
stages 4-6 changes nothing about `N_c` (the detections are identical by
construction, see below), so this is safe *within* the axis -- but it means a
candidate hop measured on this clip is not a lower bound for a busier scene.
Record `n_cand` (candidates entering NMS) so the report can carry it.

## Naming: the label is a coordinate, not a count

An earlier draft of this document labelled shapes `R3e` / `R3l` (early / late
cut relative to inference) and `R5m` (merged). The first two have a basis --
the position of the cut in the stage order -- but they only work while
`infer` is the reference point, and at five nodes that reference no longer
disambiguates anything, so `m` became a mnemonic instead of a rule. Nothing is
built on those names yet, so this is the cheap moment to replace them.

**Basis: the stage list.** Every configuration here is a *contiguous partition
of the same twelve stages*, in canonical order:

```
acq | op1 op2 op3 op4 op5 op6 op7 | infer | nms filter coord
       \_________________________/          \_________/
          the pixel segment                 the candidate segment
```

A shape is therefore just a set of cuts, with `N = cuts + 1`. Two counts
describe any partition completely, because the cuts fall into exactly two
classes:

| | meaning | |
| --- | --- | --- |
| `p` | cuts at or before `infer` | hops that carry **pixels** (`3A` bytes) |
| `c` | cuts after `infer` | hops that carry **candidates** (~`22*N_c` bytes) |

with `N = p + c + 1`. So the label is **`p{p}c{c}`**, e.g. `acq|pre|infer|post`
= `p2c1`, and the 12-stage chain = `p8c3`. Those are the two numbers the cost
model actually consumes:

```
T = C + p*(a + beta*3A) + c*(a + beta*bytes_candidates)
```

which makes the label a coordinate rather than a nickname: collision-free by
construction, and it states how much of the chain is pixel traffic instead of
how many processes there are. `p8c3` is a more honest name than "12-node": 8 of
its 11 hops carry a frame and 3 carry candidates, and the label says so; the
node count says nothing about that ratio, which is the whole point.

The **shape string** stays the canonical identity (`acq|pre|infer|post`): it is
what the launch argument takes and what the provenance records, and the label is
its short form for humans and directory names. The **conditioning chain is not
part of the label** because it is not an axis: every cell runs the same seven
ops (`conditioning.ops` in `bench/scenarios.json`), the monolith spells them as
repeated `--pre` and the launch file takes `pre_ops:=`, and `run.py` rejects any
cell whose command disagrees. A cell name therefore encodes architecture,
resolution, model and frames in flight, and nothing else.

## The configurations

Notation: `|` separates processes, `+` means "these stages share one process".
Stage ids are the repo's own: `acq`, `pre` (the whole conditioning chain),
`infer`, `nms`, `filter`, `coord`, and `op1..op7` for the individual
conditioning ops.

| label | shape | N | `p` | `c` | what it isolates |
| --- | --- | --- | --- | --- | --- |
| `monolith` | `acq+pre+infer+nms+filter+coord` | 1 | 0 | 0 | single process (existing anchor) |
| **`p1c0`** | `acq` \| `pre+infer+post` | 2 | 1 | 0 | **the process boundary alone** |
| **`p2c0`** | `acq` \| `pre` \| `infer+post` | 3 | 2 | 0 | ...vs cutting *before* the model |
| **`p1c1`** | `acq` \| `pre+infer` \| `post` | 3 | 1 | 1 | ...vs cutting *after* it |
| **`p2c1`** | `acq` \| `pre` \| `infer` \| `post` | 4 | 2 | 1 | current chain (`ros-N4-p2c1`) |
| `p2c2` | `acq` \| `pre` \| `infer` \| `nms` \| `filter+coord` | 5 | 2 | 2 | cheap cut, one hop |
| `p2c3` | `acq` \| `pre` \| `infer` \| `nms` \| `filter` \| `coord` | 6 | 2 | 3 | candidate side exhausted |
| **`p3c1`** | `acq` \| `op1+op2` \| `op3..op7` \| `infer` \| `post` | 5 | 3 | 1 | **one pixel hop more than `p2c1`** (one step up the pixel side) |
| `p8c3` | `acq` \| `op1`..`op7` \| `infer` \| `nms` \| `filter` \| `coord` | 12 | 8 | 3 | both sides exhausted |

Note that `p2c2` and `p3c1` are both five-node shapes and are *adjacent in the
table but not in cost*: same count, opposite side of the inference boundary.

`p2c0` / `p1c1` are the pair worth staring at: **the same number of processes,
a different cut, 6.1 ms apart at 720p** (19.4 vs 13.3 ms of transport) because
one puts the pixel payload on the wire *twice* and the other *once*. That is
the hypothesis in one line:

> **H1: cost is governed by the payload class of the boundary, not by the
> number of nodes.**

Two consequences, both falsifiable:

* `p1c1` should beat `p2c0` in p50 despite being indistinguishable on any
  process-oriented metric (both are 3 nodes, 4 stages, ~the same RSS). If they
  tie, the payload does not matter and the `beta` term is measuring something
  else.
* `p1c0` should be the cheapest ROS shape of all, since it carries the fewest
  hops (one pixel, no candidate). If it is not -- if a 2-node chain is slower
  than a 4-node one -- there is a fixed per-process cost that the `a` term has
  been hiding, and the transport model in the READMEs is missing a term.

### Predicted deltas (all vs `p2c1`, ms of p50)

From the per-hop model fitted on the five-message chain (`a=3.6`,
`beta=2.2 ms/MB`; the same fit that predicted the measured 9.6 / 5.0 / 3.9 ms
per hop at 720p / 360p / 180p):

| shape | 720p | 360p | 180p | because |
| --- | --- | --- | --- | --- |
| `p1c0` | **-13.3** | -8.7 | **-7.6** | one pixel + one candidate hop fewer |
| `p1c1` | **-9.7** | -5.1 | **-4.0** | one pixel hop fewer |
| `p2c0` | -3.6 | -3.6 | -3.6 | one candidate hop fewer |
| **`p3c1`** | **+9.7** | +5.1 | **+4.0** | one pixel hop more -- the additivity check |
| `p2c2` | +3.6 | +3.6 | +3.6 | one candidate hop more |
| `p2c3` | +7.2 | +7.2 | +7.2 | two candidate hops more |
| `p8c3` | **+65.4** | +37.8 | **+31.2** | 6 pixel + 2 candidate hops more than `p2c1` (11 hops) |

`p3c1` is worth its own cell rather than a footnote: it is `p2c1` plus exactly
one pixel hop and nothing else, so its delta should reproduce `a + beta*3A` within
noise. If it does not, additivity -- the assumption every other prediction on
this page rests on -- is false, and the k-sweep has to be read as a measured
`T(k)` curve rather than resolved into per-hop terms.

Read the last row against the monolith's whole p50: at 180p, `p8c3`'s *transport
alone* (8x4.0 + 3x3.6 = 42.8 ms) is **3.4x the monolith's total latency**
(12.7 ms). That is the strongest statement the axis can produce, and it needs
no model at all to be interesting -- only the measurement.

## The resolution interaction (why both axes are needed)

Both terms on the pixel branch scale with `A`, but not identically:

* hop: `a + beta*3A` -- 3.6 ms fixed, growing to 9.7 ms at 720p. Ratio
  720:180 is **2.4x for 16x the pixels**.
* op compute: ~linear in `A` for arithmetic ops, but *sub*linear for the
  memory-bound ones at small `A` (a 0.17 MB frame is cache-resident on
  Orin's 2 MB L2; a 2.76 MB frame is not).

So the *ratio* transport/(transport+compute) rises as resolution falls, which
is the opposite of the folk intuition. It is already visible in the existing
baseline without any new measurement:

| cell | xport | p50 | share |
| --- | --- | --- | --- |
| `ros-4node-720p-n` (op-free, historic) | 24.2 ms | 42.4 ms | 57 % |
| `ros-4node-180p-n` (op-free, historic) | 18.4 ms | 36.2 ms | 51 % |

and it is the quantitative form of "an op should only get its own node if it
costs more than the hop it needs". On the calibrated six-op chain that is a
short list, and it shrinks with resolution:

| hop cost | ops that beat it |
| --- | --- |
| 9.7 ms (720p) | clahe (16.7), bilateral (53.9), median (16.4), hist_eq (11.9); gaussian (9.5) is a wash |
| 4.0 ms (180p) | bilateral (9.8), clahe (4.9) -- and clahe only marginally |

So at 720p a per-op split is *sometimes* justified by compute alone, while at
180p it is justified only for the two most expensive ops, and only if the
resulting graph is worth it for some reason other than latency (it never is
under one frame in flight -- see *What this axis cannot answer*).

**H2: at small resolution a chain of hops is dominated by fixed per-hop cost,
and no amount of shrinking the frame fixes it.** The k-sweep at 180p is the
test: the slope of `xport` in `k` should be flat at 3.9 ms/hop while the
720p slope is 9.6 ms/hop.

## Correctness invariant (non-negotiable)

Every shape must produce **identical detections** for the same frames. Same
`n_det` per frame, same boxes, same classes. A split that changes the result is
a bug in the split, not a data point -- and it would silently change the
candidate payload, which is the very thing being measured.

Cheap to enforce: one test runs the same 20 frames through every configured
shape and compares `detections.jsonl` ignoring the timing keys. It is the ROS
analogue of the monolith's frozen `tests/golden/postprocess.json`.

## What has to change in the code

All of it is mechanical, and none of it touches CV logic.

1. **One `chain` parameter instead of `pre_split`.** A shape is a list of
   groups; a group is a list of stage ids. `chain:=acq|pre|infer|post` is
   `p2c1`, `chain:=acq|pre+infer|post` is `p1c1`, and so on. The existing
   `stage_two_instances()` in the launch file generalises to
   `instances(shape) -> list[{name, node, in_topic, out_topic, params}]` --
   still a pure function, still testable without launching anything. Keep
   `pre_ops`/`pre_split` as sugar over it, or drop them once cells are migrated.
2. **Stage-runner nodes take a stage list and in/out topics.** `filter_node`
   already does this for stage 2. `PostprocessingNode` needs the same treatment
   (`stages:=nms,filter,coord`, `in_topic`, `out_topic`), and only the terminal
   node writes `records.jsonl` / `summary.json` / the ack. Without this the
   right-hand branch cannot be measured at all.
3. **Timings must stay additive per stage.** Splitting stage 4-6 needs the
   same trick stage 2 uses: `post_op_names` / `post_op_ms` arrays on
   `FrameMeta`, `post_ms` accumulated not overwritten, an optional `post_ops`
   record key, and `stage_ms.postprocess_ops` in the summary. Otherwise a split
   postprocessing node reports only its own share and "post got cheaper" becomes
   an artifact.
4. **The shape must be in the provenance.** Done: `summary.provenance.shape`
   carries the `chain` string and the detections sidecar's producer is the same
   string, so an artefact names the graph that made it. A records file from a
   12-node run would otherwise be indistinguishable from a 4-node one -- same
   stage buckets, different transport -- which defeats the audit trail.
5. **`n_cand` as an optional record key** -- *not implemented*. The payload
   caveat above still applies: a candidate hop's cost is a function of the
   candidate count, which the records do not carry. Until it does, a candidate
   hop measured on this clip is not a lower bound for a busier scene.

The ack does **not** need work: it stays one control hop from the terminal node
to acquisition regardless of `N`. The per-frame scheduling cost of `N` nodes is
inside the hop term, which is precisely what the sweep measures.

## Stage 2 order: which op is `op1`

The seven conditioning ops are one canonical chain, applied in this order:

| | op | function | bottleneck | cost driver |
| --- | --- | --- | --- | --- |
| `op1` | `clahe` | local contrast (YCrCb round trip) | mixed | fixed tile count, per-pixel LUT interpolation |
| `op2` | `bilateral` | edge-preserving smoothing | **arithmetic** | `exp` per neighbour, `O(k^2)` |
| `op3` | `gaussian` | separable blur | mixed | `O(pixels * k)` |
| `op4` | `gamma` | tone curve via LUT | **memory** | one pass, cheapest op in the set |
| `op5` | `hist_eq` | luma histogram equalisation | **memory, multi-pass** | histogram + LUT + apply |
| `op6` | `median` | non-separable median | arithmetic | `O(pixels * k^2)`, small constants |
| `op7` | `unsharp` | unsharp mask (blur + blend) | mixed | one separable Gaussian plus a memory-bound `addWeighted` |

**Why this order is admissible.** With one frame in flight the ops are strictly
sequential, so their order cannot change the sum of their costs -- only *where
in the graph* each cost lands, which is what `p` counts, not which op sits at
which index. The one order-dependent effect is cache locality: seven ops inside
one node re-read a hot frame, seven nodes re-read a cold one, and that
difference is exactly the "decomposition makes compute slower" term the axis is
supposed to expose, so it is part of the measurement rather than a confound.
Concretely: any permutation of `op1..op7` yields the same shape costs, and the
default order is the one a practitioner would write, so the demo renders look
plausible and the cells are comparable to a real pipeline.

Two consequences that are *not* neutral and must be recorded:

* **The per-op cost is not proportional to `p`.** `p` counts hops (a transport
  quantity); the fact that `op2` costs 21x `op4` is a compute fact that shows
  up in `pre_ops` and in the *node* that owns it, not in the label.
* **The seventh slot is a budget decision, not a taste one.** It was
  `denoise` (non-local means) until measurement: 1.36 s/frame at 720p, i.e.
  every op-split cell at 0.6 fps and an hour per trial. It is replaced by
  `unsharp` and deleted from the package -- the measurement stays in this
  document as the reason, so it is not re-added by someone who wants a
  "proper" denoiser in the chain.

### Calibration (measured)

Each op run alone through the monolith CLI, then the whole chain, reading
`pre_ops` from the records the run already writes. 200 frames per run, first 20
dropped, medians of the rest. **Method note:** 8-frame runs are useless here --
the same op moved by 3x between runs (gaussian at 180p: 1.5 then 5.5 ms), which
is the noise floor, not the op. Everything below is from the 200-frame pass.

| | op | 720p | 180p | 720p/180p | in-chain @720p |
| --- | --- | --- | --- | --- | --- |
| `op1` | clahe | 12.0 | 1.50 | 8.0 | 13.4 |
| `op2` | bilateral | 50.0 | 8.30 | 6.0 | 51.2 |
| `op3` | gaussian | 3.5 | 0.37 | 9.5 | 2.4 |
| `op4` | gamma | 2.4 | 0.22 | 10.9 | 1.4 |
| `op5` | hist_eq | 9.1 | 0.59 | 15.4 | 8.3 |
| `op6` | median | 16.6 | 1.27 | 13.1 | 15.4 |
| `op7` | unsharp | 7.2 | 0.57 | 12.7 | 6.3 |
| | **chain (`op1..op7`)** | **103.5** | **13.6** | 7.6 | sum 98.4 |
| | sum of singles | 100.8 | 12.8 | | |

The pixel ratio is 15.9x (0.92 vs 0.058 MP), so the `720p/180p` column says
which term each op is dominated by: `hist_eq` (15.4), `median` (13.1),
`unsharp` (12.7) and `gamma` (10.9) scale with pixels, while `clahe` (8.0) and
`bilateral` (6.0) carry a large size-independent part -- clahe because its tile
count is fixed, bilateral because it is arithmetic-bound rather than
bandwidth-bound. That is the span the set was chosen for, and it is now
measured rather than asserted.

Three things follow, and only the first changes the design:

1. **All seven ops stay in the matrix.** Every op now costs between 0.2 and
   50 ms at 720p; there is no pathological member left to quarantine. The full
   `p8c3` cell is `88.4 (transport) + 103.5 (chain) + ~17 (acq, contract,
   engine, decode) ~= 209 ms/frame`, i.e. ~4.8 fps and a ~600 s ceiling per
   trial (~10 min). At 180p it is ~70 ms/frame, ~14 fps, ~230 s.
2. **A chain costs what its ops cost, to within 3 %** at 720p (103.5 measured
   vs 100.8 summed) and 6 % at 180p. This is the assumption the entire per-hop
   model rests on: splitting the chain does not silently change the compute it
   splits, so a delta between shapes is a transport delta. It had to be
   checked, not assumed.
3. **With seven ops, compute and transport are the same order of magnitude,
   which is the only regime where the question has an answer.** Compute share
   of the `p8c3` pipeline: **54 %** at 720p (103.5 vs 88.4), ~33 % at 360p,
   **24 %** at 180p (13.6 vs 42.8). The crossover sits between 360p and 720p
   and moves with the shape, which is exactly "which grows faster with
   resolution" -- stated as a number rather than an adjective.

One correction, recorded because it was written down wrongly first: an earlier
draft of this section claimed in-chain ops are dramatically cheaper than
standalone (gaussian 9.5 -> 2.4 ms). On 200-frame data the expensive ops are
within 10 % and only the *cheap* ones read 30-40 % cheaper (gamma 2.4 -> 1.4,
gaussian 3.5 -> 2.4), which is a ~1 ms cold-start effect on a ~1 ms op. The
8-frame numbers that produced the first claim were noise.

## Shape -> process map

Every shape in this document is expressible with **one** new executable,
`chain_node`, parameterised by which stages it runs and by the cut points
inside stage 2 and stage 4-6. Nothing is duplicated per shape and no shape gets
its own binary: a shape is a list of configs, and the launch file is the only
thing that knows the shape.

| shape | processes | notes |
| --- | --- | --- |
| `monolith` | `python3 -m yolocar.main` | 1 process, no ROS |
| `p1c0` | `acquisition` \| `chain_node stages:=pre,infer,post` | the monolith in a second process: 1 pixel hop, nothing else |
| `p2c0` | `acquisition` \| `chain_node stages:=pre` \| `chain_node stages:=infer,post` | |
| `p1c1` | `acquisition` \| `chain_node stages:=pre,infer` \| `chain_node stages:=post` | pixels cross once |
| `p2c1` | `acquisition` \| `pre` \| `infer` \| `post` | `chain_node` three times, `stages:=` one family each |
| `p2c2` | `... \| post_stages:=nms` \| `post_stages:=filter,coord` | same binary, different slice |
| `p2c3` | `... \| post_stages:=nms` \| `post_stages:=filter` \| `post_stages:=coord` | |
| `p3c1` | `acquisition` \| `pre pre_ops:=<ops 1-2>` \| `pre pre_ops:=<ops 3-7>` \| `infer` \| `post` | two aggregate stage-2 nodes |
| `p8c3` | `acquisition` \| 7x `pre pre_ops:=<one op>` \| `infer` \| 3x `post_stages:=<one stage>` | the 12-process extreme |

Two properties make this safe to generalise rather than special-case:

* **The terminal node is derived, not configured.** Whichever node runs the last
  stage (`coord`) is the one that writes `records.jsonl`/`summary.json` and
  sends the ack. A shape that ends somewhere else is not expressible, so the
  file contract cannot be silently lost.
* **Timing stays additive per stage.** `pre_ms`/`infer_ms`/`post_ms` are
  accumulated by each node that contributes, and the per-stage breakdown travels
  in parallel arrays (`pre_op_names`/`pre_op_ms`, and the same for post), so a
  split stage reports the same number as an unsplit one and the extra hops land
  in `transport_ms`.

## Implementation plan

Twelve commits, each leaving the tree in a state that runs and a reason to be
bisected, in the order that keeps the blast radius smallest:

| # | commit | leaves behind |
| --- | --- | --- |
| 1 | `docs: shape naming, stage-2 order and the shape -> process map` | this document; no code |
| 2 | `docs(yolocar): measure the stage-2 ops and pin the chain` | the calibration above |
| 3 | `feat(yolocar): unsharp masking is the seventh op, denoise is gone` | op set final: seven ops, no outlier |
| 4 | `refactor(yolocar_ros): the terminal bookkeeping is one class` | `Terminal` owns records/summary/ack; postprocessing uses it |
| 5 | `feat(yolocar_ros): one stage-runner node for any contiguous stage span` | **done** -- `ChainNode`, engine loaded only by the `infer` span, terminal derived from `coord` |
| 6 | `refactor(yolocar_ros)!: the three stage nodes are chain_node with fixed stages` | **done in 5**: the three executables are one-line entry points |
| 7 | `feat(yolocar_ros): split stage 4-6, with per-stage timing` | **done in 5**: `post_stages`, `post_ops`, `stage_ms.postprocess_ops` |
| 8 | `feat(yolocar_ros): the launch file composes a shape from a chain string` | **done** -- `chain:=` + `pre_groups:=`, partition policy as a pure function |
| 9 | `feat(yolocar_ros): group the stage-2 chain into several nodes` | **done in 8**: `pre_groups:=2,5` etc. |
| 10 | `test(yolocar_ros): every shape yields the same detections` | **done** -- 11 tests, all eight shapes |
| 11 | `feat(bench): cells for the shape axis` | **done** -- 16 cells, cell name checked against the graph |
| 12 | `docs: re-measure and record the shape axis` | **not done**: needs the machine for a few hours |

Steps 1-11 have landed; step 12 is a measurement session, not a commit. The
calibration changed one decision (the seventh op: `denoise` out, `unsharp` in)
and confirmed the rest. Three things the implementation added to the design
without changing it: the node learns nothing about the shape (only its own
stage span), the terminal node is derived from running `coord`, and the cell
name is validated against the graph before anything launches.

## Measurement design

Cells keep the existing `<chain>-<resolution>-<model>` form, and the chain
factor spells the shape as **`ros-N{x}-p{p}c{c}`**, where `x` is the total
process count and the label is the coordinate from the naming section:

| cell prefix | N | p | c |
| --- | --- | --- | --- |
| `ros-N2-p1c0` | 2 | 1 | 0 |
| `ros-N3-p2c0` | 3 | 2 | 0 |
| `ros-N3-p1c1` | 3 | 1 | 1 |
| `ros-N4-p2c1` | 4 | 2 | 1 |
| `ros-N5-p2c2` | 5 | 2 | 2 |
| `ros-N5-p3c1` | 5 | 3 | 1 |
| `ros-N6-p2c3` | 6 | 2 | 3 |
| `ros-N12-p8c3` | 12 | 8 | 3 |

`x` is redundant with `p + c + 1` and that is on purpose: the redundant number
is what a human reads first in a directory listing ("twelve processes"), and
the coordinate is what the analysis needs; a mismatch between them is a bug in
the scenario file that `bench/run.py` can check before launching anything.
There is one `p2c1` cell, not two: the seven-op chain is a constant of the
experiment, so the model matrix and the shape axis share `ros-N4-p2c1`.
The `chain` factor in `bench/scenarios.json` selects which command template
runs, so a shape is one extra entry per resolution, not a new harness, and the
shape string goes in the summary provenance.

**Tier 1 -- the decisive minimum** (10 cells, 5 shapes x 720p and 180p):
`monolith`, `ros-N2-p1c0`, `ros-N3-p2c0`, `ros-N3-p1c1`, `ros-N4-p2c1`. This
answers H1 (the `p2c0` vs `p1c1` pair, 720p alone) and H2 (adding 180p). Dropping
to 720p only -- 5 cells -- still answers H1, which is the cheaper first step if
the budget is tight.
Run `ros-N2-p1c0` first: it is the highest information per minute, because
everything the repo claims about "ROS overhead is transport" rests on the
assumption that a process boundary with a single hop costs what the model says.

**Tier 2 -- the long tail** (2 cells): `ros-N12-p8c3` at 720p and 180p. This
validates linearity in `k` at the extreme where the additive model is most
likely to break (DDS participants, executor wakeups, memory pressure from 12
processes). If `xport(p8c3) / xport(p2c1)` comes out well above the hop ratio it
should obey (11/3 at 180p, where every hop costs about the same; 88.4/23 at
720p), the model is wrong and the answer is *super*linear -- which is itself the
finding, and would mean the per-hop term is not a constant.

Both tiers must be interleaved with the anchor per pass (`bench/run.py` does
this already) -- the whole point is that the deltas are within-pass.

Budget: the ceiling is `settle + 1.1 * analysis_frames / fps_hint`. Slow shapes
do not cost fewer frames, they cost more seconds, so a pass containing
`ros-N12-p8c3` at 720p (~4.8 fps predicted, ~950 s ceiling) is ~5x the wall time
of the current `ros-N4-p2c1` cell. These cells all carry the same seven-op chain,
so the op cost is common and the deltas still isolate the partition -- but the
absolute per-frame cost is ~100 ms higher than an op-free run, and the ceilings
reflect that. `--dry-run` prints the ceilings per cell; the `fps_hint`s for new
cells start from the `p2c1` value scaled by predicted hop cost, and `analyze.py`
warns when a window came up short.

## What this axis cannot answer

* **Throughput and parallelism.** One frame in flight means the stages are
  strictly serialised: a split can only add latency. Splitting pays only when
  `in-flight > 1`, which is a separate factor with its own semantics (latency
  and fps decouple, and queueing latency re-enters the measurement). Deliberately
  out of scope here.
* **GPU contention** between inference and other processes: the inference node
  is the only GPU user in every shape, so the axis cannot see it.
* **"ROS 2 overhead" as an abstraction.** Only `p1c0` approximates that, and only
  for one hop with a 3 MB payload.

## Decisions to ratify

| # | decision |
| --- | --- |
| D1 | A shape is **data** (one `chain` string + a composition function), not a family of launch flags. |
| D2 | Stage 4-6 get the same stage-runner treatment as stage 2, including `post_ops` timing, because the candidate branch is unmeasurable otherwise. |
| D3 | Every shape must be **detection-identical**; enforced by a test, not by review. |
| D4 | A shape is named `p{p}c{c}` (the pixel-hop and candidate-hop counts, `N = p+c+1`); the full shape string goes in the provenance of every artefact. |
| D5 | `in-flight > 1` is a different experiment and is not smuggled into this one. |
| D6 | The anchor stays `monolith-720p-n`; shape deltas are within-pass against `ros-N4-p2c1`, never against a mean from another session. |

## First light (not a measurement)

One pass, 150-frame windows, 5 s settle, 180p, `n` -- run to prove the harness
drives the shapes and that the records mean what the report thinks they mean.
**Not quotable**: one pass, no interleaving, a settle far below the 30 s the
design uses, and windows that cover different parts of the clip (dets/frame
therefore differs between cells for reasons that have nothing to do with the
shape).

| cell | hops | xport ms | xport/hop | p50 ms |
| --- | --- | --- | --- | --- |
| `p1c0` | 1 | 2.47 | 2.47 | 39.76 |
| `p1c1` | 2 | 6.52 | 3.26 | 44.88 |
| `p2c0` | 2 | 6.53 | 3.27 | 47.14 |
| `p2c1` | 3 | 10.12 | 3.37 | 53.36 |
| `p8c3` | 11 | 34.50 | 3.14 | 76.98 |

The structure is what the model predicts: transport is linear in hop count and
the per-hop cost is flat across shapes, ~3.2 ms at 180p (the model said 3.9).
`p1c1` and `p2c0` -- same node count, different payload -- agree to 0.01 ms,
which is *not* the H1 prediction (they were expected 2.6 ms apart); at one pass
and 150 frames that difference is below the noise floor, so H1 remains the
thing the first real pass has to decide.

## Expected outcome (for the record, before measuring)

1. Two branches leave `p2c1` with very different slopes. Coarsening falls
   below it: `p2c0` (-3.6), `p1c1` (-9.7), `p1c0` (-13.3). Refining rises above
   it, and the side you refine on sets the rate: the candidate side `p2c2`
   (+3.6) and `p2c3` (+7.2), the pixel side `p3c1` (+9.7) and `p8c3` (+65.4).
   The tell for H1 is that `p2c0` and `p1c1` -- *same node count* -- sit
   6.1 ms apart at 720p, while `p2c2` -> `p2c3`, two extra processes, costs
   3.6 ms. In the label scheme this is not a surprise, it is arithmetic.
2. `xport` linear in `k` with a slope of ~9.6 ms/hop at 720p and ~3.9 at 180p,
   i.e. the fixed `a` term dominates at low resolution.
3. `p8c3` at 180p: transport alone exceeds the monolith's total latency by ~3.4x,
   and the cell runs at <15 fps while consuming more CPU per frame than any
   other cell in the matrix.
4. The most useful sentence this axis can produce is not a ranking but a rule:
   **co-locate the pixel-consuming stages; cut freely where candidates flow.**

If (1)-(3) come out as predicted, the axis is done and should not be re-run
every matrix; if a cell violates (1), the additive model needs a third term and
that is worth a session of its own.
