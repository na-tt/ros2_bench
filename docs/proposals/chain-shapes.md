# Chain shapes: how many processes, and where the cuts go

Status: **proposal**, nothing implemented. Branch `feature/preprocessing-ops`
already ships the mechanism for one axis (stage 2, `pre_ops` + `pre_split`);
this document is about generalising it to the whole pipeline and about what
each shape is allowed to claim.

Related: `preprocessing-chain.md` (why the model contract moved into the
inference node -- that decision is a precondition for the numbers here).

## The question, stated so it can fail

The existing axis is *resolution*. This proposal adds *chain length*, and the
point of it is not "which shape is fastest" -- with one frame in flight that
answer is known in advance and is boring (fewer hops, always). The point is:

> **What does a process boundary cost, where is it cheapest, and does that
> change with resolution?**

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

## The configurations

Notation: `|` separates processes, `+` means "these stages share one process".
Stage ids are the repo's own: `acq`, `pre` (the whole conditioning chain),
`infer`, `nms`, `filter`, `coord`, and `op1..op7` for the individual
conditioning ops.

| # | shape | N | pixel hops | candidate hops | what it isolates |
| --- | --- | --- | --- | --- | --- |
| M0 | `acq+pre+infer+nms+filter+coord` | 1 | 0 | 0 | monolith (existing anchor) |
| **R2** | `acq` \| `pre+infer+post` | 2 | 1 | 0 | **the process boundary alone** |
| **R3e** | `acq` \| `pre` \| `infer+post` | 3 | 2 | 0 | ...vs cutting *before* the model |
| **R3l** | `acq` \| `pre+infer` \| `post` | 3 | 1 | 1 | ...vs cutting *after* it |
| **R4** | `acq` \| `pre` \| `infer` \| `post` | 4 | 2 | 1 | current chain |
| R5 | `acq` \| `pre` \| `infer` \| `nms` \| `filter+coord` | 5 | 2 | 2 | cheap cut, one hop |
| R6 | `acq` \| `pre` \| `infer` \| `nms` \| `filter` \| `coord` | 6 | 2 | 3 | post branch exhausted |
| **R5m** | `acq` \| `op1+op2` \| `op3..op7` \| `infer` \| `post` | 5 | 3 | 1 | **one pixel hop more than R4** (one step up the left branch) |
| R12 | `acq` \| `op1`..`op7` \| `infer` \| `nms` \| `filter` \| `coord` | 12 | 8 | 3 | both branches exhausted |

`R3e` / `R3l` are the pair worth staring at: **the same number of processes,
a different cut, 6.1 ms apart at 720p** (19.4 vs 13.3 ms of transport) because
one puts the pixel payload on the wire *twice* and the other *once*. That is
the hypothesis in one line:

> **H1: cost is governed by the payload class of the boundary, not by the
> number of nodes.**

Two consequences, both falsifiable:

* `R3l` should beat `R3e` in p50 despite being indistinguishable on any
  process-oriented metric (both are 3 nodes, 4 stages, ~the same RSS). If they
  tie, the payload does not matter and the `beta` term is measuring something
  else.
* `R2` should be the cheapest ROS shape of all, since it carries the fewest
  hops (one pixel, no candidate). If it is not -- if a 2-node chain is slower
  than a 4-node one -- there is a fixed per-process cost that the `a` term has
  been hiding, and the transport model in the READMEs is missing a term.

### Predicted deltas (all vs R4, ms of p50)

From the per-hop model fitted on the five-message chain (`a=3.6`,
`beta=2.2 ms/MB`; the same fit that predicted the measured 9.6 / 5.0 / 3.9 ms
per hop at 720p / 360p / 180p):

| shape | 720p | 360p | 180p | because |
| --- | --- | --- | --- | --- |
| R2 | **-13.3** | -8.7 | **-7.6** | one pixel + one candidate hop fewer |
| R3l | **-9.7** | -5.1 | **-4.0** | one pixel hop fewer |
| R3e | -3.6 | -3.6 | -3.6 | one candidate hop fewer |
| **R5m** | **+9.7** | +5.1 | **+4.0** | one pixel hop more -- the additivity check |
| R5 | +3.6 | +3.6 | +3.6 | one candidate hop more |
| R6 | +7.2 | +7.2 | +7.2 | two candidate hops more |
| R12 | **+65.4** | +37.8 | **+31.2** | 6 pixel + 2 candidate hops more than R4 (11 hops) |

`R5m` is worth its own cell rather than a footnote: it is R4 plus exactly one
pixel hop and nothing else, so its delta should reproduce `a + beta*3A` within
noise. If it does not, additivity -- the assumption every other prediction on
this page rests on -- is false, and the k-sweep has to be read as a measured
`T(k)` curve rather than resolved into per-hop terms.

Read the last row against the monolith's whole p50: at 180p, R12's *transport
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
| ros-4node-720p-n | 24.2 ms | 42.4 ms | 57 % |
| ros-4node-180p-n | 18.4 ms | 36.2 ms | 51 % |

and it is why the earlier note "only ops > ~10 ms/frame justify a hop even at
720p" is really "at 180p *nothing* justifies a hop": the floor is 3.9 ms there,
above gamma (0.5), hist_eq (1.5), gaussian and median.

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
   groups; a group is a list of stage ids. `chain:=acq|pre|infer|post` is R4,
   `chain:=acq|pre+infer|post` is R3l, and so on. The existing
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
4. **The shape must be in the provenance.** `summary.provenance` (and the
   detections sidecar header, where `producer='ros-4node'` is currently
   hardcoded) must record the shape string. A records file from a 12-node run
   is otherwise indistinguishable from a 4-node one, which defeats the audit
   trail these commits exist for.
5. **`n_cand` as an optional record key** -- see the payload caveat above.

The ack does **not** need work: it stays one control hop from the terminal node
to acquisition regardless of `N`. The per-frame scheduling cost of `N` nodes is
inside the hop term, which is precisely what the sweep measures.

## Measurement design

Cells keep the existing `<chain>-<resolution>-<model>` form, `chain` naming the
shape: `monolith`, `ros-2node`, `ros-3node-pre`, `ros-3node-post`, `ros-4node`
(unchanged, for continuity with existing results), `ros-5node`,
`ros-5node-merged` (R5m: same count, cut on the left instead of the right),
`ros-6node`, `ros-12node`. The number in the name is `N`; the `chain` factor in
`bench/scenarios.json` selects which command template runs, so a shape is one
extra entry per resolution, not a new harness.

**Tier 1 -- the decisive minimum** (10 cells, 5 shapes x 720p and 180p):
`monolith`, `ros-2node`, `ros-3node-pre`, `ros-3node-post`, `ros-4node`. This
answers H1 (the `3node-pre` vs `3node-post` pair, 720p alone) and H2 (adding
180p). Dropping to 720p only -- 5 cells -- still answers H1, which is the
cheaper first step if the budget is tight.
Run `ros-2node` first: it is the highest information per minute, because
everything the repo claims about "ROS overhead is transport" rests on the
assumption that a process boundary with a single hop costs what the model says.

**Tier 2 -- the long tail** (2 cells): `ros-12node` at 720p and 180p. This
validates linearity in `k` at the extreme where the additive model is most
likely to break (DDS participants, executor wakeups, memory pressure from 12
processes). If `xport(R12) / xport(R4)` comes out well above the hop ratio it
should obey (11/3 at 180p, where every hop costs about the same; 88.4/23 at
720p), the model is wrong and the answer is *super*linear -- which is itself the
finding, and would mean the per-hop term is not a constant.

Both tiers must be interleaved with the anchor per pass (`bench/run.py` does
this already) -- the whole point is that the deltas are within-pass.

Budget: the ceiling is `settle + 1.1 * analysis_frames / fps_hint`. Slow shapes
do not cost fewer frames, they cost more seconds, so a pass containing
`ros-12node` at 720p (~7 fps predicted, ~430 s ceiling) is ~5x the wall time of
the current `ros-4node` cell. `--dry-run` prints the ceilings per cell; the
`fps_hint`s for new cells start from the R4 value scaled by predicted hop cost,
and `analyze.py` warns when a window came up short.

## What this axis cannot answer

* **Throughput and parallelism.** One frame in flight means the stages are
  strictly serialised: a split can only add latency. Splitting pays only when
  `in-flight > 1`, which is a separate factor with its own semantics (latency
  and fps decouple, and queueing latency re-enters the measurement). Deliberately
  out of scope here.
* **GPU contention** between inference and other processes: the inference node
  is the only GPU user in every shape, so the axis cannot see it.
* **"ROS 2 overhead" as an abstraction.** Only R2 approximates that, and only
  for one hop with a 3 MB payload.

## Decisions to ratify

| # | decision |
| --- | --- |
| D1 | A shape is **data** (one `chain` string + a composition function), not a family of launch flags. |
| D2 | Stage 4-6 get the same stage-runner treatment as stage 2, including `post_ops` timing, because the candidate branch is unmeasurable otherwise. |
| D3 | Every shape must be **detection-identical**; enforced by a test, not by review. |
| D4 | Cells are named `ros-<k>node[-pre\|-post]`; the full shape string goes in the provenance of every artefact. |
| D5 | `in-flight > 1` is a different experiment and is not smuggled into this one. |
| D6 | The anchor stays `monolith-720p-n`; shape deltas are within-pass against `ros-4node`, never against a mean from another session. |

## Expected outcome (for the record, before measuring)

1. Two branches leave `R4` with very different slopes. Coarsening falls below
   it: `R3e` (-3.6), `R3l` (-9.7), `R2` (-13.3). Refining rises above it, and
   the side you refine on sets the rate: the candidate side `R5` (+3.6) and
   `R6` (+7.2), the pixel side `R5m` (+9.7) and `R12` (+65.4). The tell for H1
   is that `R3l` and `R3e` -- *same node count* -- sit 6.1 ms apart at 720p,
   while `R5` -> `R6`, two extra processes, costs 3.6 ms.
2. `xport` linear in `k` with a slope of ~9.6 ms/hop at 720p and ~3.9 at 180p,
   i.e. the fixed `a` term dominates at low resolution.
3. R12 at 180p: transport alone exceeds the monolith's total latency by ~3.4x,
   and the cell runs at <15 fps while consuming more CPU per frame than any
   other cell in the matrix.
4. The most useful sentence this axis can produce is not a ranking but a rule:
   **co-locate the pixel-consuming stages; cut freely where candidates flow.**

If (1)-(3) come out as predicted, the axis is done and should not be re-run
every matrix; if a cell violates (1), the additive model needs a third term and
that is worth a session of its own.
