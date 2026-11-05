# Method — what the numbers in the report mean

`README.md` describes the harness: files, order, budgets. This file is about the
**estimators** — what each number is computed *over*, and which comparisons it
does and does not license. Nothing here is exotic statistics; it is the small
amount of design needed to make a latency difference on an unpinned board mean
something. Read it once before quoting a number from a report.

Every number used as a worked example below is from `results/bench/` of the
first full matrix run (3 passes, window = `settle_s 30` + `analysis_frames 2500`,
anchor `monolith-720p-n`), so you can recompute all of it with `analyze.py` from
the same directory. That run predates the conditioning chain becoming a constant
of the experiment: its cells ran stage 2 empty and the ROS ones are filed under
the then-current name `ros-4node-*` (now `ros-N4-p2c1-*`). The artefacts are
still on disk under those names, so the examples keep them; a re-run under the
current scenario is a *different workload* (the seven ops cost ~103.5 ms/frame at
720p) and its numbers are not comparable to these. The other exception is the
A/A pair in *Level 3*, which comes from the retired `monolith-ctrl` cell of an
earlier session and is labelled as such -- it is kept because a duplicate cell is
the clearest way to show a delta that is *not* a difference.

## Words

| word | meaning |
|---|---|
| **cell** | one point of the matrix — `monolith-720p-n`. A "test condition". Called a cell because `scenarios.json` is a grid: factors × design. |
| **trial** | one *execution* of a cell → one directory, `results/bench/<cell>/r3/`. |
| **pass** | one round in which every cell of the run runs once. `--repeats 3` = 3 passes; trial `r3` belongs to pass 3. |
| **slot** | a cell's position inside a pass. Only used by the order audit (`run.json`). |
| **block** | a group of cells run back-to-back (both chains at one resolution/model), so the pair is minutes apart instead of hours. |
| **anchor** | the reference cell every delta is expressed against (`anchor` in `scenarios.json`). Every cell runs once per pass, so the anchor is *not* measured more often — it is the shared zero, which is why any run of a subset must include it. |
| **window** | the frames a trial is summarised over: everything after that trial's settle (30 s, 45 s for a windowed cell — read back from its `run.meta.json`), capped at `analysis_frames` frames. |
| **dispersion** | how much a single trial's value moves between passes (std). |
| **CI** | how well the *mean* is known. Different object, see below. |
| **Δ (delta)** | a difference between two cells, computed *inside* one pass. |
| **pseudoreplication** | treating frames as independent replicates. They are not. |

## The shape of a run

```
pass 1:  block(720p,n)…  [monolith-720p-n] [ros-4node-720p-n]   block(720p,m)…
pass 2:  blocks shuffled with a logged seed, direction reversed inside each
pass 3:  …
```

Each trial has the same frame budget (see *Trial budget* in the README), so every
cell contributes the same number of frames to its percentiles regardless of how
fast it is. What varies deliberately is the *order*, so the machine state a cell
inherits is randomised rather than fixed.

## Level 1 — inside a trial: frames → one number per metric

`summarise_trial()` keeps the window's frames and computes percentiles over
their `latency_ms`. One trial therefore contributes exactly one `p50`, one `p95`,
one `fps` — not 2500.

Three ROS-only quantities are attached to the same window, and each one is a
*median over the window's frames* like the stage buckets: `inflight` (realized
concurrency at publish, aggregated into the report as `fill` = realized /
requested), `hop_ms` (the measured wait in front of each node) and `drops` (a
count of `seq` gaps, which must be 0 -- a gap means a topic history was
shallower than the window and evicted an unacked sample). `L·X/i` is not a
per-frame quantity at all: it is Little's law evaluated on the window
(`p50 · fps / 1000 / i`), and it is there to catch a window that is configured
but never resident.

The 2500 frames are **not** independent observations: consecutive frames share a
thermal state, a GC epoch and a cache state. Pooling them as if they were gives
absurdly tight error bars and false claims. On this cell the three trials agree on
p50 within ±0.2 ms, while the per-frame std of the 7500 pooled frames is 3.7 ms —
that 3.7 ms is the frame-to-frame spread of *this* distribution, not the
uncertainty of the estimate.

## Level 2 — across passes: mean ± std

The unit of replication is the trial, so the headline is computed over trials:

| pass | frames | p50 | p95 | fps |
|---|---|---|---|---|
| r1 | 2500 | 12.79 | 24.29 | 58.21 |
| r2 | 2500 | 12.44 | 24.62 | 60.33 |
| r3 | 2500 | 12.49 | 24.37 | 60.08 |
| **mean ± std** | | **12.57 ± 0.19** | 24.43 ± 0.17 | 59.54 ± 1.16 |

`12.57` is `fmean(12.79, 12.44, 12.49)`, `0.19` is `stdev(...)` with ddof = 1. The
statement it supports: *run this cell again and its p50 lands within about
±0.2 ms of 12.6.* The `passes` column of the report is this `n`; the `frames`
column is the median window length of the passes, a per-trial quantity.

(A third pass is enough here because this session was unusually quiet: the
dispersion is 1.5 %, i.e. better than the ~3 % and ~8 % the README quotes from
other runs. Never carry that confidence into a different session.)

A std says nothing about whether the *mean* moved. That is the next problem.

## Level 3 — the delta: subtract inside a pass

The board drifts ~8 % over a long session, in either direction: thermal state, the
desktop, the thing the user did an hour ago. A difference between two cells
measured hours apart is therefore part pipeline and part machine state, and the
std of either cell cannot separate the two.

Two trials from the **same pass** shared most of that state, so:

```
Δ(pass k) = cell(pass k) − anchor(pass k)
```

One Δ per pass, then mean and CI over *those*. Two examples from the same run.

A difference that is **not** one — 720p vs 180p, the same network on the same
frames at a different decode size:

| pass | monolith-720p-n | monolith-180p-n | Δ |
|---|---|---|---|
| r1 | 12.79 | 12.78 | −0.01 |
| r2 | 12.44 | 12.67 | +0.23 |
| r3 | 12.49 | 12.69 | +0.21 |
| | | | **mean +0.14 ± 0.33** (no `*`) |

The three deltas do not even agree on the sign, so the honest statement is
"resolution does not change p50", not "180p is 0.14 ms slower". Note that this is
*not* a null result for the axis: the same two cells differ by `−11.26 ± 0.52\*`
in CPU per frame and `+10.77 ± 3.12\*` in throughput. The axis moves something; it
just does not move latency.

A difference that **is** one — the four-node chain at the same 720p/n point:

| pass | monolith-720p-n | ros-4node-720p-n | Δ |
|---|---|---|---|
| r1 | 12.79 | 42.49 | +29.70 |
| r2 | 12.44 | 43.29 | +30.85 |
| r3 | 12.49 | 41.42 | +28.94 |
| | | | **mean +29.83 ± 2.39\*** |

Here the CI is nowhere near zero, so the sign is stable and the claim is safe:
3.4x the latency, missed on every single pass.

The clearest illustration that a narrow CI is not the same as a difference: the
retired A/A pair (a byte-identical clone, so the true Δ is 0) gave
−1.33 / +0.40 / +0.20 — **mean −0.24 ± 2.35**, three passes failing to establish
even the sign of a difference that does not exist. That is the noise floor as a
number: at n = 3, per-pass wobble of ±1.3 ms is indistinguishable from nothing.

Two honest caveats. Pairing cancels the *slow* trend (hours), not the wobble
within a pass (minutes) — that A/A pair ran as a block and still differed by
1.33 ms on r1, which is exactly why the noise floor is that wide. And pairing is
not automatically tighter: on that data the unpaired CI would have been ±1.69 ms,
narrower than the paired ±2.35, because the outlier happened to sit *inside* a
pass. Pairing removes the drift common to the pair; it does not remove
fluctuation, and it cannot help when the pair is far apart in time.

A `*` in the report means exactly that: the CI excludes zero.

## Three uncertainties, do not mix them

All computed on the same three numbers (`12.79, 12.44, 12.49`):

| statement | formula | value | says |
|---|---|---|---|
| **std** (dispersion) | `stdev(x)` | 0.19 ms | one trial's p50 lands ±0.2 from the mean |
| **SE** (standard error) | `std/√n` | 0.11 ms | how much the *mean* wobbles |
| **CI95** | `t₀.₉₇₅,n−1 · SE` | **±0.47 ms** | 95 % of such experiments would bracket the truth here |

The headline table prints the **std** — that is what a dispersion column is for,
and the legend says as much. The delta table prints the **CI**, because that is
the number that decides whether a difference exists.

The Student t factor is why small n looks so bad: 12.71 at n = 2, 4.30 at n = 3,
3.18 at n = 4, 2.57 at n = 5, 2.26 at n = 10, 2.09 at n = 20. At n = 3 the
std is also only itself good to ~40 %, which is why the report prints the raw
per-trial values next to the summary and why `AGENTS.md` says to read those when
n < 6.

## How many passes a claim needs

With σ_Δ ≈ 0.96 ms -- the spread of the ros-vs-monolith deltas above, which is
the *pessimistic* case because those two cells are the far apart ones --
`n ≈ (t·σ/Δ)²`:

| want to resolve | passes | CI you actually get |
|---|---|---|
| ±2.4 ms | 3 | 4.30 · 0.96/√3 = 2.38 |
| ±1.5 ms | 4 | 3.18 · 0.96/2 = 1.53 |
| ±1.0 ms | 6 | 2.57 · 0.96/√6 = 1.01 |
| ±0.5 ms | 17 | 2.12 · 0.96/√17 = 0.49 |

For the neighbouring cells in the same block (720p vs 180p, σ_Δ = 0.13 ms) three
passes already resolve ±0.32 ms -- the pairing quality, not the pass count, is what
sets the resolution, and cells that run minutes apart pair better than cells that
run hours apart.

Cells are cheap to add in *analysis* (`--anchor`, `--settle`, `--frames` are all
re-derived from records on disk); resolution is bought with passes. That is the
argument for pruning the matrix rather than running all 18 cells: passes buy
resolution, cells do not.

## Four ways to read a number wrongly

1. **Pooling frames across trials.** Gives n = 7500 instead of 3 and an error bar
   up to √(7500/3) ≈ 50× too small, so every difference becomes "significant".
   The trial is what was randomised. (Cross-check: the pooled median is 12.59 ms,
   within 0.02 ms of the mean of the per-trial medians -- pooling the *location*
   is harmless, pooling the *spread* is not.)
2. **Comparing means from different hours.** The ~8 % drift is larger than most
   effects, and the desktop is worse than drift: the same design measured a 27 %
   board load on a quiet desk and 95 % with a browser open, with the pipeline
   1.4x slower at the higher load and identical code. Use the delta table, or
   don't compare -- and check the `load %` column first, which is why it is
   printed.
3. **Quoting the std as a CI** (or the reverse). `12.57 ± 0.19` and
   `12.57 ± 0.47` are both true and answer different questions.
4. **Reading the stage table as latency.** Acquisition is excluded from
   `latency_ms` on purpose (it is the decoder, ~28 % of CPU time), and in a ROS
   cell most of the frame is `xport` — serialisation, DDS and scheduling. A p50
   claim without the stage split will attribute transport to the pipeline.

## Reading the report, in order

1. **`passes` column** — if it is 1 there is no std and no delta; that run is a
   smoke test, not a measurement.
2. **Per-trial detail table** — is one pass an outlier? Tight agreement makes the
   summary trustworthy; one far-off value means the mean is lying and the median
   would not.
3. **Δ table vs the anchor** — the only place a sentence like "X is faster than
   Y" may come from, and only on rows carrying a `*`.
4. **Stage table** — before believing a Δ, check *which* stage moved.

## Where this lives in the code

| concept | code |
|---|---|
| window selection | `analyze.py: window_indices()` — settle, then frames |
| per-trial percentiles | `analyze.py: summarise_trial()` |
| mean ± std across passes | `analyze.py: _stats()` (docstring: "never across frames") |
| CI | `analyze.py: _ci95()`, table `_T95` |
| deltas | `analyze.py: paired_deltas()`, matched on the repeat index |
| order / anchor | `run.py: build_schedule()`, `scenarios.json: order`, `anchor` |
