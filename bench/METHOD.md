# Method — what the numbers in the report mean

`README.md` describes the harness: files, order, budgets. This file is about the
**estimators** — what each number is computed *over*, and which comparisons it
does and does not license. Nothing here is exotic statistics; it is the small
amount of design needed to make a latency difference on an unpinned board mean
something. Read it once before quoting a number from a report.

Numbers used as worked examples come from `results/bench/` (3 passes,
`monolith` / `monolith-ctrl` / `ros-4node`, analysed with `--settle 30
--frames 2500`). They are illustrations of the arithmetic, not current results.

## Words

| word | meaning |
|---|---|
| **cell** | one point of the matrix — `monolith-720p-n`. A "test condition". Called a cell because `scenarios.json` is a grid: factors × design. |
| **trial** | one *execution* of a cell → one directory, `results/bench/<cell>/r3/`. |
| **pass** | one round in which every cell of the run runs once. `--repeats 3` = 3 passes; trial `r3` belongs to pass 3. |
| **slot** | a cell's position inside a pass. Only used by the order audit (`run.json`). |
| **block** | a group of cells run back-to-back (both chains at one resolution/model), so the pair is minutes apart instead of hours. |
| **anchor** | the reference cell every delta is expressed against (`anchor` in `scenarios.json`). Every cell runs once per pass, so the anchor is *not* measured more often — it is the shared zero, which is why any run of a subset must include it. |
| **window** | the frames a trial is summarised over: everything after `settle_s` seconds, capped at `analysis_frames` frames. |
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

The 2500 frames are **not** independent observations: consecutive frames share a
thermal state, a GC epoch and a cache state. Pooling them as if they were gives
absurdly tight error bars and false claims. On the example data the three trials
agree on p50 within ±0.5 ms, while the per-frame std is 6.4 ms — that 6.4 ms is
the frame-to-frame spread of *this* distribution, not the uncertainty of the
estimate.

## Level 2 — across passes: mean ± std

The unit of replication is the trial, so the headline is computed over trials:

| pass | frames | p50 | p95 | fps |
|---|---|---|---|---|
| r1 | 2500 | 16.30 | 33.06 | 35.06 |
| r2 | 2500 | 15.29 | 31.75 | 36.25 |
| r3 | 2500 | 15.55 | 32.59 | 35.86 |
| **mean ± std** | | **15.72 ± 0.52** | 32.47 ± 0.66 | 35.72 ± 0.61 |

`15.72` is `fmean(16.30, 15.29, 15.55)`, `0.52` is `stdev(...)` with ddof = 1. The
statement it supports: *run this cell again and its p50 lands within about
±0.5 ms of 15.7.* The `passes` column of the report is this `n`; the `frames`
column is the median window length of the passes, a per-trial quantity.

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

One Δ per pass, then mean and CI over *those*. Worked example, an A/A pair (a
byte-identical clone measured against the anchor — the true Δ is 0):

| pass | monolith | monolith-ctrl | Δ |
|---|---|---|---|
| r1 | 16.30 | 14.97 | −1.33 |
| r2 | 15.29 | 15.69 | +0.40 |
| r3 | 15.55 | 15.75 | +0.20 |
| | | | **mean −0.24 ± 2.35** |

Read it as: three passes failed to establish even the *sign* of a difference that
does not exist. That is the noise floor, as a number: **at n = 3 a real p50
difference of ±1.3 ms per pass is indistinguishable from nothing.**

Two honest caveats. Pairing cancels the *slow* trend (hours), not the wobble
within a pass (minutes) — r1 above still differs by 1.33 ms for two runs of the
same code. And pairing is not automatically tighter: on this data the unpaired
CI would have been ±1.69 ms, narrower than the paired ±2.35, because the outlier
happened to sit *inside* a pass. Pairing removes the drift that is common to the
pair; it does not remove fluctuation, and it cannot help when the pair is far
apart in time.

Contrast: `ros-4node` vs the same anchor gives **+69.01 ± 2.85\*** — the CI never
comes near zero, so the sign is stable and the claim is safe.

A `*` in the report means exactly that: the CI excludes zero.

## Three uncertainties, do not mix them

All computed on the same three numbers (`16.30, 15.29, 15.55`):

| statement | formula | value | says |
|---|---|---|---|
| **std** (dispersion) | `stdev(x)` | 0.52 ms | one trial's p50 lands ±0.5 from the mean |
| **SE** (standard error) | `std/√n` | 0.30 ms | how much the *mean* wobbles |
| **CI95** | `t₀.₉₇₅,n−1 · SE` | **±1.30 ms** | 95 % of such experiments would bracket the truth here |

The headline table prints the **std** — that is what a dispersion column is for,
and the legend says as much. The delta table prints the **CI**, because that is
the number that decides whether a difference exists.

The Student t factor is why small n looks so bad: 12.71 at n = 2, 4.30 at n = 3,
3.18 at n = 4, 2.57 at n = 5, 2.26 at n = 10, 2.09 at n = 20. At n = 3 the
std is also only itself good to ~40 %, which is why the report prints the raw
per-trial values next to the summary and why `AGENTS.md` says to read those when
n < 6.

## How many passes a claim needs

With σ_Δ ≈ 0.95 ms (the A/A above), `n ≈ (t·σ/Δ)²`:

| want to resolve | passes | CI you actually get |
|---|---|---|
| ±2.4 ms | 3 | 4.30 · 0.95/√3 = 2.36 |
| ±1.5 ms | 4 | 3.18 · 0.95/2 = 1.51 |
| ±1.0 ms | 6 | 2.57 · 0.95/√6 = 1.00 |
| ±0.5 ms | 17 | 2.12 · 0.95/√17 = 0.49 |

Cells are cheap to add in *analysis* (`--anchor`, `--settle`, `--frames` are all
re-derived from records on disk); resolution is bought with passes. That is the
argument for pruning the matrix rather than running all 18 cells: passes buy
resolution, cells do not.

## Four ways to read a number wrongly

1. **Pooling frames across trials.** Gives n = 7500 instead of 3 and an error bar
   up to √(7500/3) ≈ 50× too small, so every difference becomes "significant".
   The trial is what was randomised.
2. **Comparing means from different hours.** The ±8 % drift is larger than most
   effects. Use the delta table, or don't compare.
3. **Quoting the std as a CI** (or the reverse). `15.72 ± 0.52` and
   `15.72 ± 1.30` are both true and answer different questions.
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
