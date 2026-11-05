# bench — measurement harness

Out-of-process tooling for comparing the single-process pipeline with the ROS 2
implementations. It lives beside `monolith/` and `ws/`, **imports neither**, and
talks to them only through a command template and a file contract. That keeps it
outside the "no cross-project imports" rule and makes the monolith and a ROS
launch file configurable in exactly the same way.

```
bench/
├── run.py          # orchestrator: interleaved trials + sampler + artefacts (+ --demo)
├── sampler.py      # board (tegrastats) + workload (/proc) telemetry -> CSV
├── analyze.py      # trials -> markdown/JSON report, window chosen here
├── scenarios.json  # factors x design -> cells, + an `artifacts` (demo) block
└── METHOD.md       # what the reported numbers are computed over, and why
```

New to the report? Read `METHOD.md` first: it defines cell / trial / pass /
anchor, and explains why a claim is a per-pass delta with a CI rather than a
difference of two means.

## Policy (deliberate, not accidental)

* **No clock pinning, no governor changes, no GC tuning, no headless mode.**
  The desktop session's background activity is part of the measurement — it is
  the noise floor. Anything that tries to make the machine quiet makes the
  result less representative of the deployment.
* **No warmup stage.** The workload records every frame; startup and settling
  show up in the record stream and the analysis window is chosen afterwards.
* **Interleaved trials.** Clocks are unpinned, so a run drifts by several
  percent as the machine warms (measured: ~8 % between the first and third
  replay of a 120 s run, in *either* direction). Running all of A and then all
  of B would attribute that drift to B. `run.py` orders trials so no cell is
  systematically first (see *Trial order*).
* **An anchor cell.** Every cell runs once per pass, so a pass yields one trial
  per cell -- the anchor is not measured more often than the rest. It is the
  *reference* cell: every delta in the report is expressed against it, so a
  comparison is a per-pass delta rather than two means measured hours apart, and
  every run of a subset must include it so that numbers from different subsets
  and sessions share one zero. The duplicate control cell that used to serve this
  purpose is now only a demo artifact — see *Calibration*.

## Trial order

`order` in `scenarios.json` decides who runs when, and the audit of that decision
is written to `run/<out>/run.json`.

* **`policy: auto`** keeps the cyclic Latin square while it is cheap and correct
  (`<= 6` cells *and* `repeats` a multiple of the cell count). It balances every
  slot position exactly — but it needs `repeats % n == 0` (with 2 cells and 3
  passes, A is first twice) and it gives every cell a fixed successor, so
  adjacency is never balanced.
* **Otherwise** cells are grouped into blocks and the block order is shuffled per
  pass with a logged seed, while the direction inside a block alternates
  (boustrophedon). `block_by: ["resolution", "model"]` puts a monolith and a ROS
  cell at the same resolution and model next to each other, so the chain delta is
  a difference measured minutes apart; across passes each block lands at a
  different time of day. Blocks are grouped *by factor*, not by cell name, so
  editing `design` cannot desynchronise them.

The point of shuffling rather than rotating: a deterministic order that is not
balanced turns drift into a *bias* the report cannot see, while a randomised one
turns it into *variance* the std does see. `run.json` records the schedule, the
slot-position and predecessor histograms per cell, and any warning (a cell that
always ran after the same cell, a pass that was not a full pass).

## Running a measurement

### 0. Get the project shell (mandatory, every time)

`run.py` launches each condition with `{python}` = **its own** interpreter and
inherits **its own** environment, so starting it from fish (Homebrew `python3`,
no ROS setup) silently measures the wrong thing. Always come in through bash
with a clean `PATH` (`AGENTS.md`, Step 0):

```bash
env -i HOME="$HOME" TERM=dumb \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash -ic 'cd ~/Memoria/ros2_bench && <command>'
```

Inside this repo, from the repository root, that is the only form any command
below needs.

### 1. Prerequisites (once per session, or after touching `ws/`)

```bash
# ROS 2 wrapper packages; --symlink-install so Python edits need no rebuild
(cd ws && colcon build --symlink-install)
# engine + clip; there is no pip on the Jetson's /usr/bin/python3, so the
# conditions pass PYTHONPATH=monolith/yolocar/src instead of installing
ls assets/models/yolov8n_fp16.engine assets/video/0000f77c-6257be58.mov
```

**Rotation belongs to the footage, not to the experiment.** The original clip is
stored upside down (`CAP_PROP_ORIENTATION_META` reports 270, which is wrong) and
needs `--rotate 180`. The re-encoded `_720p`/`_360p`/`_180p` files had the
rotation baked into the pixels by ffmpeg, which also cleared the display matrix,
so they need `rotate: 0`. A condition may override the scenario-level `rotate`,
and the ROS launch defaults to 0. Getting this wrong does not fail a run: it
silently measures inference on an upside-down road scene, and only `n_det` and the
demo video look wrong.

### 2. Smoke test first (~40 s)

Cheap and it catches a moved engine, a stale workspace or a broken topic before
you spend an hour:

```bash
bench/run.py --repeats 1 --gap 2 --out results/smoke \
  --settle 3 --analysis-frames 200 \
  --conditions monolith-720p-n,ros-N4-p2c1-720p-n
bench/analyze.py --dir results/smoke
rm -rf results/smoke          # gitignored, but do not let it be read as data
```

Expect `2/2 trials ok`; if a trial exits non-zero, read
`results/smoke/<condition>/r1/workload.log`. Both cells now run the seven-op
conditioning chain, so a smoke cell is ~120-150 ms/frame at 720p, not ~15.

### 3. The full run (the headline pair, 3 passes ≈ 1 h)

```bash
bench/run.py --repeats 3 --gap 5 --out results/bench \
  --conditions monolith-720p-n,ros-N4-p2c1-720p-n
```

A matrix run is `--conditions` on a subset, not always the whole product; see
*The matrix* for the axes and pruning recipes. Wall clock per trial is the cell's
computed time ceiling (*Trial budget*) plus engine load (~1.5 s for `yolov8n`,
tens of seconds for `yolov8x`) plus `--gap`. The conditioning chain dominates a
720p cell (~103.5 ms/frame of ops), so the headline pair runs at roughly 8 and
6.5 fps against 59.5 and 20.1 before it.
Re-running into the same `--out` **overwrites** `<out>/<condition>/r<N>/`, so
give each session its own directory if you want to keep the old numbers
(`--out results/bench-$(date +%m%d-%H%M)`).

Long runs: detach it and poll, so a dropped terminal does not kill the trial:

```bash
nohup env -i HOME="$HOME" TERM=dumb \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash -ic 'cd ~/Memoria/ros2_bench && python3 bench/run.py --repeats 3 --gap 5 \
    --out results/bench \
    --conditions monolith-720p-n,ros-N4-p2c1-720p-n' > results/full-run.log 2>&1 &
tail -f results/full-run.log
```

### 4. Analyse (window chosen here, not during the run)

```bash
bench/analyze.py --md results/report.md --json results/report.json
bench/analyze.py --timeline-out results/report/timeline
```

The window defaults come from the scenario's `trial` block: drop `settle_s`
seconds after the **first recorded frame** (startup and the thermal ramp), then
take `analysis_frames` frames. So every cell reports percentiles over the same
`n` while the settling skip stays in the physics' unit. Startup and settling stay
in the recorded data, so a different window is a re-analysis, not a re-measure:
`--settle S`, `--frames N`, or the legacy `--from/--to` (seconds after the first
frame, no frame cap). A cell that came up short prints a warning naming the
likely cause (an over-optimistic `fps_hint`). `--dir` points at another artefact
root.

`--timeline-out DIR` writes one figure per condition (`timeline-<condition>.png`)
from that condition's first usable trial: latency stacked by stage, acquisition
as a separate dashed line (it is excluded from latency), replay boundaries as
thin verticals, and the analysis window shaded. It is the figure the table
cannot replace — per-frame jitter, the stage split, and (for ROS) how much of
the frame is transport. `--demo-dir` (default `results/demo`) appends a list of
rendered demo videos to the report when any exist.

### Options worth knowing

| flag | applies to | effect |
|---|---|---|
| `--conditions a,b` | `run.py` | subset by cell name (`monolith-720p-n`, ...) |
| `--design scenario\|full` | `run.py` | which cells exist: the pruned `design.cells` (default) or every combination (297 cells, 891 trials) |
| `--repeats N` | `run.py` | passes over the selected cells (see *Trial order*) |
| `--settle S` | `run.py` | seconds per trial treated as settling (default from `trial.settle_s`) |
| `--analysis-frames N` | `run.py` | frames of analysis window per trial (default from `trial.analysis_frames`) |
| `--duration S` | `run.py` | override the computed time ceiling for every cell (quick looks only) |
| `--gap S` | `run.py` | idle seconds between trials (default 5) |
| `--out DIR` | `run.py` | artefact root (default `results/bench`) |
| `--interval-ms N` | `run.py` | sampler period (default 100) |
| `--demo` | `run.py` | run the `artifacts` block instead of trials (see *Demo material*) |
| `--dry-run` | `run.py` | print the expanded commands and exit |
| `--scenario FILE` | `run.py` | alternate condition set |
| `--settle` / `--frames` | `analyze.py` | window: settling seconds, then a frame count (defaults from `trial`) |
| `--from` / `--to` | `analyze.py` | legacy absolute window in seconds, no frame cap |
| `--dir` / `--records` / `--sampler` | `analyze.py` | where to read trials from |
| `--timeline-out DIR` | `analyze.py` | one stage-stacked latency figure per condition |
| `--demo-dir DIR` | `analyze.py` | list rendered demo videos in the report |

`--loops` lives in `scenarios.json`, not on the CLI: it is what lets a 40 s clip
drive a 180 s trial.

### Demo material (`bench/run.py --demo`)

Separate mode, separate invocation, never inside a measured trial. For each
condition in the scenario's `artifacts.conditions` it runs **one full replay** of
the clip with `--detections` on (no sampler, no repeats), then renders the video
with `yolocar.demo`. `artifacts.compare` then builds the side-by-side video from
the per-condition sidecars.

```bash
bench/run.py --demo                                   # all artifact conditions
bench/run.py --demo --conditions monolith,ros-N4-p2c1 # just the headline pair
bench/run.py --demo --dry-run                         # print the steps
```

It cannot damage a measurement: the measured commands above are untouched, the
run length/shape is different by design (one replay, `--loops 1`, no `--duration`),
and a failure only marks `results/demo/run.json`.

```
results/demo/
├── run.json                                   # demo steps + return codes
├── <condition>/{records,detections,summary}.json[ l]
├── <condition>.mp4
└── compare-monolith-vs-ros-N4-p2c1.mp4       # --align wall: each panel at its own pace
```

`detections.jsonl` is **not** part of the benchmark contract and is documented in
`monolith/yolocar/README.md` (*Seeing a run*). A front-end that never writes one
is still compliant; `records.jsonl`/`summary.json` are unchanged by its presence.

Wall clock for a full `--demo` pass on this board: ~15 min (monolith ~2 min run
+ 1 min render, control likewise, ROS chain ~3 min run + 3 min render, compare
~2 min). The seven-op chain is inside every demo run as well, so the panels look
like the measured workload.

Common variants:

```bash
bench/run.py --dry-run                                  # what would be launched
bench/run.py --conditions monolith-720p-n,ros-N4-p2c1-720p-n --repeats 4  # headline pair
bench/run.py --repeats 1 --settle 5 --analysis-frames 500  # quick look
```

### 5. Running one front-end by hand (no harness, no sampler)

Useful while editing a node. These write `records.jsonl` / `summary.json` but no
`sampler.csv`, so `cpu ms/f`, `cores`, `pss MB`, `gpu %`, `W`, `J/f` and `tj °C`
are missing from the report:

```bash
# monolith — run from the repository root (assets paths are relative)
PYTHONPATH=monolith/yolocar/src python3 -m yolocar.main \
  --source assets/video/0000f77c-6257be58.mov --rotate 180 --duration 60 \
  --records results/manual/records.jsonl --json results/manual/summary.json

# ROS 2 four-node chain (workspace already built + sourced in ~/.bashrc)
PYTHONPATH=monolith/yolocar/src:$PYTHONPATH \
  ros2 launch yolocar_ros bench.launch.py duration:=60 \
    records:=results/manual/ros-records.jsonl summary:=results/manual/ros-summary.json
```

`--rotate 180` is not optional: the clip is stored upside down and its container
orientation flag (270) is wrong. `duration:=-1` in the launch file runs the
source to its end.

## Artefacts

```
results/bench/
├── run.json            # scenario, order (schedule + balance audit), per-cell budgets
└── <condition>/r<repeat>/
    ├── records.jsonl     # one JSON object per frame (the contract)
    ├── summary.json      # workload summary (latency, stages, provenance)
    ├── sampler.csv       # board + workload telemetry, same monotonic clock
    ├── sampler.meta.json
    ├── workload.log
    └── run.meta.json     # command, exit code, machine state before/after
```

The per-frame contract is the piece every front-end must satisfy:

```json
{"seq": 0, "loop": 0, "t_mono": 12345.678,
 "acq_ms": 9.1, "pre_ms": 4.2, "infer_ms": 6.0, "post_ms": 3.1,
 "present_ms": 0.0, "latency_ms": 13.3, "n_det": 5}
```

`t_mono` is `time.monotonic()`, which is system-wide on Linux, so the sampler's
`t_mono` column joins directly against it.

Two keys are **optional** and appear only when the run had something to put in
them, so a record written before they existed still analyses:

* **`transport_ms`** -- ROS only: end-to-end latency minus the stage timings,
  i.e. serialisation + DDS + scheduling. The monolith has no such cost, so its
  absence there is not a missing measurement.
* **`pre_ops`** -- `{"clahe": 4.1, "bilateral": 38.2}`: the stage-2 breakdown,
  one entry per conditioning op. `pre_ms` remains the per-frame *sum*, so the
  stage bucket means the same thing with a chain, with one op, or with none.
  Both front-ends emit it, including when the chain is split across several
  nodes (`analyze.py` prints a per-op table for cells that have it).
* **`post_ops`** -- `{"nms": 0.5, "filter": 0.2, "coord": 0.2}`: the same thing
  for stages 4-6, which `post_stages` can split over several nodes. Unlike
  `pre_ops` it is present whenever the ROS front-end ran the stage, because the
  stage always runs: the unsplit breakdown is the control for the split one.
* **`inflight`** -- ROS only: how many frames were outstanding (published and not
  yet acknowledged) when this one was published. It is the *realized* window, so
  a cell that never filled its window cannot be mistaken for one that filled it
  and bought nothing. `analyze.py` reports it as `fill` (realized/requested).
* **`hop_ms`** -- ROS only: the wait in front of every busy node, accumulated
  from each publisher's `t_pub` stamp at the next node's callback entry. At
  `inflight=1` it is transport and nothing else (the whole residual is delivery),
  which is what lets the growth with the window be read as queueing. It closes
  the decomposition: `latency_ms - acq_ms - Σ stage_ms - hop_ms` is conversion
  overhead and should stay under a millisecond.

`summary.json` carries two extra provenance fields in the ROS front-end:
`provenance.shape` is the `chain` string that built the graph (`acq|pre|infer|post`),
and it is also the `producer` of the detections sidecar; `provenance.inflight` is
the frames-in-flight window. Two runs of the same cells at different partition
depths -- or at different windows -- would otherwise be indistinguishable on
disk: identical stage buckets, different transport, different residency.

Stage attribution is the same in both front-ends: `pre_ms` is conditioning at
source resolution, `infer_ms` is the model input contract (letterbox + tensor)
*plus* the engine and the detect-head decode/threshold. The last part is why
the ROS chain's `infer_ms` must be compared as `infer + post` against the
monolith's separate buckets -- but `pre_ms` is directly comparable, because it
now measures the same thing on both sides.

## Reading the report

Three things, in this order.

1. **The headline table is mean ± sample std across passes**, with `n` in its
   second column. That std is a *dispersion, not a confidence interval*: with
   n = 3-4 it is itself only good to ~40 %, which is why the raw per-trial values
   are printed underneath rather than trusted to a summary.
2. **Deltas vs the anchor are what a claim is made of.** Two cells measured hours
   apart are not comparable -- the board drifts ~8 % across a long session -- so
   `analyze.py` differences every cell against the anchor *inside each pass* and
   reports the mean delta with a 95 % CI over paired passes. A `*` means the CI
   excludes zero. This is also why every subset run must include the anchor: the
   deltas only add up across subsets and sessions if they share the same zero.
3. **`load %` before any cross-run comparison.** Whole-board CPU load averaged
   across cores: a quiet desk sits at ~20-35 %, a busy one at 90 %+, and the same
   pipeline runs up to 1.4x slower in the busy case. Within-pass deltas survive
   that; absolute numbers do not. If two runs you are comparing have different
   `load %`, they are not comparable.
4. **Stage split before the headline.** Acquisition is excluded from
   `latency_ms` on purpose (it is the decoder: 2.8 ms at 720p, 11 % of the CPU
   time on this board, and 0.4 ms at 180p), and in the ROS arms most of the frame
   is `xport`. Reading `p50` without the stage table will attribute a transport
   cost to the pipeline.

## Metrics

| column | source | note |
|---|---|---|
| `p50/p95/p99` | records | over the window's frames; printed as mean ± std across passes |
| `fps` | records | frames inside the window over the window span |
| `cpu ms/f`, `cores` | `/proc` | workload CPU time per frame; `cores` = cpu/wall |
| `pss MB` | `/proc` | peak PSS of the process group (accounts for shared pages) |
| `load %` | tegrastats | whole-board CPU load, averaged across cores over the window. This is the column that says whether a run is comparable to another: the same design measured 27 % (range 16-36) on a quiet desk and 95 % with a browser open, and the pipeline ran 1.4x slower for it |
| `gpu %` | tegrastats | `GR3D_FREQ` duty cycle, not occupancy; a latency-bound chain reads low. Mean over the samples, not median: at low duty the 100 ms samples are bimodal (GPU work / no GPU work) and the median lands on the zero mode — the ROS chain read 7 % by mean and 0 % by median on the op-free matrix (measured before conditioning became a constant, so the values move; the estimator does not) |
| `W`, `J/f` | tegrastats | board-level `VDD_IN`; energy is the trapezoidal integral over the window |
| `tj °C` | tegrastats | junction temperature, max over the window |
| `xport` | records | ROS only: `transport_ms`, i.e. serialisation + DDS + scheduling (plus, at `inflight > 1`, the queueing the window introduced) |
| `hop` | records | ROS only: `hop_ms`, the measured wait in front of each node. Equals `xport` at `inflight=1`; growth beyond it is the queueing delay |
| `i`, `fill` | provenance, records | requested frames in flight and the realized/requested ratio. `fill < 0.9` is a warning: the acquisition node never filled the window, so the cell did not measure what its name says |
| `Δfps`, `Δp50` (curve) | records | within-pass difference against the *same cell* at `i=1`, printed in the frames-in-flight section |
| `L·X/i` | records | Little's law on the closed loop. ≈1 when the window is genuinely full; reads slightly under 1 because `latency_ms` excludes acquisition |
| `drops` | records | gaps in the `seq` set inside the window. Must be 0: a non-zero value means a topic history was shallower than the window and silently evicted an unacked sample |

Board power is a single rail, so energy can only be attributed per frame
end-to-end — never per pipeline stage. `tegrastats` lines carry a 1-second
wall-clock prefix, so the sampler timestamps the line when it reads it; a board
column can be up to one interval (default 100 ms) older than its `t_mono`.

This table is the reference; `analyze.py` also appends a condensed version to
every generated report (`glossary_markdown`, one entry per column of each table
it printed), so a `report.md` read months later carries the definition of its
own columns. The two are kept in sync by hand — add the column to both.

## The matrix (factors + design)

The experiment is written once as **factors** and a **design** list, not as N
hand-copied command templates:

```
factors.resolution   720p | 360p | 180p   -> source file, rotate, clip_frames, frame_mb
factors.model        n    | m    | x      -> TensorRT engine path
factors.chain        monolith | ros-N4-p2c1 | ros-N12-p8c3 -> cmd / shell + windows
conditioning.ops     the one stage-2 chain, run by every cell
```

A cell is `<chain>-<resolution>-<model>[-i<D>]` (`monolith-720p-n`,
`ros-N4-p2c1-180p-x`, `ros-N4-p2c1-720p-n-i4`): architecture, source
resolution, model variant and frames in flight -- and nothing else.
`design.cells` is the list of combos that actually exist (40: the 18-cell model
matrix, 14 shape cells and 8 window cells); leave the list out and `run.py`
generates the full product described next. Each chain factor also declares the
fourth coordinate's domain, `windows` (`[1]` for the monolith, `[1,2,4,8]` for
the ROS shapes); a design cell asking for a window outside it is rejected.

### The full matrix (`--design full`)

The 40-cell design is a set of *pruned* subsets, which is right for answering one
factor at a time and wrong if what you want is the whole picture. `--design full`
ignores `design.cells` and expands every combination the factors admit:

| dimension | values | count |
|---|---|---|
| chain | `monolith` + 8 ROS shapes | 9 |
| resolution | `720p`, `360p`, `180p` | 3 |
| model | `n`, `m`, `x` | 3 |
| frames in flight | `1`, `2`, `4`, `8` (ROS only) | 4 |

which is 9 monolith cells (window 1 only) + 8 x 3 x 3 x 4 = 288 ROS cells =
**297 cells**, i.e. **891 trials** at three passes. At the current budget the
ceilings sum to ~34 h per pass, so a three-pass run is roughly four days of
unattended board time (`run.py` prints its own upper bound before starting, and
warns if the output is the shared default root).

```bash
bench/run.py --design full --out results/bench-full --repeats 3
bench/run.py --design full --dry-run                       # 891 planned trials
bench/run.py --design full --conditions monolith-720p-n,ros-N4-p2c1-720p-n  # still a subset
```

Nothing about a cell changes: same command template, same conditioning, same
budget. What does change is that a family's window ladder is contiguous by
construction -- the design is emitted `resolution -> model -> chain -> window`
with the `reference_chain` placed immediately after the monolith, so inside each
33-cell block the monolith and the four-node chain are adjacent (the headline
pair) and every `-iD` cell sits next to its own `i1` zero. Blocks are still
shuffled per pass and reversed on alternate passes.

Two knobs exist for this mode and do nothing otherwise:

* **`trial.settle_s_windowed` (45 s).** A windowed cell is a different power and
  thermal state, so it settles longer -- as a rule, not as 216 hand-copied
  per-cell overrides. `analyze.py` reads each trial's budget back from
  `run.meta.json`, so the window it summarises starts after that trial's own
  settle rather than one global number.
* **`fps_model`.** A derived, deliberately pessimistic frame-rate hint for every
  cell the measured `trial.fps_hint` table does not cover: per process,
  residency = its stages + the hop out of it + a small fixed cost, then
  `X = min(D/Σ, 1/max s)` times `pessimism`, with `window_gain_cap` limiting what
  a window may buy (the one gain measured on this board was 1.5x). An entry in
  `trial.fps_hint` or `cells` always wins over it. Only an over-estimate costs
  data -- it shortens the ceiling until the analysis window comes up short -- so
  the model is allowed to be too slow, never too fast; a `window_short` warning
  after a full run is the signal that `pessimism` is now too large.

**The conditioning chain is a constant, not a coordinate.** Every cell -- both
front-ends, every shape, resolution, model and window -- runs the same seven ops
in the same order, declared once under `conditioning` in `scenarios.json`. It is
spliced into the ROS templates as `pre_ops:={pre_ops}` and spelled out as
repeated `--pre` in the monolith template, and `run.py` checks both forms against
that list before launching anything. A cell that ran a different (or empty)
stage 2 would not be a point on an axis, it would be a different experiment, and
the records could not tell you: `pre_ms` is a sum, so a missing chain reads ~0
and looks like an improvement.

What each axis can and cannot move (all three engines take FLOAT
`(1,3,640,640)` and emit FLOAT `(1,84,8400)`, and `imgsz` is no longer a
launch parameter: the inference node reads the contract off the engine it owns):

| axis | hops out of stage 2 | inference | acquisition |
|---|---|---|---|
| resolution | 2 x BGR frame: 2.76 -> 0.69 -> 0.17 MB | unchanged | 3 decode sizes |
| model | unchanged | conv compute only | unchanged |
| chain | hops per frame | unchanged | unchanged |
| `pre_split` (stage depth) | one hop per conditioning op | unchanged | unchanged |

The wire no longer carries a tensor. That tension -- "the letterboxed 640x640
tensor dominates the wire either way" -- is what used to cap the resolution
axis at ~34 % of the bytes; now every hop carries the BGR frame at source
resolution, so 720p -> 180p removes ~94 % of the payload bytes and the axis
finally measures what it is named after.

**The current graph has not been re-measured.** What is known is the per-hop
model fitted on the old five-message chain, where the fit is unambiguous
because one hop was a constant and the other scaled:

    xport = hops * (3.6 ms + 2.2 ms/MB)          # ~450 MB/s, linear to ~7 %

Predicted from it: ~13.3 ms at 720p and ~7.6 ms at 180p for the two-frame-hop
chain (down from 24.2 / 18.4), and ~4.0 ms added by each extra hop at 180p.
A single unsplit-vs-split smoke run measured +1.8 ms for one extra hop, which
is inside the noise of one non-interleaved pair: quote only interleaved
`bench/run.py` deltas, never that pair.

### The chain-shape axis

The second axis is the partition itself, written as a `chain` factor:
`acq|pre|infer|post` (four processes, the default) up to the twelve-process
shape. `|` separates processes, `+` merges stages inside one, and the label is
the coordinate `p{p}c{c}` -- p hops carry pixels, c carry candidates, and
`N = p+c+1` processes. The full table, the predictions and the reasoning live in
`docs/proposals/chain-shapes.md`; what matters for the harness is:

| cell prefix | N | p | c | one line |
| --- | --- | --- | --- | --- |
| `ros-N2-p1c0` | 2 | 1 | 0 | the monolith in a second process: prices one boundary |
| `ros-N3-p1c1` | 3 | 1 | 1 | pixels cross once |
| `ros-N3-p2c0` | 3 | 2 | 0 | same N, pixels cross twice -- the H1 pair with p1c1 |
| `ros-N4-p2c1` | 4 | 2 | 1 | the current chain: matrix cell and shape baseline |
| `ros-N5-p3c1` | 5 | 3 | 1 | one pixel hop more than p2c1: the additivity check |
| `ros-N5-p2c2` | 5 | 2 | 2 | first cut inside stages 4-6 |
| `ros-N6-p2c3` | 6 | 2 | 3 | stages 4-6 fully split |
| `ros-N12-p8c3` | 12 | 8 | 3 | one process per op: the transport extreme |

Every cell -- these shapes, the model matrix and the window ladder -- runs the
same seven-op chain (the constant in `scenarios.json`), so `ros-N4-p2c1` is at
once the matrix's ROS cell and the shape axis's four-node baseline. The design
lists it once; there is no op-free copy to keep in step. `run.py` validates the
name against the command before launching anything (N must equal p+c+1, the
`chain:=` string must have N groups, `-iD` must equal `inflight:=`, and both
front-ends must carry the canonical chain), because the cell name is what the
report is built from.

Recommended subsets, in the order they pay off:

```bash
# H1 -- same node count, different payload: ~1 h for one pass at 720p
bench/run.py --conditions ros-N3-p1c1-720p-n,ros-N3-p2c0-720p-n,monolith-720p-n

# the ladder, both resolutions (the resolution axis needs the pair)
bench/run.py --conditions monolith-720p-n,monolith-180p-n,ros-N2-p1c0-720p-n,ros-N2-p1c0-180p-n,ros-N4-p2c1-720p-n,ros-N4-p2c1-180p-n

# the extreme, to test linearity in the hop count
bench/run.py --conditions ros-N12-p8c3-720p-n,ros-N12-p8c3-180p-n,monolith-720p-n
```

### The frames-in-flight axis

`inflight:=D` (cell token `-i{D}`, omitted at 1) is how many frames the graph may
hold: the acquisition node publishes while fewer than `D` of its frames are
unacked, and every ack releases one. `D=1` is the demand-driven chain every
earlier cell measured, so those cells keep their names and their numbers; `D>1`
spends latency to buy throughput. The mechanism, the closed-loop model
(`X = min(D/Σs, 1/max s)`, knee at `D* = Σs/max s`, which is *not* the node
count) and the predictions live in `docs/proposals/frames-in-flight.md`.

What the harness pins, so a mislabelled cell cannot survive:

* `run.py` rejects a design cell whose `-iD` disagrees with the `inflight:=` in
  its command, in either direction -- the same rule as `N == p+c+1`.
* the same number is the depth of every data topic (`qos.latest(D)`), derived,
  never hand-set, so a window deeper than a history cannot be configured.
* the acquisition node drains the window before the terminal node writes the
  summary, so a `max_frames` stop does not truncate the tail.

Run a ladder with its own `i1` cell in the same invocation -- that is the local
zero for every window delta:

```bash
bench/run.py --out results/fif1 --conditions \
  monolith-720p-n,ros-N4-p2c1-720p-n,ros-N4-p2c1-720p-n-i2,ros-N4-p2c1-720p-n-i4,ros-N4-p2c1-720p-n-i8
```

The ladder cells set `settle_s: 45` in `cells`: a 6 fps cell and a 25 fps cell
are different power and thermal states on the same board, so the ramp is longer
than the 30 s the rest of the matrix uses. They also set a pessimistic
`fps_hint` -- the closed-loop prediction is the starting point, not a measurement.
Note that the conditioning chain changes the shape of the curve rather than just
its scale: at 720p the `pre` node alone costs ~103.5 ms, so it becomes the
bottleneck and the window stops buying throughput after two frames; at 180p the
ops are ~13.6 ms and the model still sets the pace.

### Pruning

The 18-cell model matrix is ~2.1 h per pass at the current budget (3 passes
~6.5 h), the 14 shape cells ~1.8 h, the 8 window cells ~0.9 h, so the full
40-cell design is ~4.8 h per pass (~14.5 h for 3) before engine load and gaps.
The whole product of the factors (`--design full`) is 297 cells and ~34 h per
pass. Those are ceilings: a trial stops as soon as its window is full, which is
usually earlier. The spread is wide: `monolith-180p-n` is ~140 s while
`ros-N4-p2c1-720p-x` is ~950 s, and the conditioning chain is what moved both
(the ops cost 103.5 ms at 720p and 13.6 at 180p, so the resolution axis now
changes the *per-frame cost*, not only the payload). That is also why the shape
and window axes are meant to run as their own `--conditions` subsets rather than
in one pass with the model matrix. `--conditions` selects by cell name:

```bash
bench/run.py --dry-run --conditions monolith-720p-n,ros-N4-p2c1-720p-n
bench/run.py --conditions monolith-720p-n,monolith-360p-n,monolith-180p-n,ros-N4-p2c1-720p-n,ros-N4-p2c1-360p-n,ros-N4-p2c1-180p-n
```

A design that answers one factor beats a design that answers none: keep the
anchor cell in every `design` (see *Calibration*), then add the cells needed for
the axis under test.

## Trial budget: settle by time, count by frames

The two failure modes here have different currencies, so the design uses both.

* **Frames** are the statistical unit: `p50/p95/p99` are computed from a frame
  count, so every cell gets the same `analysis_frames` and the percentiles are
  comparable across a matrix whose cells differ by 5x in speed. This also fixes a
  content-mix bias: a fixed 180 s window held 8.9 clip passes at the old op-free
  59.5 fps but 3.0 at 20.1 fps, so dense-traffic passages weighed ~3x more in the
  monolith trial than in the ROS trial. With the conditioning chain in every cell
  the fastest and slowest cells still differ ~3x (about 8 fps at `monolith-720p-n`
  against ~3 fps at `ros-N12-p8c3-720p-n`), so the bias the frame budget removes
  is smaller than it was, not gone.
* **Seconds** are the machine-state unit: the ~8 % drift and the thermal ramp are
  functions of elapsed time, so a pure frame budget would measure the fast cells
  cold. `settle_s` is dropped by the analyser, and each trial gets a time ceiling
  derived from its frame budget.

`scenarios.json` therefore carries `trial.settle_s` (30 s, plus
`settle_s_windowed` 45 s for the `-iD` cells), `trial.analysis_frames` and
`trial.fps_hint` (a table per chain, resolution and model, with `fps_model`
deriving one for any cell the table does not cover). The
hints are **0.7x** the estimated rate *with the seven-op chain*, not 0.9x, and
they are re-derived whenever the chain or the box changes: the same desk with a
browser and an editor open measured 1.4x slower with identical code, which turned
a full window into a `window_short` warning. The resolution is a level of the
table rather than a scale on the 720p row because the ops and the model do not
scale with pixels the same way: at 720p the chain is ~7x the inference of the n
engine, at 180p they are comparable, so the model spread widens as the resolution
falls. An over-estimate shortens the ceiling until the window goes short; an
under-estimate only runs a little longer. Re-derive from a clean run, keep the
margin, and treat a `window_short` warning as "the hint is now wrong". Per trial,
`run.py` derives:

```
duration  = settle_s + 1.1 * analysis_frames / fps_hint     # stop condition
max_frames= 2.0*settle_s*fps_hint + 1.5*analysis_frames      # hard cap behind it
loops     = ceil(max_frames / clip_frames) + 1               # exact, not 1000
```

The window is per trial, not per run: `analyze.py` reads `budget.settle_s` back
out of each `run.meta.json` (falling back to `trial.settle_s`, then to `--settle`
if you pass it), so a run that mixes 30 s and 45 s cells still starts every
window after that cell's own ramp.

The cap is sized to cover an fps *overshoot* (twice the hinted rate during the
settle, 50 % slack on the window), because it must never bind before the ceiling:
if it does, the trial ends mid-window and the cell is silently measured over
fewer frames. The first full run came within 20 frames of exactly that, which is
why the term is 2.0x rather than 1.5x.

Both front-ends already stop on whichever comes first (`main.py`, and
`acquisition_node.py` for the ROS chain), so this is config, not pipeline code.
`analyze.py` warns when a window came up short. `run.json` records the whole
budget table and every `run.meta.json` repeats its cell's budget, so a suspicious
trial can be audited after the fact.

## Calibration

The measured design has no duplicate control cell. `monolith-ctrl` survives only
in the `artifacts` (demo) block, where two visually identical videos are worth
having. The noise floor it used to measure is now carried by the deltas
themselves: the spread of the per-pass deltas against the **anchor** is the
number a ROS-vs-monolith difference has to clear, measured on the same clock as
the comparison instead of once, months ago, on a different session. Run the A/A pair
on demand when the harness itself changes (`--conditions` with a throwaway
scenario), because a control cell mostly caught harness bugs keyed off the
condition name -- output paths, env merge, process-group discovery.

## Adding an axis or a cell

* **New axis value** (another resolution, another model): add an entry to the
  matching `factors` table and list the cells in `design`. Nothing else changes.
* **New chain shape**: add the chain factor with its `windows` list. `--design
  full` picks it up with no further edit; the pruned design only measures it
  where it is listed.
* **New axis** (e.g. chain length): add a `factors.<name>` table, extend the
  cells to `[chain, resolution, model, <name>]`, and reference `{<name>}` in the
  chain templates.

`cmd` is an argv template, `shell` is for things that need
`source install/setup.bash`. Placeholders: `{python}`, `{repo}`, `{source}`,
`{model}`, `{rotate}`, `{loops}`, `{duration}`, `{out}`, `{condition}`,
`{repeat}`.

## Adding a condition

Append an entry to `scenarios.json`. `cmd` is an argv template, `shell` is for
things that need `source install/setup.bash`. Placeholders: `{python}`, `{repo}`,
`{source}`, `{rotate}`, `{loops}`, `{duration}`, `{out}`, `{condition}`,
`{repeat}`.

The ROS 2 nodes must produce the same `records.jsonl` / `summary.json` pair —
that contract is what makes the two front-ends comparable. The ROS condition
currently emits one extra per-record key (`transport_ms`); extra keys are
ignored by `analyze.py`.
