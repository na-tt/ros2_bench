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
└── scenarios.json  # factors x design -> cells, + an `artifacts` (demo) block
```

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
* **An anchor cell.** One cell is repeated in every pass, so a comparison is a
  per-pass delta rather than two means measured hours apart. The duplicate
  control cell that used to serve this purpose is now only a demo artifact — see
  *Calibration*.

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
you spend half an hour:

```bash
bench/run.py --repeats 1 --duration 8 --gap 2 --out results/smoke \
  --conditions monolith-720p-n,ros-4node-720p-n
bench/analyze.py --dir results/smoke
rm -rf results/smoke          # gitignored, but do not let it be read as data
```

Expect `2/2 trials ok`; if a trial exits non-zero, read
`results/smoke/<condition>/r1/workload.log`.

### 3. The full run (3 cells x 3 passes x 180 s ≈ 28 min)

```bash
bench/run.py --repeats 3 --gap 5 --out results/bench \
  --conditions monolith-720p-n,ros-4node-720p-n
```

A matrix run is `--conditions` on a subset, not always the whole product; see
*The matrix* for the axes and pruning recipes. Wall clock per trial is
`--duration` plus engine load (~1.5 s for `yolov8n`, tens of seconds for
`yolov8x`) plus `--gap` between trials; 9 trials at 180 s ran 1690 s here.
Re-running into the same `--out` **overwrites** `<out>/<condition>/r<N>/`, so
give each session its own directory if you want to keep the old numbers
(`--out results/bench-$(date +%m%d-%H%M)`).

Long runs: detach it and poll, so a dropped terminal does not kill the trial:

```bash
nohup env -i HOME="$HOME" TERM=dumb \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash -ic 'cd ~/Memoria/ros2_bench && python3 bench/run.py --repeats 3 --duration 180 \
    --gap 5 --out results/bench' > results/full-run.log 2>&1 &
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
bench/run.py --demo --conditions monolith,ros-4node   # just the headline pair
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
└── compare-monolith-vs-ros-4node.mp4          # --align wall: each panel at its own pace
```

`detections.jsonl` is **not** part of the benchmark contract and is documented in
`monolith/yolocar/README.md` (*Seeing a run*). A front-end that never writes one
is still compliant; `records.jsonl`/`summary.json` are unchanged by its presence.

Wall clock for a full `--demo` pass on this board: ~10 min (monolith 34 s run +
61 s render, control likewise, ROS chain 140 s run + 179 s render, compare
118 s).

Common variants:

```bash
bench/run.py --dry-run                                  # what would be launched
bench/run.py --conditions monolith-720p-n,ros-4node-720p-n --repeats 4  # headline pair
bench/run.py --repeats 1 --duration 60                   # quick look
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

The ROS 2 front-end adds one key to this contract: `transport_ms`,
end-to-end latency minus the three stage timings (serialisation + DDS +
scheduling). The monolith has no such cost, so its absence there is not a
missing measurement. In the ROS chain `infer_ms` also contains the detect-head
decode/threshold (the candidates, not the raw head, cross the topic boundary):
compare `infer + post` as a block, not the two buckets separately.

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
   excludes zero. This is also why the anchor must appear in every pass: it is
   the control, and its repetition is what produces the comparator.
3. **Stage split before the headline.** Acquisition is excluded from
   `latency_ms` on purpose (it is the decoder, ~28 % of the CPU time), and in the
   ROS arms most of the frame is `xport`. Reading `p50` without the stage table
   will attribute a transport cost to the pipeline.

## Metrics

| column | source | note |
|---|---|---|
| `p50/p95/p99` | records | over the window's frames; printed as mean ± std across passes |
| `fps` | records | frames inside the window over the window span |
| `cpu ms/f`, `cores` | `/proc` | workload CPU time per frame; `cores` = cpu/wall |
| `pss MB` | `/proc` | peak PSS of the process group (accounts for shared pages) |
| `gpu %` | tegrastats | `GR3D_FREQ` duty cycle, not occupancy; a latency-bound chain reads low |
| `W`, `J/f` | tegrastats | board-level `VDD_IN`; energy is the trapezoidal integral over the window |
| `tj °C` | tegrastats | junction temperature, max over the window |
| `xport` | records | ROS only: `transport_ms`, i.e. serialisation + DDS + scheduling |

Board power is a single rail, so energy can only be attributed per frame
end-to-end — never per pipeline stage. `tegrastats` lines carry a 1-second
wall-clock prefix, so the sampler timestamps the line when it reads it; a board
column can be up to one interval (default 100 ms) older than its `t_mono`.

## The matrix (factors + design)

The experiment is written once as **factors** and a **design** list, not as N
hand-copied command templates:

```
factors.resolution  720p | 360p | 180p   -> source file, rotate, clip_frames
factors.model       n    | m    | x      -> TensorRT engine path
factors.chain       monolith | ros-4node -> cmd / shell template
```

A cell is `<chain>-<resolution>-<model>` (`monolith-720p-n`,
`ros-4node-180p-x`). `design.cells` is the list of combos that actually exist
(the full factorial is 2 x 3 x 3 = 18); leave the list out and `run.py`
generates the full product.

What each axis can and cannot move (all three engines take FLOAT
`(1,3,640,640)` and emit FLOAT `(1,84,8400)`, so `imgsz` and `input_dtype` are
constant across the matrix):

| axis | image topic | tensor topic | inference | acquisition |
|---|---|---|---|---|
| resolution | 2.76 → 0.69 → 0.17 MB | unchanged (4.92 MB) | unchanged | 3 decode sizes |
| model | unchanged | unchanged | conv compute only | unchanged |
| chain | hops per frame | hops per frame | unchanged | unchanged |

The resolution axis is therefore a **transport** probe with a modest compute
side-effect: the letterboxed 640x640 tensor dominates the wire either way, so
720p→180p removes ~34 % of the bytes, not ~22x. If the four-node chain is
bandwidth-bound, `xport` should land near 45-50 ms at 180p against ~67 ms at
720p; a much larger collapse would mean transport is scheduling-bound, not
bandwidth-bound.

### Pruning

18 cells is ~93 min per pass at the current budget (2 passes ~3.1 h, 3 passes
~4.7 h) before engine load and gaps, and the spread is wide: `monolith-180p-n`
is ~91 s while `ros-4node-720p-x` is ~718 s. `--conditions` selects by cell name:

```bash
bench/run.py --dry-run --conditions monolith-720p-n,ros-4node-720p-n
bench/run.py --conditions monolith-720p-n,monolith-360p-n,monolith-180p-n,ros-4node-720p-n,ros-4node-360p-n,ros-4node-180p-n
```

A design that answers one factor beats a design that answers none: keep the
anchor cell in every `design` (see *Calibration*), then add the cells needed for
the axis under test.

## Trial budget: settle by time, count by frames

The two failure modes here have different currencies, so the design uses both.

* **Frames** are the statistical unit: `p50/p95/p99` are computed from a frame
  count, so every cell gets the same `analysis_frames` and the percentiles are
  comparable across a matrix whose cells differ by 4x in speed. This also fixes a
  content-mix bias: a fixed 180 s window is 5.2 clip passes at 36 fps but 1.5 at
  10 fps, so dense-traffic passages weigh ~3.5x more in the monolith trial than
  in the ROS trial.
* **Seconds** are the machine-state unit: the ~8 % drift and the thermal ramp are
  functions of elapsed time, so a pure frame budget would measure the fast cells
  cold. `settle_s` is dropped by the analyser, and each trial gets a time ceiling
  derived from its frame budget.

`scenarios.json` therefore carries `trial.settle_s`, `trial.analysis_frames` and
a deliberately pessimistic `trial.fps_hint` table (`chain -> model`, scaled by
`fps_scale` per resolution). Per trial, `run.py` derives:

```
duration  = settle_s + 1.1 * analysis_frames / fps_hint     # stop condition
max_frames= 1.5*settle_s*fps_hint + 1.2*analysis_frames      # hard cap behind it
loops     = ceil(max_frames / clip_frames) + 1               # exact, not 1000
```

Both front-ends already stop on whichever comes first (`main.py`, and
`acquisition_node.py` for the ROS chain), so this is config, not pipeline code.
An over-estimated `fps_hint` truncates the window and `analyze.py` says so; an
under-estimate only costs a few seconds. `run.json` records the whole budget
table and every `run.meta.json` repeats its cell's budget, so a suspicious trial
can be audited after the fact.

## Calibration

The measured design has no duplicate control cell. `monolith-ctrl` survives only
in the `artifacts` (demo) block, where two visually identical videos are worth
having. The noise floor it used to measure is now carried by repeating the
**anchor** cell in every pass: the std of the per-pass deltas is the number a
ROS-vs-monolith difference has to clear, measured on the same clock as the
comparison instead of once, months ago, on a different session. Run the A/A pair
on demand when the harness itself changes (`--conditions` with a throwaway
scenario), because a control cell mostly caught harness bugs keyed off the
condition name -- output paths, env merge, process-group discovery.

## Adding an axis or a cell

* **New axis value** (another resolution, another model): add an entry to the
  matching `factors` table and list the cells in `design`. Nothing else changes.
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
