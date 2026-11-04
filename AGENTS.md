# Agent Guide: ros2_bench: ROS 2 vs. Monolith Benchmark Project

## Project Overview
This repository benchmarks single-process Python computer vision (CV) pipelines against equivalent ROS 2 (`rclpy`) node implementations to evaluate overhead, latency, and throughput.

## Repository Architecture
`monolith/` is a **container directory**, not a package. Each pipeline inside it is an
independent, separately installable Python project (its own `pyproject.toml`, `src/`,
`tests/`, `README.md`). The ROS 2 workspace wraps those projects without duplicating logic.

```
ros2_bench/
├── monolith/                    # Container only: no pyproject.toml here
│   ├── README.md                # Index of pipeline projects
│   └── yolocar/                 # Independent project ("yolocar")
│       ├── pyproject.toml       # name = "yolocar", src layout
│       ├── README.md
│       ├── src/yolocar/         # importable package (ALL pipeline logic)
│       │   ├── __init__.py      # lazy re-exports
│       │   ├── config.py        # tunables + COCO table (no logic)
│       │   ├── acquisition.py   # stage 1: sources + rotation
│       │   ├── preprocessing.py # stage 2: coercion + conditioning chain
│       │   ├── filters.py       #   the conditioning ops themselves
│       │   ├── model_input.py   # engine input contract (letterbox + tensor)
│       │   ├── inference.py     # stage 3: engine call + decode
│       │   ├── postprocessing.py# stages 4-6: NMS/filter/coords
│       │   ├── render.py        # optional annotation + sinks
│       │   ├── processor.py     # facade: wires stages 2-6
│       │   ├── main.py          # CLI / benchmark
│       │   ├── cuda.py
│       │   ├── engine.py
│       │   └── py.typed
│       └── tests/               # pytest suite of this project
│
├── assets/                      # Shared benchmark inputs (models, videos, images)
│
├── bench/                       # Measurement harness (imports no pipeline code)
│   ├── run.py                   # interleaved trials + sampler orchestration
│   ├── sampler.py               # tegrastats + /proc -> CSV on one clock
│   ├── analyze.py               # trials -> report, analysis window chosen here
│   ├── scenarios.json           # conditions as command templates
│   └── README.md                # policy, file contract, metric dictionary
│
├── results/                     # generated benchmark artefacts (gitignored)
│
└── ws/                          # ROS 2 Colcon Workspace
    └── src/
        ├── yolocar_interfaces/  # msgs: FrameMeta + per-stage payloads
        └── yolocar_ros/         # 4 nodes + launch/bench.launch.py
```

New pipelines follow the same shape: `monolith/<name>/` with `pyproject.toml`
(`name = "<name>"`), `src/<name>/`, `tests/` and a `ws/src/<name>_ros/` wrapper.

## Key Architectural Rules

1. **One project per pipeline directory**:
   * `monolith/` itself has **no** `pyproject.toml`, no `__init__.py` and no shared
     package metadata; it only groups the pipeline projects.
   * Each `monolith/<name>/` declares its own project name (`name = "<name>"`), its
     own dependencies/extras, and its own `[tool.pytest.ini_options]`.
   * Never add cross-project Python imports (e.g. `from monolith...`); importing a
     pipeline package is always `import <name>`.

2. **Core Logic Single Source of Truth**:
   * ALL image processing functions, algorithms, models, or data structures must be implemented inside `monolith/<name>/src/<name>/`.
   * ROS 2 nodes inside `ws/src/<name>_ros/` must **never** re-implement CV logic. They should only handle ROS 2 boilerplate (node initialization, sub/pub, `cv_bridge` conversions) and call the corresponding function imported from `<name>` (e.g. `from yolocar.processor import process_frame`, or the per-stage functions documented under *Modules*).
   * Pipeline stages are exposed as standalone functions so a chain can be split over nodes: one node may call several stages in one callback (short chain) or each stage may live in its own node (long chain). The aggregate helper is only a convenience composition of those functions.

3. **Package Linking**:
   * Every pipeline project is installed in editable mode into the active Python environment, individually: `pip install -e ./monolith/<name>`.
   * Python code inside `ws/src/` imports directly from those installed packages.

4. **Benchmark Assets**:
   * Weights, footage and frames are shared by all pipelines and live in the
     repository-root `assets/` (not inside the individual projects).
   * Default paths in configs are relative (`assets/models/...`), so run commands
     from the repository root.

5. **Measurement Harness (``bench/``)**:
   * The harness must not import pipeline code. It launches a condition as a
     command template and consumes only files, so the monolith and a ROS 2
     launch file are configured identically.
   * Every front-end must emit the same per-frame contract (``records.jsonl``)
     and summary (``summary.json``). See ``bench/README.md``.
   * Do not add clock pinning, GC tuning, warmup stages or headless-mode
     requirements to the harness: the machine's natural noise floor and
     desktop-session activity are deliberate parts of the measurement.
   * Compare conditions within interleaved passes, never as back-to-back
     blocks: a claim is a per-pass delta against a fixed reference cell (the
     *anchor*), not a difference of two means measured hours apart. The anchor is
     not measured more often than the rest in a full pass (every cell runs once
     per pass) -- it is the shared zero, so any run of a subset must include it.

6. **Workspace Isolation**:
   * The ROS 2 workspace lives entirely within `ws/`.
   * Never run `colcon build` outside the `ws/` directory.

7. **Canonical shell is bash, never fish**:
   * **Every** command that touches this project -- build, test, benchmark, ROS 2,
     pip, git, file inspection -- must run through `bash`. Fish is only the
     operator's interactive shell (and the shell the coding agent happens to run
     in); it is *not* part of the project and must not leak into it.
   * Reason: this is a custom machine. Fish's environment puts Homebrew in front
     of `/usr/bin` (`python3` resolves to `/home/linuxbrew/.linuxbrew/bin/python3`,
     not the Jetson's `/usr/bin/python3` with numpy/opencv/tensorrt), and it does
     **no** ROS 2 setup at all. Commands run under fish therefore fail or, worse,
     silently use the wrong interpreter.
   * `~/.bashrc` is the single source of truth for this project's environment: it
     sources `/opt/ros/jazzy/setup.bash` and
     `~/Memoria/ros2_bench/ws/install/setup.bash`.
   * Never `brew install` a project dependency and never `pip install` into the
     Homebrew interpreter. The project's Python is `/usr/bin/python3`.

---

## Developer & Agent Workflow Setup

When working in this workspace, follow this exact setup sequence:

### 0. Get a bash environment (always first)
Do this before any other step, from any tool or terminal:

```bash
env -i HOME="$HOME" TERM=dumb \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash -ic 'cd ~/Memoria/ros2_bench && <your command>'
```

* `bash -c` alone is **not** enough: `.bashrc` starts with
  `case $- in *i*) ;; *) return ;; esac`, so a non-interactive bash skips the ROS
  `source` lines and inherits whatever `PATH` its parent had (under fish: Homebrew
  first, no `ROS_DISTRO`). `-i` is what loads `.bashrc`.
* The explicit clean `PATH` is required because an interactive bash still inherits
  the parent's `PATH`; `~/.bashrc` only *prepends* `~/.local/bin`, it does not
  remove Homebrew. Drop the `PATH=` assignment and `python3` silently becomes the
  Homebrew one.
* `env -i` keeps the agent's own pollution (`PYTHONPATH`, `VIRTUAL_ENV`,
  `LD_LIBRARY_PATH`, brew vars) out of the measurement. Re-add only what a step
  genuinely needs, in front of the command, e.g.
  `env -i ... bash -ic 'cd ... && PYTHONPATH=monolith/yolocar/src python3 -m yolocar.main --help'`.
* Verify once per session, expecting `ROS_DISTRO=jazzy` and `/usr/bin/python3`:
  ```bash
  env -i HOME="$HOME" TERM=dumb \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    bash -ic 'echo $ROS_DISTRO; command -v python3'
  ```
* "It worked in fish" is not evidence. If a command has not been run this way,
  it has not been run.

### 1. Environment Setup & Dependency Installation
Before building or running nodes, install the pipeline project(s) you touch in editable mode within the active Python environment:

```bash
pip install -e ./monolith/yolocar          # runtime deps included
pip install -e "./monolith/yolocar[dev]"   # + pytest
```

On a machine without `pip` (the Jetson's `/usr/bin/python3` already has numpy,
opencv, tensorrt and pytest, but no pip), skip the install and run with
`PYTHONPATH=monolith/yolocar/src`. `bench/scenarios.json` passes exactly that.

### 2. Building the ROS 2 Workspace
Always build from the `ws/` root using `--symlink-install` so Python modifications apply without requiring a rebuild:

```bash
cd /path/to/ros2_bench/ws
colcon build --symlink-install
source install/setup.bash
```

### 3. Running Benchmarks

* **To run a standalone single-process benchmark** (from the repository root):
  ```bash
  python3 -m yolocar.main --source assets/video/0000f77c-6257be58.mov --rotate 180
  ```

* **To run the tests of one pipeline project**:
  ```bash
  pytest monolith/yolocar          # or: cd monolith/yolocar && pytest
  ```
  The hardware tier loads the real TensorRT engine; it runs wherever CUDA,
  `tensorrt` and the `.engine` exist (they do on this Jetson, so the suite
  covers the real driver, not just fakes).

* **To run a full measurement** (interleaved passes + telemetry, see
  *Measurement Protocol* below). Cells are `<chain>-<resolution>-<model>`, so a
  run selects them explicitly; `--settle`/`--analysis-frames` override the trial
  budget only for smoke runs:
  ```bash
  bench/run.py --repeats 3 --conditions monolith-720p-n,ros-4node-720p-n
  bench/analyze.py --md results/report.md --json results/report.json
  ```

* **To run a ROS 2 implementation** (the four-node chain -- see *Current State*):
  ```bash
  cd ws && colcon build --symlink-install && source install/setup.bash
  cd ..   # paths default to the repository root
  PYTHONPATH=monolith/yolocar/src:$PYTHONPATH \
    ros2 launch yolocar_ros bench.launch.py duration:=120 \
      records:=results/ros/records.jsonl summary:=results/ros/summary.json
  ```
  The launch file starts acquisition, preprocessing, inference and
  postprocessing as four processes and stops them together. `duration:=-1` runs
  the source to its end. Add `"pre_ops:=clahe;bilateral:d=9"` to condition the
  frame, and `pre_split:=true` to give each op its own process (see *Current
  State*).

---

## Instructions for AI Agents

* **Adding a New Pipeline**:
  1. Create `monolith/<name>/` as an independent project: `pyproject.toml`
     (`name = "<name>"`, `[tool.setuptools.packages.find] where = ["src"]`),
     `src/<name>/` (`__init__.py`, `processor.py`, `main.py`, ...), `tests/` and
     `README.md`. Copy `monolith/yolocar/` as the reference.
  2. Do **not** touch a repository-level `pyproject.toml` — there is none; each
     project discovers its own package under `src/`.
  3. Create `ws/src/<name>_ros/` as an `ament_python` package with a node
     importing from `<name>.processor` (facade) or from the per-stage modules.
* **Running anything**: route every command through the bash wrapper in
  *Step 0* above (clean `PATH` + `bash -ic`), never through fish and never through
  a plain inherited-environment `bash -c`.
* **Modifying CV Logic**:
  * Modify code **only** inside `monolith/<name>/src/<name>/`. Because the project is installed with `-e` and `colcon` uses `--symlink-install`, changes take effect immediately across both standalone scripts and ROS 2 nodes without needing reinstallations.

## Measurement Protocol (`bench/`)

The harness is out-of-process and launches a condition as a command template; it
never imports pipeline code. Details and the metric dictionary are in
`bench/README.md`.

| layer | file | job |
| --- | --- | --- |
| workload | `monolith/<name>/src/<name>/main.py`, later the ROS node | emits per-frame records + summary |
| demo (optional, never measured) | `bench/run.py --demo` + `python -m yolocar.demo` | one replay per condition, rendered offline from the detections sidecar |
| sampler | `bench/sampler.py` | `tegrastats` (board) + `/proc` (workload) -> CSV, one clock |
| orchestrator | `bench/run.py` | interleaved trials, rotates order, drives the sampler |
| analysis | `bench/analyze.py` | picks the steady-state window, aggregates across trials |
| estimators | `bench/METHOD.md` | what each number is computed over (cell/trial/pass, std vs CI, deltas) |

**File contract -- every front-end must emit this**, monolith and ROS 2 alike:

* `records.jsonl`: one object per frame, `seq, loop, t_mono, acq_ms, pre_ms,
  infer_ms, post_ms, present_ms, latency_ms, n_det`. `t_mono` is
  `time.monotonic()` (CLOCK_MONOTONIC, system-wide on Linux), so it joins
  directly against `sampler.csv`. A ROS node must carry the equivalent
  timestamp *in the message*, not re-time around its own callback. Extra keys
  are allowed: the ROS front-end adds `transport_ms` (serialisation + DDS +
  scheduling, reported as `xport`), `hop_ms` (the measured wait in front of each
  busy node: equals `transport_ms` at `inflight=1`, its growth beyond that is the
  queueing the window introduced) and `inflight` (frames outstanding when this
  one was published -- the realized window, so a cell that never filled it cannot
  be mistaken for one that bought nothing).
* `summary.json`: latency stats, `stage_ms`
  (`acquisition/preprocess/inference/postprocess/presentation`; the ROS
  front-end adds `transport`), `events`
  (start/engine_loaded/first_frame/end), `provenance`, `fps`/`fps_wall`.
  `provenance` carries `shape` (the `chain:=` string) and `inflight` (the
  frames-in-flight window) in the ROS front-end: two runs that differ only in
  partition depth or window are otherwise indistinguishable on disk.
* `sampler.csv` is produced by the harness, not by the workload.
* `detections.jsonl` is an **artifact, not contract**: the final detections per
  frame, written only when asked for, consumed by `yolocar.demo` to render a run
  offline. Its presence changes nothing in `records.jsonl`/`summary.json`.

**Non-negotiable policy** -- these are decisions, not defaults to "improve":

* **No clock pinning, no governor changes, no CPU affinity, no GC tuning, no
  headless mode.** The desktop session's background activity is the noise floor
  and is part of the measurement.
* **No warmup stage, nothing discarded at runtime.** `processor.warmup()` exists
  for interactive use only. Startup and settling are recorded; the analysis
  window is chosen in `analyze.py` (`--settle` seconds, then `--frames` frames,
  both defaulting to the scenario's `trial` block) and can be re-chosen without
  re-measuring.
* **Frames for the statistics, seconds for the machine.** Every cell's trial
  budget targets the same `analysis_frames` -- that is what makes p95/p99
  comparable across cells that differ 4x in speed, and it removes the content-mix
  bias of an equal-seconds window. The *stop* condition stays a time ceiling
  derived from those frames, because the ~8 % drift and the thermal ramp are
  functions of elapsed time: a pure frame budget would measure the fast cells
  cold. `fps_hint` only sizes the box; the report warns when a window came up
  short.
* **Blocks, zig-zag, anchor.** Cells are grouped into blocks (`block_by`) and the
  block order is shuffled per pass with a logged seed, while the direction inside
  a block alternates. The cyclic Latin square is kept only where it is exact
  (<= 6 cells, `repeats` a multiple of the cell count). A randomised order turns
  residual drift into variance the std can see; a deterministic unbalanced one
  turns it into a bias the std cannot. Measured on this Jetson: ~7 % between
  identical conditions over 4 interleaved 120 s trials, ~3 % within a condition,
  and ~8 % drift between replays inside a single run (in either direction).
  Never run all of A, then all of B.
* **Report mean ± std across passes, and deltas for claims.** The headline table
  carries mean ± sample std with `n`; with n < 6 the std itself is only good to
  ~40 %, so the per-trial values are printed next to it. Any real claim is the
  within-pass delta against the anchor with its 95 % CI -- two means measured
  hours apart are not comparable. The old duplicate control cell (`monolith-ctrl`)
  is no longer measured: the anchor's repetition provides the same noise floor on
  the same clock. Keep it as a demo artifact and as an on-demand A/A check when
  the harness itself changes.
* **Annotation is debug only.** `--draw`/`--output` cost ~22 ms/frame; that time
  is reported as `stage_ms.presentation` and subtracted from `fps`
  (`fps_wall` keeps the raw rate).

**Baseline** (Orin Nano, full matrix, 3 interleaved passes, `settle_s=30` +
`analysis_frames=2500`, rotation-corrected footage). Mean ± sample std across
passes; the anchor is bold. Measured on the chain *before* the model contract
moved into the inference node, so the `ros-4node` p50 includes the 4.9 MB
tensor hop that no longer exists -- treat it as the historical reference, not
as the current graph (see *Current State*, `pre_ms` attribution):

| cell | p50 ms | fps | cpu ms/f | cores | gpu % | W | J/f | pss MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **monolith-720p-n** | **12.57 ± 0.19** | **59.5** | 25.6 | 1.53 | 28 | 9.6 | 0.165 | 639 |
| monolith-180p-n | 12.71 ± 0.06 | 70.3 | 14.4 | 1.01 | 33 | 9.6 | 0.140 | 607 |
| ros-4node-720p-n | 42.40 ± 0.93 | 20.1 | 70.2 | 1.41 | 6 | 8.1 | 0.408 | 923 |
| ros-4node-180p-n | 36.16 ± 0.90 | 26.3 | 49.8 | 1.31 | 8 | 8.2 | 0.316 | 871 |

Reading. At fixed `imgsz=640` the resolution axis moves **CPU, not latency**:
720p→180p leaves p50 flat (`+0.14 ± 0.33 ms`, not significant) while CPU per
frame drops 44 % and throughput rises 18 %, because the only things resolution
touches are decode, letterbox and memory traffic. The model axis sets latency
(12.6 / 26.3 / 46.8 ms at 720p for n/m/x). The chain is **CPU-bound but no longer
single-core**: 25.6 ms of CPU per 15.4 ms of latency (1.53 cores) at 720p, 1.0
core at 180p -- the excess is the multithreaded letterbox/tensor conversion.
Acquisition is 2.8 ms (11 % of CPU) at 720p and 0.4 ms at 180p, so the old
"decode + rotate costs ~8 ms" figure was mostly the rotation these cells do not
pay. ROS 2 overhead lands on the saturated resource: ~70 ms of CPU per frame
across 4 processes.

## Pipelines
### yolocar
#### Description
Road objects video detection with yolo on dashcam footage.

#### Specification
- (1) **acquisition:** read frame from video file. bgr frame at experimental resolution
- (2) **preprocessing:** general image conditioning at **source resolution**, as a chain of ops (CLAHE, bilateral, gaussian, gamma, histogram equalisation, median, unsharp). BGR uint8 in, BGR uint8 out, geometry unchanged. Stage 2 says *how the image should look*; it does not know the model. Empty chain = identity (bit-exact, zero-copy), which is the baseline.
- (2b) **model input contract:** letterbox + BGR→RGB + HWC→NCHW + dtype/scale. Parameterised only by the engine's `InputSpec`, and therefore part of the inference stage, not of preprocessing.
- (3) **inference:** run yolo and produce candidate detection boxes.
- (4) **NMS:** perform Non Maximum Supression to reduce detections.
- (5) **domain filter:** filter to only class ids corresponding to road objects (default set: `yolocar.config.ROAD_CLASS_IDS`)
- (6) **coordinate adaptation:** boxes translated to input frame coordinates.

**Canonical order:** NMS (4) runs on *all* classes, then the domain filter (5)
drops the rest, then (6) maps coordinates back. Do not reorder these stages --
with class-agnostic NMS and the `max_candidates` cap the result is not
order-invariant.

#### Modules
One module per upstream/downstream boundary; the facade only wires them.

| Stage | Module |
| --- | --- |
| acquisition (sources, rotation correction) | `yolocar.acquisition` (`open_source`, `rotate_frame`) |
| preprocessing (conditioning chain) | `yolocar.preprocessing` (`OPS`, `Op`, `parse`, `enhance`, `condition`, `as_bgr_frame`) |
| conditioning ops | `yolocar.filters` (`clahe`, `bilateral`, `gaussian`, `gamma`, `hist_equalization`, `median`, `unsharp`) |
| model input contract | `yolocar.model_input` (`InputSpec`, `LetterboxInfo`, `letterbox`, `to_tensor`, `prepare`) |
| inference (engine call, decode head) | `yolocar.inference` (`infer`, `decode_predictions`, `input_spec`) |
| NMS | `yolocar.postprocessing.nms` |
| domain filter | `yolocar.postprocessing.domain_filter` |
| coordinate adaptation | `yolocar.postprocessing.coordinate_adaptation` |
| annotation / sinks (optional) | `yolocar.render` (`draw_detections`, `draw_hud`, `open_sink`) |
| detections sidecar (offline artifact) | `yolocar.detections` (`DetectionLog`, `iter_records`, `build_header`) |
| ROS node (any stage span) | `yolocar_ros.chain_node` (`ChainNode`, `parse_stages`, `entry_point`) |
| ROS terminal (records/summary/ack) | `yolocar_ros.terminal` (`Terminal`) |
| ROS flow control (frames in flight) | `yolocar_ros.window` (`CreditWindow`), `yolocar_ros.qos` (`latest(depth)`) |
| offline renderer (demo material) | `yolocar.demo` (`python -m yolocar.demo`) |
| single-process wiring | `yolocar.processor.YoloCarProcessor` |

Stages 4-6 exchange `yolocar.postprocessing.Candidates` (parallel numpy arrays:
`boxes` xyxy, `scores`, `class_ids`), which is also the natural wire format for
a ROS 2 message. `to_detections()` converts to the presentation `Detection`
type only at the very end.

## Current State & Known Issues

* **`ws/src/yolocar_ros` is the four-node chain** (acquisition ->
  preprocessing -> inference -> postprocessing), one process per stage, plus
  `ws/src/yolocar_interfaces` with the messages (`FrameMeta`, `ImageFrame`,
  `CandidatesFrame`, `FrameAck`). Each stage calls the corresponding `yolocar`
  function and adds only ROS 2 boilerplate. The acquisition node publishes one
  frame and waits for the terminal node's ack, so the graph is demand-driven
  like the monolith (one frame in flight, no silent drops).
  `bench/scenarios.json` (the `ros-4node` chain, i.e. every `ros-4node-*` cell)
  drives it through `launch/bench.launch.py`; any node exiting shuts the launch
  down.
* **Stage 2 is splittable by launch data, not by code.** `pre_ops` says *what*
  conditioning runs (`;` separated, empty = identity); `pre_split:=true` cuts
  it into one `preprocessing` process per op, wired head to tail through
  `yolocar/filter_<i>` topics. The first node still reads `yolocar/frame` and
  the last still writes `yolocar/input`, so nothing downstream can tell how
  deep the stage was split. `pre_ms` and the `pre_op_names`/`pre_op_ms` arrays
  accumulate additively, so the records are identical in both configurations
  and the extra hops land in `transport_ms`. The partition policy is one pure
  function (`stage_two_instances` in the launch file), tested without ROS.
  With one frame in flight, more nodes can only add latency -- depth is a
  *cost* axis here until in-flight depth > 1 becomes a measured factor.
* **No tensor crosses a topic.** Both hops out of stage 2 carry an
  `ImageFrame`, and the model input contract is applied by the inference node,
  which reads `InputSpec` off the engine it owns. Consequences: the 4.9 MB
  fixed tensor payload is gone (so transport scales with source resolution
  again), `TensorFrame` is deleted, and `input_dtype` no longer exists as a
  launch argument -- a dtype that disagrees with the engine is not expressible.
* **Compare `infer + post` as one block in the ROS chain.** The inference node
  publishes decoded/thresholded `Candidates` instead of the raw
  `(1, 84, 8400)` head (2.8 MB/frame), so `infer_ms` there includes the model
  contract *and* `decode_predictions`/`select_candidates`. `acq_ms`, `pre_ms`
  and the detections remain directly comparable.
* **`pre_ms` means conditioning only, in both front-ends.** Letterbox/tensor now
  counts inside `infer_ms`. Numbers recorded before this change are therefore
  not comparable bucket for bucket: the `pre` and `infer` columns of the table
  above mix the old and new attribution, only `p50/p99/fps/cpu/gpu` are
  unaffected. Re-measure before quoting a stage split.
* **The ROS chain is transport-bound, and that is the measurement.** The ROS
  records add `transport_ms` (serialisation + DDS + scheduling) and the summary
  adds a `transport` stage bucket. On this Jetson the two large payloads (2.7 MB
  BGR frame, 4.9 MB float32 tensor) make the four-node chain ~42 ms p50 / ~20 fps
  at 720p, against ~12.6 ms / ~60 fps for the monolith. Measured across the
  resolution axis, `xport` behaves like **~7 ms of fixed cost plus ~2.2 ms/MB**
  (~450 MB/s): 24.2 / 21.2 / 18.4 ms at 720p / 360p / 180p. The tensor is
  constant, so the resolution axis can only remove ~34 % of the bytes -- and it
  does, proportionally. Do not reduce the overhead by shrinking the measured
  stages or by inserting queues; that is exactly what the benchmark exists to
  show.
  **Never compare these numbers to another session's.** An earlier run of the
  same design measured `xport` = 67 ms, 84 ms p50 end to end, with identical wire
  payloads and no relevant code change. Whether that was DDS falling back to
  loopback instead of shared memory, or a busier desktop session, is unresolved
  -- which is exactly why the report is built on within-pass deltas.
* **Frames in flight (`inflight:=D`) is the concurrency axis, and it is the one
  form of parallelism allowed.** The ack is a credit: the acquisition node
  publishes while fewer than `D` of its frames are unacknowledged
  (`CreditWindow`), the same number is the history depth of every data topic
  (`qos.latest(D)` -- a shallower history evicts an unacked sample even under
  `RELIABLE`, and the window would not notice), frame 0 is synchronous
  (discovery + engine-ready gate) and the window is drained before the terminal
  node writes the summary. **No other concurrency is permitted**: every node
  stays single-threaded, one mutually exclusive callback group, batch 1, no
  reentrant callbacks, no deferred frames. That constraint is what makes the
  closed-loop model exact (`X = min(D/Σs, 1/max s)`, knee at `D* = Σs/max s`,
  which is *not* the node count) and what keeps `records.jsonl` ordered by `seq`
  with non-decreasing `t_mono` at any `D` -- queues preserve order, a bigger
  window only changes how many frames are in them. Cell token `-i{D}`, appended
  only when `D > 1`, and `run.py` rejects a name that disagrees with the
  command. Ladder {1,2,4,8} on `ros-4node` both resolutions and {1,2,4} on
  `ros-N12-p8c3`; `docs/proposals/frames-in-flight.md` has the model, the five
  predictions and the measurement plan. **Not measured yet** (a 180p smoke run
  confirmed the mechanism: 29.5 -> 44.2 -> 44.4 fps at `D` = 1/2/4 with p50
  29.5 -> 39.8 -> 85.8 ms, i.e. the knee is at 2 as predicted).
* **Presentation is offline, never a fifth node.** A run can write
  `detections.jsonl` next to `records.jsonl` (`--detections` on the monolith,
  `detections:=` on the launch file) and `python -m yolocar.demo` renders it onto
  the clip afterwards. This is how the ROS chain gets a picture without a 2.7 MB
  frame fan-out or a viewer in the ack chain. In-run drawing/encoding stays
  debug-only (`--draw`/`--output`, ~16 ms/frame with `mp4v`, timed into
  `present_ms` and subtracted from `fps`). Demo material comes from
  `bench/run.py --demo`, a separate invocation that runs one replay per
  condition; measured commands are untouched by it.
* **`imgsz` is a dynamic-shape hint; the dtype is an engine fact.** The
  inference node reads both off the engine it loads
  (`yolocar.inference.input_spec`), so `input_dtype` is no longer a parameter at
  all. `imgsz` survives for dynamic engines only
  (`TensorRTEngine(default_input_shape=...)`); a static engine such as the
  shipped `yolov8n_fp16.engine` (float32 / 640) wins. A single engine built
  with an optimization profile could serve several input sizes without
  per-resolution exports -- documented in
  `docs/proposals/preprocessing-chain.md` as a deferred experiment, because
  letting `imgsz` track resolution makes inference cost O(pixels) and swamps
  the conditioning ops the resolution axis is meant to measure.
* **The chain shape is a launch argument, and every shape is one executable.**
  `chain:=acq|pre|infer|post` (default, p2c1) up to
  `chain:=acq|pre|pre|pre|pre|pre|pre|pre|infer|nms|filter|coord` (p8c3, twelve
  processes); `|` separates processes, `+` merges stages, and `pre_groups:=2,5`
  cuts the conditioning chain inside stage 2. `ws/src/yolocar_ros/yolocar_ros/
  chain_node.py` runs any contiguous span (`stages:=pre,infer,post`) and the
  three historic executables are one-line entry points for one span each, so
  there is a single implementation. Three properties are derived, not
  configured: the engine is loaded only by the span that contains `infer`, the
  terminal node is the one that ends on `coord` (it alone writes records,
  summary, ack and the detections sidecar), and timings accumulate so a split
  stage reports the same number as an unsplit one. The cell name
  `ros-N{x}-p{p}c{c}` is validated against the graph in `bench/run.py` (N must
  be p+c+1 and the `chain:=` string must have N groups) and the shape lands in
  `summary.provenance.shape`. `docs/proposals/chain-shapes.md` has the full
  table, the predictions and the two-tier measurement design. **The axis has
  not been measured yet** -- the cells exist, the numbers do not.
* **Run the launch from the repository root.** Default source/model/records
  paths are relative, and `yolocar` must be on `PYTHONPATH`
  (`PYTHONPATH=monolith/yolocar/src:$PYTHONPATH`) because there is no `pip` on
  the Jetson's `/usr/bin/python3` and the pipeline project is not installed
  into the ROS workspace. The bench `ros-4node-*` cells do both.
* **Never name a ROS package `yolocar`.** The scaffold was originally created as
  `yolocar`, which shares the import name of `monolith/yolocar/src/yolocar`, and
  because `~/.bashrc` sources `ws/install/setup.bash` the empty scaffold shadowed
  the real package in every bash session (`import yolocar` resolved to
  `ws/install/yolocar/lib/python3.12/site-packages/yolocar/__init__.py`). Renaming
  to `yolocar_ros` fixed it: `import yolocar_ros` works and `import yolocar` now
  correctly needs `PYTHONPATH=monolith/yolocar/src`, as the `bench` conditions
  already do. Keep wrapper packages `<name>_ros`.
* **Rotation belongs to the file, not the scenario.** The original
  `assets/video/0000f77c-6257be58.mov` is stored upside down and
  `CAP_PROP_ORIENTATION_META` reports 270 for it, which is wrong -- do not
  auto-apply the container flag, pass `--rotate 180`. The per-resolution
  re-encodes (`_720p`/`_360p`/`_180p`) had the rotation baked into the pixels by
  ffmpeg, which also cleared the display matrix, so they need `rotate: 0`
  (`run.py` allows a per-cell override; `bench.launch.py` defaults to 0). Getting
  this wrong does not fail a run -- it measures inference on an upside-down road
  scene, and only `n_det` and the demo video look wrong.
* **`YoloCarConfig.source` default (`assets/videos`) is stale.** The directory
  was renamed to `assets/video/` (singular, commit `007caae`) and now holds a
  `.mov`, so the default fails either way (`no such dir`, or `no images found`
  if pointed at the directory). Always pass `--source`; fix the default when
  someone decides whether it should name the directory or the shipped clip.
* **`monolith/yolocar/tests/golden/postprocess.json`** freezes postprocessing
  output from before the module split. Do not regenerate it to make a test
  pass: a diff there means detection behaviour changed.
* **`results/` is generated and gitignored.** Keep a run's directory if you want
  to compare against it later; nothing in it is source.
* Test counts at the end of this session: 202 (`pytest monolith/yolocar`,
  including 18 hardware tests against the real engine, the detections sidecar,
  the offline renderer and the CLI `--pre` path); 82 passed + 1 skipped
  (`colcon test --packages-select yolocar_interfaces yolocar_ros`): ament lint,
  the records/summary contract, sidecar/records parity, the real-engine
  stage-split test, the launch file's partition policy for all eight shapes, the
  shape-identity test that every partition finds exactly the same detections, the
  credit-window state machine and its contract (topic depth == window), and a
  real-graph test that a windowed chain loses no frame (contiguous `seq`, no
  short tail) and finds exactly the same detections at `inflight:=1` and `:=3`.
