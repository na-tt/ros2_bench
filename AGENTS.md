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
│       │   ├── preprocessing.py # stage 2: letterbox + tensor
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
   * Compare conditions with interleaved trials plus a duplicate control
     condition, never as back-to-back blocks.

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

* **To run a full measurement** (interleaved trials + telemetry, see
  *Measurement Protocol* below):
  ```bash
  bench/run.py --repeats 3 --duration 300
  bench/analyze.py --from 30 --to 270 --md results/report.md --json results/report.json
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
  the source to its end.

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
| sampler | `bench/sampler.py` | `tegrastats` (board) + `/proc` (workload) -> CSV, one clock |
| orchestrator | `bench/run.py` | interleaved trials, rotates order, drives the sampler |
| analysis | `bench/analyze.py` | picks the steady-state window, aggregates across trials |

**File contract -- every front-end must emit this**, monolith and ROS 2 alike:

* `records.jsonl`: one object per frame, `seq, loop, t_mono, acq_ms, pre_ms,
  infer_ms, post_ms, present_ms, latency_ms, n_det`. `t_mono` is
  `time.monotonic()` (CLOCK_MONOTONIC, system-wide on Linux), so it joins
  directly against `sampler.csv`. A ROS node must carry the equivalent
  timestamp *in the message*, not re-time around its own callback. Extra keys
  are allowed: the ROS front-end adds `transport_ms` (serialisation + DDS +
  scheduling) and `analyze.py` reports it as `xport` when present.
* `summary.json`: latency stats, `stage_ms`
  (`acquisition/preprocess/inference/postprocess/presentation`; the ROS
  front-end adds `transport`), `events`
  (start/engine_loaded/first_frame/end), `provenance`, `fps`/`fps_wall`.
* `sampler.csv` is produced by the harness, not by the workload.

**Non-negotiable policy** -- these are decisions, not defaults to "improve":

* **No clock pinning, no governor changes, no CPU affinity, no GC tuning, no
  headless mode.** The desktop session's background activity is the noise floor
  and is part of the measurement.
* **No warmup stage, nothing discarded at runtime.** `processor.warmup()` exists
  for interactive use only. Startup and settling are recorded; the analysis
  window is chosen in `analyze.py` (`--from`/`--to`, seconds after the first
  frame) and can be re-chosen without re-measuring.
* **Interleaved trials plus a duplicate control.** `monolith-ctrl` is a
  byte-identical clone of `monolith`; the difference between them *is* the noise
  floor. Measured on this Jetson: ~7 % between identical conditions over 4
  interleaved 120 s trials, ~3 % within a condition, and ~8 % drift between
  replays inside a single run (in either direction). Never run all of A, then
  all of B.
* **Annotation is debug only.** `--draw`/`--output` cost ~22 ms/frame; that time
  is reported as `stage_ms.presentation` and subtracted from `fps`
  (`fps_wall` keeps the raw rate).

**Baseline** (Orin Nano, `yolov8n_fp16.engine`, 40 s clip replayed, 20 s analysis
window, 2x120 s per condition). Sanity check, not a target:

| | p50 | fps | cpu ms/f | cores | gpu % | W | J/f | pss MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monolith | 17.2 ms | 34.8 | 29.0 | 1.0 | 26 | 10.8 | 0.31 | 470 |

Reading: the chain uses ~1 core while the GPU sits at ~26 % duty, i.e. it is
**CPU-bound**, and acquisition (decode + rotate, ~8 ms) is ~28 % of the CPU time
while being excluded from the latency headline. ROS 2 overhead will land on the
saturated resource. Harness overhead itself is +0.15 ms/frame (0.5 %).

## Pipelines
### yolocar
#### Description
Road objects video detection with yolo on dashcam footage.

#### Specification
- (1) **acquisition:** read frame from video file. bgr frame at experimental resolution
- (2) **preprocessing:** image conditioning to expected input characteristics. rgb image, 1×3×640×640, float/uint8 per model
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
| preprocessing (letterbox, tensor) | `yolocar.preprocessing` (`preprocess`, `letterbox`, `to_tensor`, `LetterboxInfo`) |
| inference (engine call, decode head) | `yolocar.inference` (`infer`, `decode_predictions`, `InputSpec`) |
| NMS | `yolocar.postprocessing.nms` |
| domain filter | `yolocar.postprocessing.domain_filter` |
| coordinate adaptation | `yolocar.postprocessing.coordinate_adaptation` |
| annotation / sinks (optional) | `yolocar.render` (`draw_detections`, `open_sink`) |
| single-process wiring | `yolocar.processor.YoloCarProcessor` |

Stages 4-6 exchange `yolocar.postprocessing.Candidates` (parallel numpy arrays:
`boxes` xyxy, `scores`, `class_ids`), which is also the natural wire format for
a ROS 2 message. `to_detections()` converts to the presentation `Detection`
type only at the very end.

## Current State & Known Issues

* **`ws/src/yolocar_ros` is the four-node chain** (acquisition ->
  preprocessing -> inference -> postprocessing), one process per stage, plus
  `ws/src/yolocar_interfaces` with the messages (`FrameMeta`, `ImageFrame`,
  `TensorFrame`, `CandidatesFrame`, `FrameAck`). Each stage calls the
  corresponding `yolocar` function and adds only ROS 2 boilerplate. The
  acquisition node publishes one frame and waits for the terminal node's ack, so
  the graph is demand-driven like the monolith (one frame in flight, no silent
  drops). `bench/scenarios.json` (`ros-4node`) drives it through
  `launch/bench.launch.py`; any node exiting shuts the launch down.
* **Compare `infer + post` as one block in the ROS chain.** The inference node
  publishes decoded/thresholded `Candidates` instead of the raw
  `(1, 84, 8400)` head (2.8 MB/frame), so `infer_ms` there includes
  `decode_predictions`/`select_candidates`. `acq_ms`, `pre_ms` and the
  detections remain directly comparable.
* **The ROS chain is transport-bound, and that is the measurement.** The ROS
  records add `transport_ms` (serialisation + DDS + scheduling) and the summary
  adds a `transport` stage bucket. On this Jetson the two large payloads (2.7 MB
  BGR frame, 4.9 MB float32 tensor) make the four-node chain ~70-85 ms p50 / ~10
  fps, against ~17 ms / ~35 fps for the monolith. Do not reduce it by shrinking
  the measured stages or by inserting queues; that is exactly what the
  benchmark exists to show.
* **`imgsz` and `input_dtype` are launch parameters, not engine facts.** The
  preprocessing node cannot load the engine (that is the inference node's job),
  so the launch file passes both and the inference node validates them against
  the real engine contract. The shipped `yolov8n_fp16.engine` wants
  `float32` / 640. A mismatch raises an explicit error instead of casting.
* **Run the launch from the repository root.** Default source/model/records
  paths are relative, and `yolocar` must be on `PYTHONPATH`
  (`PYTHONPATH=monolith/yolocar/src:$PYTHONPATH`) because there is no `pip` on
  the Jetson's `/usr/bin/python3` and the pipeline project is not installed
  into the ROS workspace. The bench `ros-4node` condition does both.
* **Never name a ROS package `yolocar`.** The scaffold was originally created as
  `yolocar`, which shares the import name of `monolith/yolocar/src/yolocar`, and
  because `~/.bashrc` sources `ws/install/setup.bash` the empty scaffold shadowed
  the real package in every bash session (`import yolocar` resolved to
  `ws/install/yolocar/lib/python3.12/site-packages/yolocar/__init__.py`). Renaming
  to `yolocar_ros` fixed it: `import yolocar_ros` works and `import yolocar` now
  correctly needs `PYTHONPATH=monolith/yolocar/src`, as the `bench` conditions
  already do. Keep wrapper packages `<name>_ros`.
* **The clip is stored upside down: use `--rotate 180`.** `CAP_PROP_ORIENTATION_META`
  reports 270 for `assets/video/0000f77c-6257be58.mov`, which is wrong -- do not
  auto-apply the container flag.
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
* Test counts at the end of this session: 109 (`pytest monolith/yolocar`,
  including 17 hardware tests against the real engine); 20 passed + 1 skipped
  (`colcon test --packages-select yolocar_ros`, including the ament lint tests
  and the real-engine stage-split test).
