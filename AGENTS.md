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
        └── yolocar_ros/         # ROS 2 wrapper for `yolocar`
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

---

## Developer & Agent Workflow Setup

When working in this workspace, follow this exact setup sequence:

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

* **To run a ROS 2 implementation** (not written yet, see *Current State*):
  ```bash
  cd ws
  source install/setup.bash
  ros2 run yolocar_ros yolocar_node
  ```

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
  timestamp *in the message*, not re-time around its own callback.
* `summary.json`: latency stats, `stage_ms`
  (`acquisition/preprocess/inference/postprocess/presentation`), `events`
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

* **`ws/` does not exist yet.** No ROS 2 package has been written, so the
  `ros-short` condition in `bench/scenarios.json` is a disabled placeholder.
  Next step: `ws/src/yolocar_ros` emitting the `records.jsonl` / `summary.json`
  contract above (and carrying `seq` + capture timestamp through the graph so
  latency is measured across nodes instead of around a callback).
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
  including 17 hardware tests against the real engine).
