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
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── processor.py
│       │   ├── main.py
│       │   ├── cuda.py
│       │   ├── engine.py
│       │   └── py.typed
│       └── tests/               # pytest suite of this project
│
├── assets/                      # Shared benchmark inputs (models, videos, images)
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
   * ROS 2 nodes inside `ws/src/<name>_ros/` must **never** re-implement CV logic. They should only handle ROS 2 boilerplate (node initialization, sub/pub, `cv_bridge` conversions) and call the corresponding function imported from `<name>` (e.g. `from yolocar.processor import process_frame`).

3. **Package Linking**:
   * Every pipeline project is installed in editable mode into the active Python environment, individually: `pip install -e ./monolith/<name>`.
   * Python code inside `ws/src/` imports directly from those installed packages.

4. **Benchmark Assets**:
   * Weights, footage and frames are shared by all pipelines and live in the
     repository-root `assets/` (not inside the individual projects).
   * Default paths in configs are relative (`assets/models/...`), so run commands
     from the repository root.

5. **Workspace Isolation**:
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
  python3 -m yolocar.main --source assets/videos/clip.mp4
  ```

* **To run the tests of one pipeline project**:
  ```bash
  pytest monolith/yolocar          # or: cd monolith/yolocar && pytest
  ```

* **To run a ROS 2 implementation**:
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
     importing from `<name>.processor`.
* **Modifying CV Logic**:
  * Modify code **only** inside `monolith/<name>/src/<name>/`. Because the project is installed with `-e` and `colcon` uses `--symlink-install`, changes take effect immediately across both standalone scripts and ROS 2 nodes without needing reinstallations.

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