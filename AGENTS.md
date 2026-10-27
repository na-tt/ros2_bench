# Agent Guide: ros2_bench: ROS 2 vs. Monolith Benchmark Project

## Project Overview
This repository benchmarks single-process Python computer vision (CV) pipelines against equivalent ROS 2 (`rclpy`) node implementations to evaluate overhead, latency, and throughput.

## Repository Architecture
The repository uses a split structure between shared Python logic and the ROS 2 workspace:

ros2_bench/
├── monolith/                    # Shared CV logic & standalone benchmarks
│   ├── pyproject.toml           # Setuptools package configuration
│   ├── pipelineA/               # Pipeline A module (e.g., pipelineA/processor.py, main.py)
│   ├── pipelineB/               # Pipeline B module
│   └── pipelineC/               # Pipeline C module
│
└── ws/                          # ROS 2 Colcon Workspace
    └── src/
        ├── pipeline_a_ros/      # ROS 2 wrapper package for Pipeline A
        ├── pipeline_b_ros/      # ROS 2 wrapper package for Pipeline B
        └── pipeline_c_ros/      # ROS 2 wrapper package for Pipeline C

## Key Architectural Rules

1. **Core Logic Single Source of Truth**:
   * ALL image processing functions, algorithms, models, or data structures must be implemented inside `monolith/pipelineX/`.
   * ROS 2 nodes inside `ws/src/pipeline_x_ros/` must **never** re-implement CV logic. They should only handle ROS 2 boilerplate (node initialization, sub/pub, `cv_bridge` conversions) and call the corresponding function imported from `pipelineX`.

2. **Package Linking**:
   * `monolith` is treated as a local Python package installed in editable mode (`pip install -e ./monolith`).
   * Python code inside `ws/src/` imports directly from `monolith` modules (e.g., `from pipelineA.processor import process_frame`).

3. **Workspace Isolation**:
   * The ROS 2 workspace lives entirely within `ws/`.
   * Never run `colcon build` outside the `ws/` directory.

---

## Developer & Agent Workflow Setup

When working in this workspace, follow this exact setup sequence:

### 1. Environment Setup & Dependency Installation
Before building or running nodes, ensure `monolith` is installed in editable mode within the active Python environment:

```bash
cd /path/to/ros2_bench/monolith
pip install -e .
```

### 2. Building the ROS 2 Workspace
Always build from the `ws/` root using `--symlink-install` so Python modifications apply without requiring a rebuild:

```bash
cd /path/to/ros2_bench/ws
colcon build --symlink-install
source install/setup.bash
```

### 3. Running Benchmarks

* **To run a standalone single-process benchmark**:
  ```bash
  python3 monolith/pipelineA/main.py
  ```

* **To run a ROS 2 implementation**:
  ```bash
  cd ws
  source install/setup.bash
  ros2 run pipeline_a_ros pipeline_a_node
  ```

---

## Instructions for AI Agents

* **Adding a New Pipeline**:
  1. Create `monolith/pipelineX/` with `__init__.py`, `processor.py`, and `main.py`.
  2. Ensure `monolith/pyproject.toml` includes `pipelineX*` under setuptools package discovery if explicit pattern matching is used.
  3. Create `ws/src/pipeline_x_ros/` as an `ament_python` package with a node importing from `pipelineX.processor`.
* **Modifying CV Logic**:
  * Modify code **only** inside `monolith/pipelineX/`. Because `monolith` is installed with `-e` and `colcon` uses `--symlink-install`, changes take effect immediately across both standalone scripts and ROS 2 nodes without needing reinstallations.

## Pipelines
### yolocar
#### Description
Road objects video detection with yolo on dashcam footage.

#### Specification
- (1) **acquisition:** read frame from video file. bgr frame at experimental resolution
- (2) **preprocessing:** image conditioning to expected input characteristics. rgb image, 1×3×640×640, float/uint8 per model
- (3) **inference:** run yolo and produce candidate detection boxes.
- (4) **NMS:** perform Non Maximum Supression to reduce detections.
- (5) **domain filter:** filter to only class ids corresponding to road objects
- (6) **coordinate adaptation:** boxes translated to input frame coordinates.