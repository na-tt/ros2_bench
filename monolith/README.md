# monolith

Single-process (non-ROS) implementations of the benchmarked CV pipelines.

`monolith/` is **only a container directory**: it is not a Python package and has
no `pyproject.toml` of its own. Every subdirectory is an independent,
separately installable Python project (its own `pyproject.toml`, `src/`,
`tests/` and `README.md`) that owns all of the pipeline logic. ROS 2 packages in
`../ws/src` import from these projects and add only middleware boilerplate.

## Packages

| Package                 | Project root         | Purpose                                     | ROS 2 wrapper       |
| ----------------------- | -------------------- | ------------------------------------------- | ------------------- |
| [`yolocar`](yolocar/)   | `monolith/yolocar/`  | YOLO road-object detection (TensorRT)       | `yolocar_ros` (todo) |

## Layout

```
monolith/                        # container only, nothing importable here
└── yolocar/                     # independent project
    ├── pyproject.toml           # name = "yolocar"
    ├── README.md
    ├── src/yolocar/             # importable package
    └── tests/                   # pytest suite for this project
```

## Install / test / run

Each project installs and runs on its own, from its own directory:

```bash
pip install -e ./monolith/yolocar             # runtime deps included
pip install -e "./monolith/yolocar[dev]"      # + pytest
pytest monolith/yolocar                       # uses monolith/yolocar/pyproject.toml
python -m yolocar.main --source assets/videos # cwd = repository root
```

Benchmark inputs (weights, footage) are **shared** across pipelines and live in
the repository-root `assets/` directory, not inside the individual projects.
