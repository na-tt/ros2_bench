# ros2_bench

Benchmark ROS 2 (`rclpy`) against single-process Python on CV applications.

## Layout

```
ros2_bench/
├── monolith/          # container: one independent Python project per pipeline
│   └── yolocar/       # own pyproject.toml, src/yolocar/, tests/
├── assets/            # shared benchmark inputs (weights, videos, images)
└── ws/                # ROS 2 colcon workspace (wrappers only, no CV logic)
```

`monolith/` is not a package: every subdirectory is installed on its own, e.g.
`pip install -e ./monolith/yolocar`. Each project is wrapped by a thin ROS 2
node in `ws/src/` that imports the pipeline package instead of reimplementing
it.

## Quick start

```bash
pip install -e ./monolith/yolocar
python3 -m yolocar.main --source assets/videos/clip.mp4   # single-process baseline

cd ws && colcon build --symlink-install && source install/setup.bash
ros2 run yolocar_ros yolocar_node                         # ROS 2 counterpart (todo)
```

See `AGENTS.md` for the architecture rules and per-pipeline details.
