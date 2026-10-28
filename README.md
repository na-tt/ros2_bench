# ros2_bench

Benchmark ROS 2 (`rclpy`) against single-process Python on CV applications.

## Layout

```
ros2_bench/
├── monolith/          # container: one independent Python project per pipeline
│   └── yolocar/       # own pyproject.toml, src/yolocar/, tests/
├── assets/            # shared benchmark inputs (weights, videos, images)
├── bench/             # measurement harness: interleaved trials, telemetry, reports
└── ws/                # ROS 2 colcon workspace (wrappers only, no CV logic)
```

`monolith/` is not a package: every subdirectory is installed on its own, e.g.
`pip install -e ./monolith/yolocar`. Each project is wrapped by a thin ROS 2
node in `ws/src/` that imports the pipeline package instead of reimplementing
it.

## Quick start

```bash
pip install -e ./monolith/yolocar
python3 -m yolocar.main --source assets/video/0000f77c-6257be58.mov --rotate 180  # baseline

cd ws && colcon build --symlink-install && source install/setup.bash
cd .. && PYTHONPATH=monolith/yolocar/src:$PYTHONPATH \
  ros2 launch yolocar_ros bench.launch.py duration:=120 \
    records:=results/ros/records.jsonl summary:=results/ros/summary.json
```

The ROS 2 counterpart is a four-node chain
(acquisition → preprocessing → inference → postprocessing), one process per
stage; see `ws/src/yolocar_ros/README.md`.

See `AGENTS.md` for the architecture rules and per-pipeline details, and
`bench/README.md` for the measurement protocol.
