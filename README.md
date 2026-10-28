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

Run everything from the repository root, and only from bash with a clean `PATH`
(otherwise `python3` resolves to Homebrew's interpreter and ROS 2 is unset —
see `AGENTS.md`, Step 0):

```bash
env -i HOME="$HOME" TERM=dumb \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash -ic 'cd ~/Memoria/ros2_bench && <command>'
```

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

## Running the benchmark

Three steps: build the workspace, measure, analyse.

```bash
cd ws && colcon build --symlink-install && cd ..   # once, or after touching ws/

bench/run.py --repeats 3 --duration 180 --gap 5 --out results/bench   # ~28 min
bench/analyze.py --from 30 --to 170 --md results/report.md --json results/report.json
```

`bench/run.py` interleaves the conditions (`monolith`, `monolith-ctrl`, `ros-4node`)
and samples board + process telemetry; `bench/analyze.py` picks the analysis
window afterwards, so the same trials can be re-read with different `--from/--to`
without re-measuring. Quick checks:

```bash
bench/run.py --dry-run                                # print what would launch
bench/run.py --repeats 1 --duration 8 --out results/smoke   # smoke test (~40 s)
bench/analyze.py --dir results/smoke
```

Full flag reference, artefact layout, and the metric dictionary are in
`bench/README.md`; measurement policy (no pinning, no warmup, interleaved trials)
is in its *Policy* section.

See `AGENTS.md` for the architecture rules and per-pipeline details.
