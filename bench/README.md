# bench — measurement harness

Out-of-process tooling for comparing the single-process pipeline with the ROS 2
implementations. It lives beside `monolith/` and `ws/`, **imports neither**, and
talks to them only through a command template and a file contract. That keeps it
outside the "no cross-project imports" rule and makes the monolith and a ROS
launch file configurable in exactly the same way.

```
bench/
├── run.py          # orchestrator: interleaved trials + sampler + artefacts
├── sampler.py      # board (tegrastats) + workload (/proc) telemetry -> CSV
├── analyze.py      # trials -> markdown/JSON report, window chosen here
└── scenarios.json  # conditions: command templates
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
  of B would attribute that drift to B. `run.py` rotates the condition order
  every repeat.
* **A control condition.** `monolith-ctrl` is a byte-identical clone of
  `monolith` under a different label. The difference between the two is the
  noise floor; a ROS-vs-monolith delta is only meaningful if it clears it. It is
  an A/A test (it measures the measurement), not a third treatment — see
  *Conditions* below.

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

### 2. Smoke test first (~40 s)

Cheap and it catches a moved engine, a stale workspace or a broken topic before
you spend half an hour:

```bash
bench/run.py --repeats 1 --duration 8 --gap 2 --out results/smoke
bench/analyze.py --dir results/smoke
rm -rf results/smoke          # gitignored, but do not let it be read as data
```

Expect `3/3 trials ok`; if a trial exits non-zero, read
`results/smoke/<condition>/r1/workload.log`.

### 3. The full run (3 conditions x 3 x 180 s ≈ 28 min)

```bash
bench/run.py --repeats 3 --duration 180 --gap 5 --out results/bench
```

Wall clock per trial is `--duration` plus ~1.5 s for the monolith and ~4.5 s for
the ROS chain (TensorRT engine load), plus `--gap` between trials; 9 trials at
180 s ran 1690 s here. Re-running into the same `--out` **overwrites**
`<out>/<condition>/r<N>/`, so give each session its own directory if you want to
keep the old numbers (`--out results/bench-$(date +%m%d-%H%M)`).

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
bench/analyze.py --from 30 --to 170 --md results/report.md --json results/report.json
```

`--from`/`--to` are seconds after the **first recorded frame**, so startup and
settling stay in the data and you can re-window without re-measuring (drop
`--from/--to` to use whole trials). `--dir` points at a different artefact root.

### Options worth knowing

| flag | applies to | effect |
|---|---|---|
| `--conditions a,b` | `run.py` | subset by name from `scenarios.json` |
| `--repeats N` | `run.py` | trials per condition (rotation needs ≥ `len(conditions)` for a balanced design) |
| `--duration S` | `run.py` | measurement seconds; the clock starts once the engine is loaded |
| `--gap S` | `run.py` | idle seconds between trials (default 5) |
| `--out DIR` | `run.py` | artefact root (default `results/bench`) |
| `--interval-ms N` | `run.py` | sampler period (default 100) |
| `--dry-run` | `run.py` | print the expanded commands and exit |
| `--scenario FILE` | `run.py` | alternate condition set |
| `--dir` / `--records` / `--sampler` | `analyze.py` | where to read trials from |

`--loops` lives in `scenarios.json`, not on the CLI: it is what lets a 40 s clip
drive a 180 s trial.

Common variants:

```bash
bench/run.py --dry-run                                  # what would be launched
bench/run.py --conditions monolith,ros-4node --repeats 5 # drop the control, add repeats
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
├── run.json
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

## Metrics

| column | source | note |
|---|---|---|
| `p50/p95/p99` | records | p99/max are usually desktop-session noise, not the pipeline |
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

## Conditions

* `monolith` — the single-process baseline.
* `monolith-ctrl` — a byte-identical clone of it with a different label and
  output directory (an A/A test). It is not a third condition: it measures the
  *measurement*, so the ROS-vs-monolith delta can be compared against a real
  number instead of a guess. It also catches harness bugs that key off the
  condition (output paths, env merge, process-group discovery) and a broken
  rotation, neither of which shows up as within-condition spread. With
  `--conditions monolith,ros-4node` you can trade it for extra repeats.
* `ros-4node` — the four-node chain (acquisition → preprocessing → inference →
  postprocessing), one process per node, paced by an ack from the terminal
  node. It needs the ROS 2 workspace to be built
  (`cd ws && colcon build --symlink-install`) and `yolocar` importable via
  `PYTHONPATH`; the shell condition in `scenarios.json` sources the workspace
  and sets `PYTHONPATH`. See `ws/src/yolocar_ros/README.md`.

## Adding a condition

Append an entry to `scenarios.json`. `cmd` is an argv template, `shell` is for
things that need `source install/setup.bash`. Placeholders: `{python}`, `{repo}`,
`{source}`, `{rotate}`, `{loops}`, `{duration}`, `{out}`, `{condition}`,
`{repeat}`.

The ROS 2 nodes must produce the same `records.jsonl` / `summary.json` pair —
that contract is what makes the two front-ends comparable. The ROS condition
currently emits one extra per-record key (`transport_ms`); extra keys are
ignored by `analyze.py`.
