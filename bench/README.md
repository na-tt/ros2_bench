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
  `monolith`. The difference between the two is the noise floor; a ROS-vs-
  monolith delta is only meaningful if it clears it.

## Usage

```bash
# 3 repeats x 120 s per condition, alternating order
bench/run.py --repeats 3 --duration 120

# just look at what would be launched
bench/run.py --dry-run

# aggregate, with the steady-state window chosen now rather than during the run
bench/analyze.py --from 30 --to 240 --md results/report.md --json results/report.json
```

`--duration` is measurement time only (the workload starts its clock once the
engine is loaded) and `--loops` lets a short clip drive a long run.

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

Board power is a single rail, so energy can only be attributed per frame
end-to-end — never per pipeline stage. `tegrastats` lines carry a 1-second
wall-clock prefix, so the sampler timestamps the line when it reads it; a board
column can be up to one interval (default 100 ms) older than its `t_mono`.

## Adding a condition

Append an entry to `scenarios.json`. `cmd` is an argv template, `shell` is for
things that need `source install/setup.bash`. Placeholders: `{python}`, `{repo}`,
`{source}`, `{rotate}`, `{loops}`, `{duration}`, `{out}`, `{condition}`,
`{repeat}`.

The ROS 2 node must produce the same `records.jsonl` / `summary.json` pair —
that contract is what makes the two front-ends comparable.
