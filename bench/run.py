#!/usr/bin/env python3
"""Interleaved trial orchestrator: runs conditions round-robin and samples them.

Why interleaved and not "all of A, then all of B": clocks are not pinned and the
garbage collector is not touched, so a run drifts by several percent as the
machine warms. Sequential blocks attribute that drift to whichever condition
ran second. Round-robin with alternating order spreads it across conditions,
and a duplicated *control* condition (same command twice) measures the noise
floor directly, so any real difference can be quoted against it.

``--demo`` is a separate mode: instead of trials it runs each condition's
``artifacts`` block once (one replay of the clip, detections sidecar on, no
sampler) and then renders the videos. Demo material is never produced inside a
measured trial, so a rendering failure cannot damage a measurement and the
measured commands stay byte-identical to the published baseline.

Each trial gets its own directory with the workload's artefacts, a
``sampler.csv`` (board + workload telemetry, see ``bench/sampler.py``) and a
``run.meta.json`` (command, exit code, machine state before/after).

The orchestrator never imports pipeline code: a condition is just a command
template, so the monolith and a ROS 2 launch file are configured the same way.

Usage::

    bench/run.py --repeats 3 --duration 300
    bench/run.py --conditions monolith,monolith-ctrl --dry-run
    bench/run.py --scenario bench/scenarios.json --out results/bench
    bench/run.py --demo --conditions monolith,ros-4node
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLER = Path(__file__).resolve().with_name("sampler.py")


# ------------------------------------------------------------- machine state
def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def machine_state() -> dict[str, object]:
    """Cheap snapshot so a suspicious trial can be checked afterwards."""
    temperatures = []
    for zone in sorted(Path("/sys/devices/virtual/thermal").glob("thermal_zone*")):
        value = _read(str(zone / "temp"))
        if value and value.lstrip("-").isdigit():
            temperatures.append(int(value) / 1000.0)
    return {
        "loadavg": _read("/proc/loadavg"),
        "thermal_c_max": round(max(temperatures), 2) if temperatures else None,
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
    }


# ------------------------------------------------------------------- commands
def expand(value: str | list[str], context: dict[str, str]) -> str | list[str]:
    if isinstance(value, list):
        return [expand(item, context) for item in value]  # type: ignore[misc]
    return value.format(**context)


def _rotate_of(condition: dict[str, object], scenario: dict[str, object]) -> object:
    """Rotation is a property of the footage, not of the experiment.

    The original clip is stored upside down and needs 180; the re-encoded
    ``_720p``/``_360p``/``_180p`` files had the rotation baked into the pixels
    by ffmpeg, which also cleared the display matrix, so they need 0. A
    condition may therefore override the scenario-level default.
    """
    if "rotate" in condition:
        return condition["rotate"]
    return scenario.get("rotate", 0)


def build_argv(condition: dict[str, object], context: dict[str, str]) -> list[str]:
    """A condition is either an argv list or a shell string (for `source ...`)."""
    shell = condition.get("shell")
    if shell:
        return ["bash", "-lc", str(expand(str(shell), context))]
    cmd = condition.get("cmd")
    if not cmd:
        raise SystemExit("condition needs either 'cmd' or 'shell'")
    expanded = expand(list(cmd), context)  # type: ignore[arg-type]
    return [str(part) for part in expanded]


# ----------------------------------------------------------------------- demo
def run_demo(scenario: dict[str, object], args: argparse.Namespace) -> int:
    """Run each condition's artefact pass once, then render the videos.

    Command templates only: this module never imports pipeline code, exactly
    like the trial loop. The steps are ``run`` (one replay + detections
    sidecar) then ``render`` (``yolocar.demo``), per condition, plus an
    optional ``compare`` template for the side-by-side video.
    """
    artifacts = scenario.get("artifacts") or {}
    conditions = artifacts.get("conditions") or {}  # type: ignore[union-attr]
    if not conditions:
        raise SystemExit("scenario has no 'artifacts.conditions' block")

    out_root = Path(args.out or artifacts.get("out_dir") or (REPO / "results/demo"))  # type: ignore[union-attr]
    if not out_root.is_absolute():
        out_root = (REPO / out_root).resolve()

    wanted = [name.strip() for name in (args.conditions or "").split(",") if name.strip()]
    base_context = {
        "python": sys.executable,
        "repo": str(REPO),
        "source": str(scenario.get("source", "")),
        "loops": str(artifacts.get("loops", 1)),  # type: ignore[union-attr]
        "demo": str(out_root),
    }

    steps: list[tuple[str, dict[str, object], dict[str, str], Path]] = []
    for name, condition in conditions.items():  # type: ignore[union-attr]
        if wanted and name not in wanted:
            continue
        if not condition.get("enabled", True):
            print(f"skip {name}: disabled in scenario", file=sys.stderr)
            continue
        directory = out_root / name
        context = {
            **base_context,
            "condition": name,
            "out": str(directory),
            "rotate": str(_rotate_of(condition, scenario)),
        }
        for phase in ("run", "render"):
            spec = condition.get(phase)
            if spec:
                steps.append((f"{name}:{phase}", spec, context, directory / f"{phase}.log"))

    compare = artifacts.get("compare") or {}  # type: ignore[union-attr]
    render_spec = compare.get("render") if isinstance(compare, dict) else None
    if render_spec:
        # The composite consumes the other conditions' artefacts: only build it
        # when every input condition was actually selected in this invocation.
        required = list(compare.get("requires") or [])  # type: ignore[union-attr]
        missing = [name for name in required if name not in dict(
            (step[0].split(":")[0], True) for step in steps
        )]
        if missing:
            print(
                f"skip compare: needs {', '.join(missing)} (not selected / disabled)",
                file=sys.stderr,
            )
        else:
            context = {**base_context, "condition": "compare", "out": str(out_root)}
            steps.append(("compare:render", render_spec, context, out_root / "compare.log"))

    if not steps:
        raise SystemExit("nothing to do: no matching artefact conditions")

    if args.dry_run:
        for label, spec, context, _log in steps:
            print(f"[dry-run] {label}: {shlex.join(build_argv(spec, context))}", file=sys.stderr)
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    failed = 0
    failed_conditions: set[str] = set()
    for label, spec, context, log_path in steps:
        condition_name = label.split(":")[0]
        if condition_name == "compare" and failed_conditions:
            print(
                f"skip {label}: {', '.join(sorted(failed_conditions))} did not complete",
                file=sys.stderr,
            )
            continue
        command = build_argv(spec, context)
        env = os.environ.copy()
        env.update(
            {
                str(key): str(expand(str(value), context))
                for key, value in (spec.get("env") or {}).items()  # type: ignore[union-attr]
            }
        )
        cwd = spec.get("cwd")
        cwd_path = str((REPO / str(cwd)).resolve()) if cwd else str(REPO)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"--- {label} -> {shlex.join(command)}", file=sys.stderr)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            returncode = subprocess.call(
                command, cwd=cwd_path, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        wall_s = time.monotonic() - started
        results.append(
            {
                "step": label,
                "command": command,
                "returncode": returncode,
                "wall_s": round(wall_s, 3),
                "log": str(log_path),
            }
        )
        print(f"    rc={returncode} wall={wall_s:.1f}s", file=sys.stderr)
        if returncode != 0:
            failed += 1
            failed_conditions.add(condition_name)
            print(f"    see {log_path}", file=sys.stderr)

    (out_root / "run.json").write_text(
        json.dumps(
            {
                "mode": "demo",
                "scenario": str(args.scenario),
                "source": str(scenario.get("source", "")),
                "loops": artifacts.get("loops", 1),  # type: ignore[union-attr]
                "steps": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{len(results) - failed}/{len(results)} demo steps ok -> {out_root}",
        file=sys.stderr,
    )
    return 1 if failed else 0


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path(__file__).resolve().with_name("scenarios.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="artefact root (default: results/bench)"
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--conditions", default=None, help="comma separated subset by name"
    )
    parser.add_argument(
        "--gap", type=float, default=5.0, help="seconds to idle between trials"
    )
    parser.add_argument(
        "--interval-ms", type=int, default=100, help="sampler interval"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "run each condition's 'artifacts' block once and render the videos "
            "(never inside a measured trial)"
        ),
    )
    args = parser.parse_args(argv)

    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    if args.demo:
        return run_demo(scenario, args)
    repeats = int(args.repeats or scenario.get("repeats", 3))
    duration = float(args.duration or scenario.get("duration_s", 120))
    out_root = Path(args.out or scenario.get("out_dir") or (REPO / "results/bench"))
    if not out_root.is_absolute():
        out_root = (REPO / out_root).resolve()

    wanted = [name.strip() for name in (args.conditions or "").split(",") if name.strip()]
    conditions: list[tuple[str, dict[str, object]]] = []
    for name, condition in (scenario.get("conditions") or {}).items():
        if wanted and name not in wanted:
            continue
        if not condition.get("enabled", True):
            print(f"skip {name}: disabled in scenario", file=sys.stderr)
            continue
        conditions.append((name, condition))
    if not conditions:
        raise SystemExit("no runnable conditions")

    source = str(scenario.get("source", ""))
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"{len(conditions)} condition(s) x {repeats} repeat(s), {duration:.0f}s each, "
        f"-> {out_root}",
        file=sys.stderr,
    )

    trials: list[dict[str, object]] = []
    for repeat in range(repeats):
        # Rotate the order every repeat so no condition is systematically first.
        order = conditions[repeat % len(conditions):] + conditions[: repeat % len(conditions)]
        for name, condition in order:
            trial_dir = out_root / name / f"r{repeat + 1}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            context = {
                "python": sys.executable,
                "repo": str(REPO),
                "source": source,
                "rotate": str(_rotate_of(condition, scenario)),
                "duration": f"{duration:g}",
                "out": str(trial_dir),
                "condition": name,
                "repeat": str(repeat + 1),
                "loops": str(scenario.get("loops", 1000)),
            }
            command = build_argv(condition, context)
            env = os.environ.copy()
            env.update(
                {
                    str(key): str(expand(str(value), context))
                    for key, value in (condition.get("env") or {}).items()
                }
            )
            cwd = condition.get("cwd")
            cwd_path = str((REPO / str(cwd)).resolve()) if cwd else str(REPO)

            if args.dry_run:
                print(f"[dry-run] {name} r{repeat + 1}: {shlex.join(command)}", file=sys.stderr)
                trials.append({"condition": name, "repeat": repeat + 1, "dry_run": True})
                continue

            before = machine_state()
            log_path = trial_dir / "workload.log"
            print(f"--- {name} r{repeat + 1} -> {trial_dir}", file=sys.stderr)
            t_start = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                workload = subprocess.Popen(
                    command,
                    cwd=cwd_path,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # own process group: sampler can find it
                )
                sampler = subprocess.Popen(
                    [
                        sys.executable,
                        str(SAMPLER),
                        "--out",
                        str(trial_dir / "sampler.csv"),
                        "--pgid",
                        str(workload.pid),
                        "--interval-ms",
                        str(args.interval_ms),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                )
                try:
                    returncode = workload.wait()
                except KeyboardInterrupt:
                    print("interrupted: stopping workload", file=sys.stderr)
                    workload.send_signal(signal.SIGINT)
                    returncode = workload.wait()
                finally:
                    if sampler.poll() is None:
                        sampler.terminate()
                        try:
                            sampler.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            sampler.kill()
                            sampler.wait()

            t_end = time.monotonic()
            meta = {
                "condition": name,
                "repeat": repeat + 1,
                "command": command,
                "cwd": cwd_path,
                "returncode": returncode,
                "t_start_mono": round(t_start, 6),
                "t_end_mono": round(t_end, 6),
                "wall_s": round(t_end - t_start, 3),
                "machine_before": before,
                "machine_after": machine_state(),
            }
            (trial_dir / "run.meta.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
            trials.append(
                {
                    "condition": name,
                    "repeat": repeat + 1,
                    "returncode": returncode,
                    "wall_s": meta["wall_s"],
                }
            )
            print(
                f"    rc={returncode} wall={meta['wall_s']:.1f}s",
                file=sys.stderr,
            )
            if returncode != 0:
                print(f"    see {log_path}", file=sys.stderr)
            if args.gap > 0 and (repeat, name) != (repeats - 1, order[-1][0]):
                time.sleep(args.gap)

    summary = {"scenario": str(args.scenario), "duration_s": duration, "trials": trials}
    (out_root / "run.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    failed = [trial for trial in trials if trial.get("returncode") not in (0, None)]
    print(
        f"{len(trials) - len(failed)}/{len(trials)} trials ok -> {out_root}",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
