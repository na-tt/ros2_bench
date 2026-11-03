#!/usr/bin/env python3
"""Trial orchestrator: expands the scenario matrix and runs it interleaved.

Why interleaved and not "all of A, then all of B": clocks are not pinned and the
garbage collector is not touched, so a run drifts by several percent as the
machine warms, and sequential blocks attribute that drift to whichever cell ran
second. Cells are therefore run in passes -- every cell once per pass -- with a
zig-zag inside blocks and a seeded shuffle across them, so the machine state a
cell inherits is randomised rather than fixed. Every cell runs once per pass, so
a pass yields one trial per cell; the anchor is not measured more often than that
-- it is the *reference* cell, the zero that every delta in the report is
expressed against, and any run of a subset of the matrix has to include it for
deltas from different subsets and sessions to share that zero. What replaces the
duplicate control cell is the spread of those deltas across passes.

Trials are budgeted in frames (that is what the percentiles are computed from)
with a time ceiling derived from that budget (the machine state is a function of
elapsed time), see ``trial_budget``.

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

    bench/run.py --repeats 3 --conditions monolith-720p-n,ros-4node-720p-n
    bench/run.py --dry-run                      # the whole matrix, planned
    bench/run.py --scenario bench/scenarios.json --out results/bench
    bench/run.py --demo --conditions monolith,ros-4node
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable
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


def matrix_conditions(
    scenario: dict[str, object],
) -> list[tuple[str, dict[str, object]]]:
    """Expand ``factors`` x ``design`` into ``(cell name, condition)`` pairs.

    A cell inherits the command template of its chain and gets the resolution's
    source/rotation and the model's engine path injected as ``source`` /
    ``rotate`` / ``model``. Those are the same placeholders a hand-written
    condition already uses, so everything downstream is unchanged -- the matrix
    only decides *which* combinations exist and what their assets are.
    """
    factors = scenario.get("factors") or {}
    if not factors:
        return []  # legacy flat `conditions` scenario
    resolutions = factors.get("resolution") or {}  # type: ignore[union-attr]
    models = factors.get("model") or {}  # type: ignore[union-attr]
    chains = factors.get("chain") or {}  # type: ignore[union-attr]
    if not (resolutions and models and chains):
        raise SystemExit("factors needs 'chain', 'resolution' and 'model' tables")

    design = scenario.get("design") or {}
    cells = design.get("cells") if isinstance(design, dict) else design
    if not cells:
        cells = [
            [chain, resolution, model]
            for chain in chains
            for resolution in resolutions
            for model in models
        ]

    out: list[tuple[str, dict[str, object]]] = []
    seen: set[str] = set()
    for cell in cells:  # type: ignore[union-attr]
        if not isinstance(cell, (list, tuple)) or len(cell) != 3:
            raise SystemExit(f"design cell must be [chain, resolution, model]: {cell!r}")
        chain_name, resolution_name, model_name = (str(part) for part in cell)
        for label, table, value in (
            ("chain", chains, chain_name),
            ("resolution", resolutions, resolution_name),
            ("model", models, model_name),
        ):
            if value not in table:  # type: ignore[operator]
                raise SystemExit(f"design cell names unknown {label}: {value!r}")

        name = f"{chain_name}-{resolution_name}-{model_name}"
        if name in seen:
            raise SystemExit(f"design lists {name} twice")
        seen.add(name)

        resolution_spec = resolutions[resolution_name]  # type: ignore[index]
        model_spec = models[model_name]  # type: ignore[index]
        condition = dict(chains[chain_name])  # type: ignore[index]
        condition["rotate"] = resolution_spec.get("rotate", 0)  # type: ignore[union-attr]
        condition["source"] = resolution_spec.get(  # type: ignore[union-attr]
            "source", scenario.get("source", "")
        )
        condition["model"] = model_spec.get("model", scenario.get("model", ""))  # type: ignore[union-attr]
        condition["cell"] = {
            "chain": chain_name,
            "resolution": resolution_name,
            "model": model_name,
        }
        if resolution_spec.get("fps_scale"):  # type: ignore[union-attr]
            condition["fps_scale"] = resolution_spec["fps_scale"]  # type: ignore[index]
        if resolution_spec.get("clip_frames"):  # type: ignore[union-attr]
            condition["clip_frames"] = resolution_spec["clip_frames"]  # type: ignore[index]
        if model_spec.get("note"):
            condition["note"] = model_spec["note"]  # type: ignore[index]
        out.append((name, condition))
    return out


# ------------------------------------------------------------- trial order
def _blocks(
    names: list[str],
    conditions: dict[str, dict[str, object]],
    cfg: dict[str, object],
) -> list[list[str]]:
    """Group the selected cells into blocks that run back-to-back, adjacent in time.

    Two ways to say it: ``blocks`` names the cells explicitly, or ``block_by``
    names factors and every combination of their values becomes a block. The
    second survives edits to the design, and it is the one that matters here: with
    ``block_by: ["resolution", "model"]`` a monolith and a ROS cell at the same
    resolution and model are neighbours, so the chain delta is a difference
    measured minutes apart instead of hours apart.
    """
    explicit = cfg.get("blocks")
    if explicit:
        blocks = [[str(cell) for cell in block] for block in explicit]  # type: ignore[union-attr]
        placed = {cell for block in blocks for cell in block}
        unknown = sorted(placed - set(names))
        if unknown:
            print(
                f"order: blocks name cells not in this run: {', '.join(unknown)}",
                file=sys.stderr,
            )
        # Anything the blocks forgot still gets measured, as its own singleton.
        blocks.extend([[name] for name in names if name not in placed])
        return [block for block in blocks if block]

    keys = [str(key) for key in (cfg.get("block_by") or []) if key]
    if keys:
        groups: dict[tuple[str, ...], list[str]] = {}
        for name in names:
            cell = conditions[name].get("cell") or {}
            key = tuple(str(cell.get(k)) for k in keys)  # type: ignore[union-attr]
            groups.setdefault(key, []).append(name)
        return [groups[key] for key in sorted(groups)]
    return [list(names)]


def build_schedule(
    names: list[str],
    conditions: dict[str, dict[str, object]],
    repeats: int,
    order_cfg: dict[str, object] | None,
) -> tuple[list[list[str]], dict[str, object]]:
    """Decide who runs when, and audit the result.

    A cyclic Latin square is exactly right while it is affordable: it balances
    every slot position, but it needs ``repeats % n == 0``, it never balances
    *predecessors* (cell i is always followed by i+1), and at n = 18 it costs 18
    passes. So it is used only where it is cheap and correct (n <= 6, balanced),
    and the matrix uses blocks instead: block order is shuffled per pass with a
    logged seed and the direction inside a block alternates, which balances
    short-term drift for the comparisons that matter and turns any residual
    imbalance into variance the report can see rather than a bias it cannot.
    """
    cfg = order_cfg if isinstance(order_cfg, dict) else {}
    policy = str(cfg.get("policy", "auto"))
    n = len(names)
    square = policy == "latin" or (policy == "auto" and n <= 6 and repeats % n == 0)

    blocks: list[list[str]] = []
    if square:
        schedule = [
            [names[(repeat + index) % n] for index in range(n)]
            for repeat in range(repeats)
        ]
    else:
        blocks = _blocks(names, conditions, cfg)
        rng = random.Random(cfg.get("seed"))  # type: ignore[arg-type]
        schedule = []
        for repeat in range(repeats):
            block_order = list(blocks)
            rng.shuffle(block_order)
            flat: list[str] = []
            for block in block_order:
                flat.extend(block if repeat % 2 == 0 else list(reversed(block)))
            schedule.append(flat)

    slots: dict[str, Counter] = {name: Counter() for name in names}
    predecessors: dict[str, Counter] = {name: Counter() for name in names}
    warnings: list[str] = []
    for pass_order in schedule:
        if sorted(pass_order) != sorted(names):
            missing = sorted(set(names) - set(pass_order))
            warnings.append(f"pass is not a full pass: missing {', '.join(missing)}")
        for position, name in enumerate(pass_order):
            slots[name][position] += 1
            predecessors[name][pass_order[position - 1] if position else None] += 1

    for name in names:
        if sum(slots[name].values()) != len(schedule):
            warnings.append(f"{name} did not run once per pass")
            continue
        if len(schedule) > 1:
            common, count = predecessors[name].most_common(1)[0]
            if count == len(schedule) and common is not None:
                warnings.append(f"{name} always followed {common}")

    meta: dict[str, object] = {
        "policy": "latin" if square else "blocks",
        "seed": cfg.get("seed"),
        "blocks": blocks,
        "schedule": schedule,
        "slots": {name: dict(sorted(counts.items())) for name, counts in slots.items()},
        "predecessors": {
            name: {str(key): value for key, value in counts.items()}
            for name, counts in predecessors.items()
        },
        "warnings": warnings,
    }
    return schedule, meta


# --------------------------------------------------------------- trial budget
DEFAULT_FPS_HINT = 30.0


def _fps_hint(condition: dict[str, object], scenario: dict[str, object]) -> float:
    """Best available guess of this cell's frame rate, only ever used to size a box.

    Order: an explicit per-cell override, the scenario's chain/model table scaled
    by the resolution, then a flat default. Under-estimating is harmless (the
    trial runs a little longer than needed); over-estimating truncates the
    analysis window, so the table is deliberately pessimistic and the report
    says so when a window came up short.
    """
    if condition.get("fps_hint"):
        return float(condition["fps_hint"])  # type: ignore[arg-type]
    trial = scenario.get("trial") or {}
    table = trial.get("fps_hint") if isinstance(trial, dict) else None
    cell = condition.get("cell") or {}
    if isinstance(table, dict) and cell:
        row = table.get(cell.get("chain")) or {}
        base = row.get(cell.get("model")) if isinstance(row, dict) else None
        if base:
            return float(base) * float(condition.get("fps_scale", 1.0))
    if isinstance(table, (int, float)):
        return float(table)
    return DEFAULT_FPS_HINT


def trial_budget(
    condition: dict[str, object],
    scenario: dict[str, object],
    *,
    settle: float | None,
    analysis_frames: int | None,
    duration: float | None,
) -> dict[str, object]:
    """Turn ``settle_s`` + ``analysis_frames`` into what a front-end needs.

    The design is stated in frames (that is what the percentiles are computed
    from), but the stop condition is a time ceiling computed from those frames,
    because the machine state that dominates trial-to-trial variance is a
    function of elapsed time. ``max_frames`` is a generous hard cap behind the
    ceiling: it bounds a run whose frame rate was guessed too low, and it also
    keeps the replay count deterministic.
    """
    trial = scenario.get("trial") or {}
    trial = trial if isinstance(trial, dict) else {}
    fps = _fps_hint(condition, scenario)
    settle_s = float(
        settle
        if settle is not None
        else condition.get("settle_s") or trial.get("settle_s", 30.0)
    )
    frames = int(
        analysis_frames
        if analysis_frames is not None
        else condition.get("analysis_frames") or trial.get("analysis_frames", 2500)
    )
    # 10 % headroom: the analyser drops the settle seconds and then takes exactly
    # `frames` frames, so a trial that is exactly on the boundary would be short.
    ceiling = round(settle_s + 1.1 * frames / fps, 1)
    cap = int(math.ceil(1.5 * settle_s * fps)) + int(math.ceil(1.2 * frames))
    clip_frames = int(condition.get("clip_frames") or scenario.get("clip_frames") or 0)
    loops = (
        int(math.ceil(cap / clip_frames)) + 1
        if clip_frames
        else int(scenario.get("loops", 1000))
    )
    return {
        "fps_hint": round(fps, 2),
        "settle_s": round(settle_s, 2),
        "analysis_frames": frames,
        "max_frames": int(condition.get("max_frames") or cap),
        "duration_s": float(
            duration if duration is not None else condition.get("duration_s") or ceiling
        ),
        "loops": int(condition.get("loops") or loops),
    }


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
            # The composite renders the same footage as the panels it merges.
            anchor_spec = conditions.get(required[0]) if required else None  # type: ignore[union-attr]
            context = {
                **base_context,
                "condition": "compare",
                "out": str(out_root),
                "rotate": str(_rotate_of(anchor_spec or {}, scenario)),
            }
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
    parser.add_argument(
        "--settle",
        type=float,
        default=None,
        help="seconds of each trial treated as settling and excluded from analysis",
    )
    parser.add_argument(
        "--analysis-frames",
        type=int,
        default=None,
        help="frames of analysis window after the settle period (per-trial budget)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="override the computed time ceiling for every cell",
    )
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
        "--seed", type=int, default=None, help="seed for the per-pass block shuffle"
    )
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
    out_root = Path(args.out or scenario.get("out_dir") or (REPO / "results/bench"))
    if not out_root.is_absolute():
        out_root = (REPO / out_root).resolve()

    wanted = [name.strip() for name in (args.conditions or "").split(",") if name.strip()]
    conditions: list[tuple[str, dict[str, object]]] = []
    generated = matrix_conditions(scenario)
    pool: Iterable[tuple[str, dict[str, object]]] = (
        generated
        if generated
        else [
            (str(name), condition)  # type: ignore[misc]
            for name, condition in (scenario.get("conditions") or {}).items()  # type: ignore[union-attr]
        ]
    )
    known: set[str] = set()
    overrides = scenario.get("cells") or {}
    for name, condition in pool:
        known.add(name)
        if wanted and name not in wanted:
            continue
        if not condition.get("enabled", True):
            print(f"skip {name}: disabled in scenario", file=sys.stderr)
            continue
        # Per-cell knobs (fps_hint, settle_s, analysis_frames, max_frames,
        # duration_s, loops) live in `cells`, away from the factor tables.
        merged = dict(condition)
        if isinstance(overrides, dict):
            merged.update(overrides.get(name) or {})
        if merged.get("enabled", True) is False:
            print(f"skip {name}: disabled in cells", file=sys.stderr)
            continue
        conditions.append((name, merged))
    if wanted:
        missing = [name for name in wanted if name not in known]
        if missing:
            raise SystemExit(
                f"unknown condition(s): {', '.join(missing)} "
                f"(known: {', '.join(sorted(known))})"
            )
    if not conditions:
        raise SystemExit("no runnable conditions")

    source = str(scenario.get("source", ""))
    out_root.mkdir(parents=True, exist_ok=True)

    budgets = {
        name: trial_budget(
            condition,
            scenario,
            settle=args.settle,
            analysis_frames=args.analysis_frames,
            duration=args.duration,
        )
        for name, condition in conditions
    }
    ceilings = [float(budgets[name]["duration_s"]) for name, _ in conditions]
    settles = {float(budgets[name]["settle_s"]) for name, _ in conditions}
    windows = {int(budgets[name]["analysis_frames"]) for name, _ in conditions}
    print(
        f"{len(conditions)} cell(s) x {repeats} pass(es); analysis window "
        f"{', '.join(str(n) for n in sorted(windows))} frames after "
        f"{', '.join(f'{s:g}' for s in sorted(settles))}s settle; "
        f"time ceiling {min(ceilings):.0f}-{max(ceilings):.0f}s -> {out_root}",
        file=sys.stderr,
    )

    order_cfg = dict(scenario.get("order") or {})
    if args.seed is not None:
        order_cfg["seed"] = args.seed
    schedule, order_meta = build_schedule(
        [name for name, _ in conditions],
        dict(conditions),
        repeats,
        order_cfg,
    )
    if order_meta["policy"] == "blocks":
        print(
            f"order: blocks ({len(order_meta['blocks'])}), seed {order_meta['seed']}",
            file=sys.stderr,
        )
    else:
        print("order: cyclic Latin square (balanced)", file=sys.stderr)
    for warning in order_meta["warnings"]:  # type: ignore[union-attr]
        print(f"order: warning: {warning}", file=sys.stderr)

    by_name = dict(conditions)
    trials: list[dict[str, object]] = []
    for repeat, order in enumerate(schedule):
        print(f"  pass {repeat + 1}: {' '.join(order)}", file=sys.stderr)
        for name in order:
            condition = by_name[name]
            trial_dir = out_root / name / f"r{repeat + 1}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            budget = budgets[name]
            context = {
                "python": sys.executable,
                "repo": str(REPO),
                "source": str(condition.get("source", source)),
                "model": str(condition.get("model", "")),
                "rotate": str(_rotate_of(condition, scenario)),
                "duration": f"{float(budget['duration_s']):g}",
                "frames": str(budget["max_frames"]),
                "out": str(trial_dir),
                "condition": name,
                "repeat": str(repeat + 1),
                "loops": str(budget["loops"]),
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
                "budget": budget,
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
            if args.gap > 0 and (repeat, name) != (repeats - 1, order[-1]):
                time.sleep(args.gap)

    summary = {
        "scenario": str(args.scenario),
        "order": order_meta,
        "budgets": budgets,
        "trials": trials,
    }
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
