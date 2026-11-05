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

    bench/run.py --repeats 3 --conditions monolith-720p-n,ros-N4-p2c1-720p-n
    bench/run.py --dry-run                      # the whole matrix, planned
    bench/run.py --design full                  # every combination: 297 cells
    bench/run.py --scenario bench/scenarios.json --out results/bench
    bench/run.py --demo --conditions monolith,ros-N4-p2c1

``--design`` picks which cells exist: ``scenario`` (default) measures the pruned
``design.cells`` list -- the cheap matrix, run as subsets -- while ``full``
expands every chain x resolution x model x window combination. The full matrix is
a long haul (297 cells, 891 trials at three passes) and is meant for a dedicated
session with its own ``--out``; ``--conditions`` still selects a subset of it, so
a full-design session can be done in pieces as long as every piece contains the
anchor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
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


def chain_windows(chain_spec: dict[str, object]) -> list[int]:
    """The frames-in-flight values a chain may be measured at.

    Declared per chain factor as ``windows`` (``[1]`` for the monolith, which is
    one thread by construction). It is the domain of the fourth coordinate: the
    pruned ``design.cells`` list picks points out of it, ``--design full``
    expands all of it. A design cell outside the domain is rejected, so the two
    cannot drift apart.
    """
    windows = chain_spec.get("windows") or [1]
    values = sorted({int(window) for window in windows})  # type: ignore[union-attr]
    if not values or values[0] < 1:
        raise SystemExit(f"chain 'windows' must be >= 1, got {windows!r}")
    return values


def full_design(
    resolutions: dict[str, object],
    models: dict[str, object],
    chains: dict[str, object],
    reference: object = None,
) -> list[list[object]]:
    """Every combination the factors admit: chain x resolution x model x window.

    This is what ``--design full`` (and a scenario with no ``design.cells``)
    measures -- 9 monolith cells plus 8 ROS shapes x 3 resolutions x 3 models x
    4 windows = 297 here. It is ordered ``resolution -> model -> chain ->
    window`` on purpose: ``order.block_by: [resolution, model]`` then puts each
    (resolution, model) block's cells in *family* order, so a chain's window
    ladder stays contiguous inside the block. ``reference`` (the scenario's
    ``reference_chain``) is placed first among the chains, which puts the
    headline pair -- the monolith and the four-node chain, same footage, same
    model, window of one -- in adjacent slots of every block instead of ten
    trials apart.
    """
    order = [
        name for name, spec in chains.items() if isinstance(spec, dict)
    ]  # skip the table's `note`
    if reference in order:
        order.remove(str(reference))
        # Immediately after the first declared chain (the monolith), so the
        # headline pair shares a slot boundary in every block.
        order.insert(1 if order else 0, str(reference))
    return [
        [chain_name, resolution_name, model_name, inflight]
        for resolution_name, resolution_spec in resolutions.items()
        if isinstance(resolution_spec, dict)
        for model_name, model_spec in models.items()
        if isinstance(model_spec, dict)
        for chain_name in order
        for inflight in chain_windows(chains[chain_name])  # type: ignore[arg-type]
    ]


def matrix_conditions(
    scenario: dict[str, object], design: str = "scenario"
) -> list[tuple[str, dict[str, object]]]:
    """Expand ``factors`` x ``design`` into ``(cell name, condition)`` pairs.

    ``design`` selects the cell list: ``scenario`` (the pruned ``design.cells``
    in ``scenarios.json``, the cheap matrix every ordinary run measures) or
    ``full`` (every factor combination, windows included -- ``full_design``).
    An absent ``design.cells`` means ``full`` too, so the file can say "all of
    it" by saying nothing.

    A cell inherits the command template of its chain and gets the resolution's
    source/rotation and the model's engine path injected as ``source`` /
    ``rotate`` / ``model``. Those are the same placeholders a hand-written
    condition already uses, so everything downstream is unchanged -- the matrix
    only decides *which* combinations exist and what their assets are.

    A fourth coordinate, ``inflight``, is optional and defaults to 1. It is not
    a shape (the process plan does not change) but it is part of the cell's
    identity: the same chain, resolution and model at two windows are two
    different measurements of the same work, and the name has to say which. Its
    domain comes from the chain factor's ``windows`` list.

    The conditioning chain is **not** a coordinate: it is a constant of the
    experiment, declared once under ``conditioning`` and injected into every
    command. Every front-end, at every shape/resolution/model/window, has to
    condition the frame identically, so the only axes a cell name may encode are
    architecture, resolution, model and frames in flight.
    """
    factors = scenario.get("factors") or {}
    if not factors:
        return []  # legacy flat `conditions` scenario
    ops = conditioning_ops(scenario)
    resolutions = factors.get("resolution") or {}  # type: ignore[union-attr]
    models = factors.get("model") or {}  # type: ignore[union-attr]
    chains = factors.get("chain") or {}  # type: ignore[union-attr]
    if not (resolutions and models and chains):
        raise SystemExit("factors needs 'chain', 'resolution' and 'model' tables")
    if design not in ("scenario", "full"):
        raise SystemExit(f"unknown design {design!r} (expected scenario or full)")

    block = scenario.get("design") or {}
    cells = block.get("cells") if isinstance(block, dict) else block
    if design == "full":
        cells = None
    if not cells:
        cells = full_design(
            resolutions,
            models,
            chains,
            scenario.get("reference_chain"),
        )

    out: list[tuple[str, dict[str, object]]] = []
    seen: set[str] = set()
    for cell in cells:  # type: ignore[union-attr]
        if not isinstance(cell, (list, tuple)) or len(cell) not in (3, 4):
            raise SystemExit(
                "design cell must be [chain, resolution, model] or "
                f"[chain, resolution, model, inflight]: {cell!r}"
            )
        chain_name, resolution_name, model_name = (str(part) for part in cell[:3])
        inflight = int(cell[3]) if len(cell) == 4 else 1
        if inflight < 1:
            raise SystemExit(f"inflight must be >= 1, got {inflight} in {cell!r}")
        for label, table, value in (
            ("chain", chains, chain_name),
            ("resolution", resolutions, resolution_name),
            ("model", models, model_name),
        ):
            if value not in table:  # type: ignore[operator]
                raise SystemExit(f"design cell names unknown {label}: {value!r}")
        windows = chain_windows(chains[chain_name])  # type: ignore[index]
        if inflight not in windows:
            raise SystemExit(
                f"design cell {chain_name}-{resolution_name}-{model_name} wants "
                f"inflight={inflight}, but {chain_name} declares windows "
                f"{windows}"
            )

        name = f"{chain_name}-{resolution_name}-{model_name}"
        if inflight != 1:
            name = f"{name}-i{inflight}"
        if name in seen:
            raise SystemExit(f"design lists {name} twice")
        seen.add(name)

        resolution_spec = resolutions[resolution_name]  # type: ignore[index]
        model_spec = models[model_name]  # type: ignore[index]
        # ``windows`` is metadata about the factor, not a field of the command.
        condition = {
            key: value
            for key, value in chains[chain_name].items()  # type: ignore[union-attr]
            if key != "windows"
        }
        condition["rotate"] = resolution_spec.get("rotate", 0)  # type: ignore[union-attr]
        condition["source"] = resolution_spec.get(  # type: ignore[union-attr]
            "source", scenario.get("source", "")
        )
        condition["model"] = model_spec.get("model", scenario.get("model", ""))  # type: ignore[union-attr]
        condition["cell"] = {
            "chain": chain_name,
            "resolution": resolution_name,
            "model": model_name,
            "inflight": inflight,
        }
        condition["inflight"] = inflight
        #: The one conditioning chain, in the form the ROS launch takes. The
        #: monolith template spells the same ops as repeated `--pre`; both are
        #: checked against this list before anything launches.
        condition["pre_ops"] = ";".join(ops)
        if resolution_spec.get("fps_scale"):  # type: ignore[union-attr]
            condition["fps_scale"] = resolution_spec["fps_scale"]  # type: ignore[index]
        if resolution_spec.get("clip_frames"):  # type: ignore[union-attr]
            condition["clip_frames"] = resolution_spec["clip_frames"]  # type: ignore[index]
        if resolution_spec.get("frame_mb"):  # type: ignore[union-attr]
            condition["frame_mb"] = resolution_spec["frame_mb"]  # type: ignore[index]
        if model_spec.get("note"):
            condition["note"] = model_spec["note"]  # type: ignore[index]
        check_cell_shape(name, condition)
        check_cell_conditioning(name, condition, ops)
        out.append((name, condition))
    return out


def conditioning_ops(scenario: dict[str, object]) -> list[str]:
    """The experiment's one conditioning chain, in canonical order.

    Declared once in ``scenarios.json`` under ``conditioning.ops``. It is not a
    factor: a cell that ran a different chain would not be a different point on
    an axis, it would be a different experiment, and the records cannot tell the
    two apart (stage 2 is reported as a sum, so an empty chain reads ~0 and
    looks faster).
    """
    block = scenario.get("conditioning") or {}
    ops = block.get("ops") if isinstance(block, dict) else block
    return [str(op) for op in (ops or [])]  # type: ignore[union-attr]


# ------------------------------------------------------------ cell naming
#: `ros-N{x}-p{p}c{c}`: x processes, p pixel-carrying hops, c candidate hops.
#: The two halves of that claim are redundant on purpose (x == p + c + 1) and
#: both are checkable before anything launches, which is the only thing keeping
#: the names honest -- the report is built from them.
SHAPE_NAME = re.compile(r'(?:^|-)N(\d+)-p(\d+)c(\d+)(?:-|$)')

#: `-i{D}`: frames in flight, appended only when D > 1 so the window of one --
#: the demand-driven chain every earlier cell measured -- keeps its old name.
INFLIGHT_NAME = re.compile(r'(?:^|-)i(\d+)(?:-|$)')


def _command_text(condition: dict[str, object]) -> str:
    """The condition's command as one string, for keyword sniffing."""
    shell = condition.get("shell")
    if shell:
        return str(shell)
    return " ".join(str(part) for part in condition.get("cmd") or [])  # type: ignore[union-attr]


def check_cell_shape(name: str, condition: dict[str, object]) -> None:
    """Reject a cell whose name and command disagree about its shape.

    A cell called ``ros-N12-p8c3-720p-n`` claims twelve processes, eight hops
    carrying pixels and three carrying candidates. ``N == p + c + 1`` is
    arithmetic; the group count in the ``chain:=`` string is the graph itself.
    Both are checked here, before a single process is started, because a
    mislabelled cell is a mistake that survives all the way into the report.

    The window (``-i{D}``) is checked for **every** cell, not only the ones whose
    shape is in the name: the frames-in-flight curve is indexed by that token, so
    a ``-i4`` cell that launches a window of one is misfiled silently. A ROS cell
    must always pass ``chain:=``, for the same reason: the launch default is a
    shape, and a cell that inherits it cannot be validated against its name.
    """
    text = _command_text(condition)
    check_cell_inflight(name, condition, text)
    match = SHAPE_NAME.search(name)
    declared = None
    if match:
        nodes, pixel, candidate = (int(part) for part in match.groups())
        if nodes != pixel + candidate + 1:
            raise SystemExit(
                f"{name}: N{nodes} contradicts p{pixel}c{candidate} "
                f"(N should be p+c+1 = {pixel + candidate + 1})"
            )
        declared = nodes
    shape = re.search(r'chain:=([^ ]+)', text)
    if shape is None:
        if "ros2 launch" in text:
            raise SystemExit(
                f"{name}: a ROS cell must pass chain:=... so the shape it runs "
                "is the shape it declares"
            )
        return
    value = shape.group(1).rstrip('\'"')
    groups = [group for group in value.split('|') if group]
    if declared is not None and len(groups) != declared:
        raise SystemExit(
            f"{name}: chain {value!r} has {len(groups)} process(es) "
            f"but the name says N{declared}"
        )


def check_cell_conditioning(
    name: str, condition: dict[str, object], ops: list[str]
) -> None:
    """Reject a cell that does not run the experiment's conditioning chain.

    The chain is a constant, so there is nothing to encode in the name -- which
    is exactly why it needs a check: a cell that silently skipped stage 2 is
    still a valid-looking name and still produces records, and because
    ``pre_ms`` is a sum, the missing ops read as *better* performance. Both
    front-ends are checked in the form they actually take: repeated ``--pre`` for
    the monolith, ``pre_ops:=`` for the launch file.
    """
    if not ops:
        raise SystemExit(
            "scenario has no 'conditioning.ops': the conditioning chain is a "
            "constant of the experiment, not an optional knob"
        )
    text = _command_text(condition).replace("{pre_ops}", ";".join(ops))
    if "ros2 launch" in text:
        match = re.search(r'pre_ops:=([^ ]+)', text)
        if match is None:
            raise SystemExit(
                f"{name}: a ROS cell must pass pre_ops:=... "
                f"(the chain has {len(ops)} ops)"
            )
        actual = [op for op in match.group(1).rstrip('\'"').split(';') if op]
    else:
        cmd = condition.get("cmd")
        if not isinstance(cmd, list):
            raise SystemExit(
                f"{name}: cannot check the conditioning chain of a command "
                "that is not an argv list"
            )
        actual = []
        for index, part in enumerate(cmd):
            if str(part) != "--pre":
                continue
            if index + 1 >= len(cmd):
                raise SystemExit(f"{name}: --pre with no value")
            actual.append(str(cmd[index + 1]))
    if actual != ops:
        raise SystemExit(
            f"{name}: conditioning chain is {actual!r}, but the experiment's "
            f"constant is {ops!r}"
        )


def check_cell_inflight(
    name: str, condition: dict[str, object], text: str
) -> None:
    """Reject a cell whose name and command disagree about the window.

    Same rule as the shape, for the same reason: a cell called ``...-i4`` that
    launches ``inflight:=1`` (or the reverse) produces a number that is filed
    under the wrong window, and nothing downstream can tell.
    """
    declared = int(condition.get("inflight", 1))
    named = INFLIGHT_NAME.search(name)
    named_value = int(named.group(1)) if named else 1
    # The command is a template at this point: `inflight:={inflight}` has not
    # been expanded yet, and the placeholder is the truthful form (it is what
    # the trial loop will substitute).
    text = text.replace("{inflight}", str(declared))
    passed = re.search(r'inflight:=(["\']?)(\d+)', text)
    passed_value = int(passed.group(2)) if passed else None
    if passed_value is None and "ros2 launch" in text and declared != 1:
        raise SystemExit(
            f"{name}: a windowed ROS cell must pass inflight:={declared} "
            "(the launch default is 1, so the name would be a lie)"
        )
    if passed_value is not None and passed_value != declared:
        raise SystemExit(
            f"{name}: command passes inflight:={passed_value} but the cell "
            f"declares inflight={declared}"
        )
    if declared != 1 and named is None:
        raise SystemExit(
            f"{name}: a windowed cell must say so in its name (expected "
            f"-i{declared})"
        )
    if named_value != declared:
        raise SystemExit(
            f"{name}: name says i{named_value} but the cell declares "
            f"inflight={declared}"
        )


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

#: Stages that emit a BGR frame onto the wire, and stages that emit
#: ``Candidates`` (kilobytes). ``coord`` is the terminal stage: it publishes
#: nothing, so it contributes no hop.
PIXEL_PUBLISHERS = ("acq", "pre")
CANDIDATE_PUBLISHERS = ("infer", "nms", "filter")

#: The single-process pipeline, as one "group" of stages.
MONOLITH_STAGES = ("acq", "pre", "infer", "post")


def _stage_ms(
    stage: str, cell: dict[str, object], scenario: dict[str, object]
) -> float | None:
    """Cost of one pipeline stage, from ``fps_model.stage_ms``.

    The table is keyed by whichever factor the stage scales with: resolution for
    decode/conditioning/postprocessing, model for the engine.
    """
    model = scenario.get("fps_model") or {}
    table = model.get("stage_ms") if isinstance(model, dict) else None
    row = table.get(stage) if isinstance(table, dict) else None
    if not isinstance(row, dict):
        return None
    for key in (cell.get("resolution"), cell.get("model")):
        if key is None:
            continue
        value = row.get(str(key))
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _hop_ms(
    publisher: str, condition: dict[str, object], scenario: dict[str, object]
) -> float | None:
    """Cost of one hop, priced by the payload its publisher emits.

    Same shape as the fitted transport model in ``bench/README.md``: a fixed
    cost per hop plus a per-megabyte term for a frame. A candidate hop carries
    kilobytes, so it is a one-number entry.
    """
    model = scenario.get("fps_model") or {}
    hop = model.get("hop_ms") if isinstance(model, dict) else None
    if not isinstance(hop, dict):
        return None
    if publisher in CANDIDATE_PUBLISHERS:
        value = hop.get("candidate")
        return float(value) if isinstance(value, (int, float)) else None
    if publisher in PIXEL_PUBLISHERS:
        fixed = hop.get("fixed")
        per_mb = hop.get("per_mb")
        frame_mb = condition.get("frame_mb")
        numbers = (fixed, per_mb, frame_mb)
        if not all(isinstance(number, (int, float)) for number in numbers):
            return None
        return float(fixed) + float(per_mb) * float(frame_mb)  # type: ignore[arg-type]
    return None  # terminal stage: nothing published


def _chain_groups(condition: dict[str, object]) -> list[list[str]] | None:
    """The process plan, read off the ``chain:=`` string a ROS cell passes.

    ``acq|pre|infer|post`` is four processes, ``acq|pre+infer+post`` is two. This
    is the same string ``check_cell_shape`` validates against the cell name, so a
    shape that can be launched can also be costed.
    """
    match = re.search(r'chain:=([^ "\']+)', _command_text(condition))
    if match is None:
        return None
    groups = [group.split("+") for group in match.group(1).split("|") if group]
    return groups or None


def _residencies(
    condition: dict[str, object], scenario: dict[str, object]
) -> list[float] | None:
    """Per-process residency of one frame, in milliseconds.

    Each process pays for the stages it runs plus the hop out of it (the
    serialisation happens in its thread) plus a small per-process overhead, and a
    hop costs what its publisher's payload costs. One subtlety: ``pre`` is the
    *whole* conditioning chain, so a stage-2 split (``chain:=acq|pre|pre|...`` or
    ``pre_groups:=2,5``) divides that cost across the groups that run it --
    otherwise the twelve-process shape would be charged seven full chains.

    ``None`` whenever a number the model needs is missing: a guess is worse than
    falling back to the flat default.
    """
    groups = _chain_groups(condition)
    if groups is None:
        if "ros2 launch" in _command_text(condition):
            return None
        groups = [list(MONOLITH_STAGES)]
    cell = condition.get("cell") or {}
    if not isinstance(cell, dict):
        return None
    model = scenario.get("fps_model") or {}
    if not isinstance(model, dict):
        model = {}
    node_ms = float(model.get("node_ms", 0.0))  # type: ignore[arg-type]
    pre_total = _stage_ms("pre", cell, scenario)
    pre_groups = sum(group.count("pre") for group in groups)
    out: list[float] = []
    for index, group in enumerate(groups):
        total = node_ms
        for stage in group:
            if stage == "pre" and pre_groups > 1 and pre_total is not None:
                total += pre_total / pre_groups
                continue
            cost = _stage_ms(stage, cell, scenario)
            if cost is None:
                return None
            total += cost
        if index + 1 < len(groups):
            hop = _hop_ms(group[-1], condition, scenario)
            if hop is None:
                return None
            total += hop
        out.append(total)
    return out


def _fps_from_model(
    condition: dict[str, object], scenario: dict[str, object]
) -> float | None:
    """Derive a frame-rate hint from the closed-loop model.

    *``X = min(D / sum(residency), 1 / max(residency))``*: with ``D`` frames in
    flight the loop cannot go faster than the frame that takes longest, and the
    knee is at ``D* = sum(residency) / max(residency)``. Two corrections keep the
    number on the safe side, because only an *over*-estimate costs data (the
    ceiling truncates the window; an under-estimate just runs a little longer):
    a ``pessimism`` factor on the rate, and ``window_gain_cap`` on what a window
    may buy -- the one window gain measured on this board was 1.5x (op-free at
    180p, 29.5 -> 44.2 fps), well under what the closed model allows, because
    queueing and DDS overhead grow with the window too.

    This is what makes ``--design full`` measurable: 297 cells cannot all have a
    hand-measured hint, and an *optimistic* hint is the failure that costs data
    (the ceiling truncates the window), while an under-estimate only runs a
    little longer. Cells the ``trial.fps_hint`` table does cover never reach
    here -- a measured number always wins over a derived one.
    """
    model = scenario.get("fps_model") or {}
    if not isinstance(model, dict) or not model.get("stage_ms"):
        return None
    residencies = _residencies(condition, scenario)
    if not residencies or min(residencies) <= 0:
        return None
    pessimism = float(model.get("pessimism", 0.6))  # type: ignore[arg-type]
    total = sum(residencies) / 1000.0
    gain = _window_gain(condition, scenario)
    return max(0.5, round(pessimism * gain / total, 2))


def _window_gain(
    condition: dict[str, object], scenario: dict[str, object]
) -> float:
    """How much throughput a window of ``D`` can add over the same cell at 1.

    From the same closed loop as ``_fps_from_model``: ``X(D) / X(1)`` collapses to
    ``min(D, D*)`` with the knee ``D* = sum(residency) / max(residency)`` -- no
    window buys more than the loop's slowest frame allows, which is why the
    process count is not the knee. Used to carry a *measured* window-of-one hint
    to a windowed cell of the same family instead of measuring every rung.
    """
    inflight = int(condition.get("inflight", 1))
    if inflight <= 1:
        return 1.0
    residencies = _residencies(condition, scenario)
    if not residencies or min(residencies) <= 0:
        return 1.0
    knee = sum(residencies) / max(residencies)
    model = scenario.get("fps_model") or {}
    cap = float(model.get("window_gain_cap", 1.5)) if isinstance(model, dict) else 1.5
    return max(1.0, min(float(inflight), knee, cap))


def _fps_hint(condition: dict[str, object], scenario: dict[str, object]) -> float:
    """Best available guess of this cell's frame rate, only ever used to size a box.

    Order: an explicit per-cell override, the scenario's
    ``chain -> resolution -> model`` table (or the older ``chain -> model`` form
    with a ``fps_scale`` on the resolution), the derived closed-loop model
    (``_fps_from_model``), then a flat default. The first two are measurements
    with the margin already taken off; the third is a derivation used where no
    measurement exists yet (every cell of ``--design full`` that is not in the
    pruned matrix). An over-estimate shortens a trial's ceiling until the window
    comes up short (``analyze.py`` warns), while under-estimating only costs a
    few seconds of extra run -- so the fallbacks are deliberately conservative.
    See ``trial_budget`` for why a wrong hint also puts the frame cap in the
    wrong place.
    """
    if condition.get("fps_hint"):
        return float(condition["fps_hint"])  # type: ignore[arg-type]
    trial = scenario.get("trial") or {}
    table = trial.get("fps_hint") if isinstance(trial, dict) else None
    cell = condition.get("cell") or {}
    if isinstance(table, dict) and cell:
        row = table.get(cell.get("chain")) or {}
        # Nested form: the resolution is a level of the table, because the ops
        # and the model do not scale with pixels the same way -- one scalar per
        # chain cannot describe both ends of the resolution axis.
        resolution = cell.get("resolution")
        if isinstance(row, dict) and isinstance(row.get(resolution), dict):
            row = row[resolution]  # type: ignore[index]
        base = row.get(cell.get("model")) if isinstance(row, dict) else None
        if base:
            # The table holds window-of-one measurements, so a windowed cell is
            # scaled by the loop's gain over one rather than inheriting the
            # demand-driven rate unchanged.
            return round(
                float(base)
                * float(condition.get("fps_scale", 1.0))
                * _window_gain(condition, scenario),
                2,
            )
    if isinstance(table, (int, float)):
        return float(table)
    derived = _fps_from_model(condition, scenario)
    if derived is not None:
        return derived
    return DEFAULT_FPS_HINT


def _settle_s(
    condition: dict[str, object], trial: dict[str, object], override: float | None
) -> float:
    """Settling seconds for one trial.

    A windowed cell (`-iD`) is a different power and thermal state from the
    demand-driven one, so the scenario can declare a longer ramp for it
    (``trial.settle_s_windowed``) instead of every window cell repeating the same
    override in ``cells``. Explicit per-cell overrides still win, and the CLI
    wins over everything.
    """
    if override is not None:
        return float(override)
    if condition.get("settle_s"):
        return float(condition["settle_s"])  # type: ignore[arg-type]
    inflight = int(condition.get("inflight", 1))
    windowed = trial.get("settle_s_windowed")
    if inflight > 1 and isinstance(windowed, (int, float)):
        return float(windowed)
    return float(trial.get("settle_s", 30.0))  # type: ignore[arg-type]


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
    function of elapsed time. ``max_frames`` is a hard cap *behind* the ceiling and
    is sized to cover an fps overshoot: the frames the settle period eats at twice
    the hinted rate, plus 50 % slack on the window. It is not the budget -- the
    ceiling is -- but it must never bind first, or a cell that is simply faster
    than its hint would be truncated mid-window. (Measured: at the old 1.5x/1.2x
    sizing, ``monolith-180p-n`` cleared 2500 window frames with 20 frames to
    spare, i.e. a truncation that would have gone unnoticed.)
    """
    trial = scenario.get("trial") or {}
    trial = trial if isinstance(trial, dict) else {}
    fps = _fps_hint(condition, scenario)
    settle_s = _settle_s(condition, trial, settle)
    frames = int(
        analysis_frames
        if analysis_frames is not None
        else condition.get("analysis_frames") or trial.get("analysis_frames", 2500)
    )
    # 10 % headroom: the analyser drops the settle seconds and then takes exactly
    # `frames` frames, so a trial that is exactly on the boundary would be short.
    ceiling = round(settle_s + 1.1 * frames / fps, 1)
    cap = int(math.ceil(2.0 * settle_s * fps)) + int(math.ceil(1.5 * frames))
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
        "pre_ops": ";".join(conditioning_ops(scenario)),
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
        "--design",
        choices=("scenario", "full"),
        default="scenario",
        help=(
            "which cell list to expand: 'scenario' (the pruned design.cells in "
            "scenarios.json, default) or 'full' (every chain x resolution x "
            "model x window combination; 297 cells here, 891 trials at "
            "--repeats 3)"
        ),
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
    default_root = (REPO / "results/bench").resolve()
    if args.design == "full" and out_root == default_root and not args.dry_run:
        print(
            "warning: --design full is the whole matrix (~300 cells, days of board "
            "time) and is about to write into the shared default output. Give it "
            "its own root: --out results/bench-full",
            file=sys.stderr,
        )

    wanted = [name.strip() for name in (args.conditions or "").split(",") if name.strip()]
    conditions: list[tuple[str, dict[str, object]]] = []
    generated = matrix_conditions(scenario, design=args.design)
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
    if not args.dry_run:
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
    # Ceilings only: engine load, gaps and the fact that a trial stops as soon
    # as its window is full both push this number, in opposite directions.
    ceiling_hours = sum(ceilings) * repeats / 3600.0
    print(
        f"{len(conditions)} cell(s) x {repeats} pass(es); analysis window "
        f"{', '.join(str(n) for n in sorted(windows))} frames after "
        f"{', '.join(f'{s:g}' for s in sorted(settles))}s settle; "
        f"time ceiling {min(ceilings):.0f}-{max(ceilings):.0f}s "
        f"(<= {ceiling_hours:.1f} h total) -> {out_root}",
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
                "inflight": str(condition.get("inflight", 1)),
                "pre_ops": str(condition.get("pre_ops", "")),
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
            trial_dir.mkdir(parents=True, exist_ok=True)
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
    if not args.dry_run:
        # A dry run plans, it does not write: the artefact root may be the
        # shared `results/bench` and a plan there would look like a measurement.
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
