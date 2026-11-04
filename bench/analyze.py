#!/usr/bin/env python3
"""Aggregate benchmark trials into one comparable report.

Reads the artefacts written by ``bench/run.py``::

    <root>/<condition>/r<N>/records.jsonl   per-frame stream from the workload
    <root>/<condition>/r<N>/sampler.csv     board + workload telemetry
    <root>/<condition>/r<N>/run.meta.json   command, exit code, machine state

and prints (and optionally writes) a markdown report.

Window selection happens *here*, not during the run: ``--settle`` seconds after
the first recorded frame are dropped (startup and the thermal ramp), then
``--frames`` frames are taken, so the same data can be re-examined with a
different window without repeating the measurement. Both default to the
scenario's ``trial`` block.

Reported per cell: latency percentiles, throughput, per-stage medians, CPU
milliseconds per frame (and how many cores that is), PSS, GPU duty cycle, board
power and joules per frame, peak temperature -- as mean ± sample std across
passes, with the per-pass values printed underneath.

Claims are not made from those means: every cell is also differenced against the
anchor *within each pass* and reported with a 95 % CI, because two means measured
hours apart share no machine state and the board drifts more than most effects.

Usage::

    bench/analyze.py                       # scenario window, deltas vs anchor
    bench/analyze.py --settle 20 --frames 1500
    bench/analyze.py --from 30 --to 240    # legacy: absolute seconds, no cap
    bench/analyze.py --md results/report.md --json results/report.json
    bench/analyze.py --timeline-out results/report/timeline
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- loading
def _percentiles(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        return {}
    p50, p95, p99 = np.percentile(data, [50, 95, 99])
    return {
        "n": len(data),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "mean": float(np.mean(data)),
        "max": float(np.max(data)),
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summary_provenance(trial_dir: Path) -> dict[str, Any]:
    """The workload's own summary, for what only it knows (the window it ran)."""
    path = trial_dir / "summary.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    return provenance if isinstance(provenance, dict) else {}


def _csv_rows(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    out: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    continue
            out.append(parsed)
    return out


def _energy_j(times: list[float], watts: list[float]) -> float:
    """Trapezoidal integral of the power rail over the window."""
    if len(times) < 2:
        return 0.0
    return float(
        sum(
            0.5 * (watts[i] + watts[i + 1]) * (times[i + 1] - times[i])
            for i in range(len(times) - 1)
        )
    )


def _median(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values if value is not None]
    return statistics.median(data) if data else None


def _rate(values: Iterable[float]) -> float | None:
    """Mean of a duty-cycle metric, where ``_median`` would collapse to zero.

    ``GR3D_FREQ`` and ``cpu_load_pct`` are rates: a 100 ms sample either caught
    GPU work or did not, so at low duty the samples are bimodal and the median
    lands on the zero mode. Measured on this board (op-free matrix, before the
    conditioning chain became a constant): the ROS chain reads 0 % by median
    while its non-zero samples are 22 % and its mean is 7 % -- and ``infer_ms``
    says the GPU is working 10 ms/frame. Rates are averaged; levels (PSS, power,
    temperature) stay medians, where an outlier should not move the number.
    """
    data = [float(value) for value in values if value is not None]
    return statistics.fmean(data) if data else None


def window_indices(
    records: list[dict[str, Any]],
    *,
    settle_s: float | None = None,
    frames: int | None = None,
    to_s: float | None = None,
) -> tuple[int, int]:
    """Half-open ``(start, stop)`` slice selecting the analysis window.

    Two knobs with two different jobs: ``settle_s`` drops the thermal/clock ramp
    (a property of elapsed time, so it is the same number for every cell), and
    ``frames`` then takes a fixed count (a property of the statistics, so every
    cell reports percentiles from the same n even when cells differ 4x in speed).
    ``to_s`` is the legacy absolute end of a time window and is kept for
    re-analysing old trials.
    """
    if not records:
        return 0, 0
    t_first = records[0]["t_mono"]
    start = t_first + (settle_s or 0.0)
    low = len(records)
    for index, record in enumerate(records):
        if record["t_mono"] >= start:
            low = index
            break
    high = len(records)
    if to_s is not None:
        end = t_first + to_s
        for index in range(low, len(records)):
            if records[index]["t_mono"] > end:
                high = index
                break
    if frames:
        high = min(high, low + int(frames))
    return low, high


def _stats(values: Iterable[float | None]) -> dict[str, Any] | None:
    """Mean and sample std across *passes* (never across frames).

    A per-frame std is milliseconds and meaningless (frames are autocorrelated
    and the distribution is skewed); the unit of replication is the trial.
    """
    data = [float(value) for value in values if value is not None]
    if not data:
        return None
    return {
        "n": len(data),
        "mean": statistics.fmean(data),
        "std": statistics.stdev(data) if len(data) > 1 else None,
        "min": min(data),
        "max": max(data),
    }


#: Two-sided 95 % Student t for df = n - 1; beyond the table the normal limit
#: (1.96) is close enough for a dispersion that is already only indicative.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
}


def _ci95(std: float | None, n: int) -> float | None:
    if std is None or n < 2:
        return None
    return _T95.get(n - 1, 1.96) * std / (n**0.5)


def _fmt_stats(stats: dict[str, Any] | None, digits: int = 2) -> str:
    if not stats:
        return "–"
    mean = f"{stats['mean']:.{digits}f}"
    if stats.get("std") is None:
        return mean
    return f"{mean} ± {stats['std']:.{digits}f}"


# ---------------------------------------------------------------------- trials
def summarise_trial(
    trial_dir: Path,
    *,
    settle_s: float | None,
    frames: int | None,
    to_s: float | None,
    records_name: str,
    sampler_name: str,
) -> dict[str, Any] | None:
    records_path = trial_dir / records_name
    if not records_path.is_file():
        return None
    records = _rows(records_path)
    if not records:
        return None

    meta_path = trial_dir / "run.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if meta.get("returncode") not in (None, 0):
        return {"condition": trial_dir.parent.name, "repeat": trial_dir.name, "failed": True}

    low, high = window_indices(records, settle_s=settle_s, frames=frames, to_s=to_s)
    window = records[low:high]
    if not window:
        span = records[-1]["t_mono"] - records[0]["t_mono"]
        return {
            "condition": trial_dir.parent.name,
            "repeat": trial_dir.name,
            "failed": True,
            "reason": (
                f"has no frames in the window (a {settle_s or 0:g}s settle over a "
                f"{span:.1f}s trial)"
            ),
        }

    count = len(window)
    latency = [record["latency_ms"] for record in window]
    span = window[-1]["t_mono"] - window[0]["t_mono"]
    throughput = count / span if span > 0 else 0.0

    result: dict[str, Any] = {
        "condition": trial_dir.parent.name,
        "repeat": trial_dir.name,
        "frames": count,
        "requested_frames": int(frames) if frames else None,
        "window_short": bool(frames) and count < int(frames),
        "window_s": round(span, 3),
        "replays": max(record["loop"] for record in window) - window[0]["loop"] + 1,
        "latency": _percentiles(latency),
        "fps": throughput,
        "detections_per_frame": sum(record["n_det"] for record in window) / count,
        "stage_ms": {
            key: _median(record[key] for record in window)
            for key in ("acq_ms", "pre_ms", "infer_ms", "post_ms")
        },
    }
    # ROS-only: serialisation + DDS + scheduling, absent from the monolith.
    if all("transport_ms" in record for record in window):
        result["stage_ms"]["transport_ms"] = _median(
            record["transport_ms"] for record in window
        )
    # ROS-only: wait in front of every busy node (delivery + queueing). At a
    # window of one this is transport; its growth with the window is the cost.
    if all("hop_ms" in record for record in window):
        result["stage_ms"]["hop_ms"] = _median(
            record["hop_ms"] for record in window
        )
    # Frames in flight: `inflight` is the realized concurrency at publish, so a
    # window that was never filled cannot be mistaken for one that filled and
    # bought nothing. `drops` counts holes in the seq set -- the failure mode a
    # window introduces when a topic's history is shallower than the window.
    seqs = [int(record["seq"]) for record in window]
    result["drops"] = sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1)
    if all("inflight" in record for record in window):
        realized = _median(record["inflight"] for record in window)
        result["inflight_real"] = realized
        requested = _summary_provenance(trial_dir).get("inflight")
        if requested:
            requested = int(requested)
            result["inflight_requested"] = requested
            result["fill"] = (realized / requested) if realized else None
            # Little's law on the closed loop: L * X / D ~ 1 when the window is
            # genuinely full and stable. Latency excludes decode (as everywhere
            # in this contract), so it reads slightly under 1.
            p50 = result["latency"].get("p50") or 0.0
            result["little"] = (p50 * throughput / 1000.0 / requested) if throughput else None
            result["window_not_full"] = bool(realized and realized < 0.9 * requested)

    # Stage-2 breakdown: one entry per conditioning op the workload ran. Absent
    # when it ran none, exactly like transport_ms, so an older records.jsonl
    # still analyses.
    for key, stage in (("pre_ops", "pre_ops"), ("post_ops", "post_ops")):
        if all(key in record for record in window):
            names = sorted({name for record in window for name in record[key]})
            result["stage_ms"][stage] = {
                name: _median(
                    record[key][name] for record in window if name in record[key]
                )
                for name in names
            }

    samples = [
        row
        for row in _csv_rows(trial_dir / sampler_name)
        if window[0]["t_mono"] - 0.5 <= row.get("t_mono", -1) <= window[-1]["t_mono"] + 0.5
    ]
    if samples:
        cpu = [row.get("work_cpu_ms") for row in samples if row.get("work_cpu_ms") is not None]
        if len(cpu) >= 2:
            cpu_used = max(cpu) - min(cpu)
            result["cpu_ms_per_frame"] = cpu_used / count
            result["cores_used"] = cpu_used / (span * 1000.0) if span > 0 else None
        result["pss_mb"] = _median(
            row.get("work_peak_pss_mb") or row.get("work_peak_rss_mb") for row in samples
        )
        result["gpu_util_pct"] = _rate(row.get("gpu_util_pct") for row in samples)
        result["cpu_load_pct"] = _rate(row.get("cpu_load_pct") for row in samples)
        result["ram_used_mb"] = _median(row.get("ram_used_mb") for row in samples)
        result["tj_c_max"] = max(
            (row["tj_c"] for row in samples if "tj_c" in row), default=None
        )

        powered = [
            (row["t_mono"], row["vdd_in_mw"] / 1000.0)
            for row in samples
            if "vdd_in_mw" in row and row["vdd_in_mw"] > 0
        ]
        if powered:
            times = [t for t, _ in powered]
            watts = [w for _, w in powered]
            result["power_w"] = _median(watts)
            energy = _energy_j(times, watts)
            result["joules"] = energy
            result["joules_per_frame"] = energy / count
    return result


def collect(
    root: Path,
    *,
    settle_s: float | None,
    frames: int | None,
    to_s: float | None,
    records_name: str,
    sampler_name: str,
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for trial_dir in sorted(root.glob("*/r*")):
        if not trial_dir.is_dir():
            continue
        trial = summarise_trial(
            trial_dir,
            settle_s=settle_s,
            frames=frames,
            to_s=to_s,
            records_name=records_name,
            sampler_name=sampler_name,
        )
        if trial is not None:
            trials.append(trial)
    return trials


# --------------------------------------------------------------------- report
SCALARS = (
    ("p50_ms", lambda t: t["latency"].get("p50")),
    ("p95_ms", lambda t: t["latency"].get("p95")),
    ("p99_ms", lambda t: t["latency"].get("p99")),
    ("fps", lambda t: t["fps"]),
    ("cpu_ms_f", lambda t: t.get("cpu_ms_per_frame")),
    ("load_pct", lambda t: t.get("cpu_load_pct")),
    ("cores", lambda t: t.get("cores_used")),
    ("pss_mb", lambda t: t.get("pss_mb")),
    ("gpu_pct", lambda t: t.get("gpu_util_pct")),
    ("power_w", lambda t: t.get("power_w")),
    ("j_f", lambda t: t.get("joules_per_frame")),
    ("tj_max", lambda t: t.get("tj_c_max")),
)


def aggregate(trials: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        if trial.get("failed"):
            continue
        by_condition.setdefault(trial["condition"], []).append(trial)

    report: dict[str, dict[str, Any]] = {}
    for condition, group in by_condition.items():
        entry: dict[str, Any] = {
            "trials": len(group),
            "frames": int(round(_median(t["frames"] for t in group) or 0)),
            "detections_per_frame": _median(t["detections_per_frame"] for t in group),
            "stats": {},
        }
        for name, getter in SCALARS:
            stats = _stats(getter(trial) for trial in group)
            entry["stats"][name] = stats
            # Flat mean, so a consumer does not have to dig; the dispersion lives
            # next to it in `stats` because it is the number a reader must see.
            entry[name] = stats["mean"] if stats else None
        entry["trials_detail"] = [
            {
                "repeat": trial["repeat"],
                "frames": trial["frames"],
                "p50_ms": trial["latency"].get("p50"),
                "p95_ms": trial["latency"].get("p95"),
                "fps": trial["fps"],
                "replays": trial.get("replays"),
            }
            for trial in group
        ]
        entry["stage_ms"] = {
            stage: _median(trial["stage_ms"].get(stage) for trial in group)
            for stage in (
                "acq_ms", "pre_ms", "infer_ms", "post_ms", "transport_ms", "hop_ms"
            )
        }
        # The window: requested (provenance) and realized (records), the fill
        # ratio and the closed-loop consistency check.
        entry["inflight"] = _median(
            trial.get("inflight_requested") for trial in group
        )
        entry["inflight_real"] = _median(
            trial.get("inflight_real") for trial in group
        )
        entry["fill"] = _median(trial.get("fill") for trial in group)
        entry["little"] = _median(trial.get("little") for trial in group)
        entry["drops"] = sum(int(trial.get("drops") or 0) for trial in group)
        # Per-op medians for the stages that can be split, aggregated the same
        # way as the stage buckets themselves.
        for stage in ("pre_ops", "post_ops"):
            op_trials = [trial for trial in group if trial["stage_ms"].get(stage)]
            if not op_trials:
                continue
            names = sorted(
                {name for trial in op_trials for name in trial["stage_ms"][stage]}
            )
            entry["stage_ms"][stage] = {
                name: _median(trial["stage_ms"][stage].get(name) for trial in op_trials)
                for name in names
            }
        report[condition] = entry
    return report


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    return f"{float(value):.{digits}f}"


# ------------------------------------------------------------- paired deltas
#: Metrics worth a per-pass difference against the anchor. Latency, throughput
#: and the two resource ratios are what a claim is actually made of.
DELTA_METRICS = (
    ("p50_ms", lambda t: t["latency"].get("p50")),
    ("p95_ms", lambda t: t["latency"].get("p95")),
    ("fps", lambda t: t.get("fps")),
    ("cpu_ms_f", lambda t: t.get("cpu_ms_per_frame")),
    ("j_f", lambda t: t.get("joules_per_frame")),
)


def paired_deltas(
    trials: list[dict[str, Any]], anchor: str | None
) -> dict[str, dict[str, Any]]:
    """Per-pass difference of every cell against the anchor cell.

    Two cells from different hours are not comparable: the board drifts ~8 % over
    a long session, which is larger than most effects and inflates the pooled
    std until nothing is significant. Differencing inside a pass cancels the
    machine state the two trials shared, so the spread that comes out is the one
    a claim has to clear. Passes are matched on the repeat index, which is a pass
    because every pass runs every cell once.
    """
    if not anchor:
        return {}
    usable = [trial for trial in trials if not trial.get("failed")]
    by_key = {(trial["condition"], str(trial["repeat"])): trial for trial in usable}
    anchor_trials = {
        str(trial["repeat"]): trial
        for trial in usable
        if trial["condition"] == anchor
    }
    if not anchor_trials:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for condition in dict.fromkeys(trial["condition"] for trial in usable):
        if condition == anchor:
            continue
        rows: list[dict[str, float]] = []
        for repeat, anchor_trial in sorted(anchor_trials.items()):
            cell_trial = by_key.get((condition, repeat))
            if cell_trial is None:
                continue
            row = {}
            for name, getter in DELTA_METRICS:
                cell_value, anchor_value = getter(cell_trial), getter(anchor_trial)
                if cell_value is None or anchor_value is None:
                    continue
                row[name] = float(cell_value) - float(anchor_value)
            if row:
                rows.append(row)
        if not rows:
            continue
        metrics: dict[str, dict[str, Any]] = {}
        for name, _getter in DELTA_METRICS:
            stats = _stats(row.get(name) for row in rows)
            if stats:
                metrics[name] = {**stats, "ci95": _ci95(stats["std"], stats["n"])}
        if metrics:
            out[condition] = metrics
    return out


def _fmt_delta(stats: dict[str, Any] | None, digits: int = 2) -> str:
    if not stats:
        return "–"
    mean = f"{stats['mean']:+.{digits}f}"
    ci = stats.get("ci95")
    if ci is None:
        return mean
    # A CI that excludes zero is the only kind of difference worth a sentence.
    star = "*" if abs(stats["mean"]) > ci else ""
    return f"{mean} ± {ci:.{digits}f}{star}"


def delta_markdown(
    deltas: dict[str, dict[str, Any]], anchor: str | None, n_passes: int
) -> list[str]:
    if not deltas or not anchor:
        return [
            "",
            "Deltas: none (set `anchor` in the scenario or pass `--anchor` to "
            "difference every cell against a reference measured in the same pass).",
        ]
    out = [
        "",
        f"Deltas vs `{anchor}`, within pass (mean ± 95 % CI over paired passes; "
        "`*` = CI excludes zero)",
        "",
        "| cell | Δp50 ms | Δp95 ms | Δfps | Δcpu ms/f | ΔJ/f | passes |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition, metrics in deltas.items():
        first = next(iter(metrics.values()))
        out.append(
            "| "
            + " | ".join(
                [
                    condition,
                    _fmt_delta(metrics.get("p50_ms")),
                    _fmt_delta(metrics.get("p95_ms")),
                    _fmt_delta(metrics.get("fps")),
                    _fmt_delta(metrics.get("cpu_ms_f"), 2),
                    _fmt_delta(metrics.get("j_f"), 3),
                    str(first.get("n", n_passes)),
                ]
            )
            + " |"
        )
    return out


#: `-i{D}` suffix: the cells of one chain/resolution/model at different windows.
WINDOW_SUFFIX = re.compile(r"-i(\d+)$")


def window_groups(report: dict[str, dict[str, Any]]) -> dict[str, dict[int, str]]:
    """Group cells that differ *only* in frames in flight."""
    groups: dict[str, dict[int, str]] = {}
    for condition, entry in report.items():
        match = WINDOW_SUFFIX.search(condition)
        base = condition[: match.start()] if match else condition
        declared = entry.get("inflight")
        inflight = int(declared) if declared else (int(match.group(1)) if match else 1)
        groups.setdefault(base, {})[inflight] = condition
    return {base: cells for base, cells in groups.items() if len(cells) > 1}


def operating_curve_markdown(
    report: dict[str, dict[str, Any]], trials: list[dict[str, Any]]
) -> list[str]:
    """One table per cell family: the (throughput, latency, joules) curve in D.

    This is the frames-in-flight result: latency is the *price* of a window and
    throughput/energy the dividend, so neither number means anything alone. The
    deltas are within-pass differences against the same cell at a window of one,
    which is the only reference measured on the same clock -- the global anchor
    stays the monolith.
    """
    groups = window_groups(report)
    if not groups:
        return []
    out = [
        "",
        "Frames in flight: the closed-loop operating curve",
        "",
        "Rows are one cell family at increasing windows; `i` is the requested "
        "window and `fill` the realized one (`inflight` at publish, median), so "
        "a window that never opened is visible as `fill` below 1. Deltas are "
        "within-pass differences against the same cell at `i=1` (mean ± 95 % CI; "
        "`*` = CI excludes zero). `L·X/i` is Little's law on the closed loop: "
        "≈1 when the window is genuinely full.",
        "",
    ]
    for base, cells in sorted(groups.items()):
        reference = cells.get(1)
        deltas = paired_deltas(trials, reference) if reference else {}
        out.append(
            f"| {base} | i | fill | fps | Δfps | p50 ms | Δp50 | p99 ms "
            "| hop ms | J/f | ΔJ/f | cores | L·X/i | drops |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for inflight in sorted(cells):
            condition = cells[inflight]
            entry = report[condition]
            stats = entry.get("stats") or {}
            delta = deltas.get(condition, {})
            out.append(
                "| "
                + " | ".join(
                    [
                        condition,
                        str(inflight),
                        _fmt(entry.get("fill"), 2),
                        _fmt_stats(stats.get("fps"), 2),
                        _fmt_delta(delta.get("fps")),
                        _fmt_stats(stats.get("p50_ms"), 2),
                        _fmt_delta(delta.get("p50_ms")),
                        _fmt(entry.get("p99_ms"), 2),
                        _fmt(entry["stage_ms"].get("hop_ms"), 2),
                        _fmt_stats(stats.get("j_f"), 3),
                        _fmt_delta(delta.get("j_f"), 3),
                        _fmt(entry.get("cores"), 2),
                        _fmt(entry.get("little"), 2),
                        str(entry.get("drops") or 0),
                    ]
                )
                + " |"
            )
        if reference is None:
            out.append(
                f"| _{base}_ | | | | | | | | | | | | | |"
            )
            out.append("")
            out.append(
                f"`{base}` has no `i=1` cell in this run: no local zero, so the "
                "deltas are missing (measure it, it is the demand-driven reference)."
            )
        out.append("")
    out.append(
        "Reading: throughput rises with `i` until the bottleneck node saturates "
        "(the knee is at `D* = Σ residency / max residency`, not at the process "
        "count), after which `p50` keeps growing by one bottleneck service time "
        "per extra frame while `fps` does not move. Energy per frame improves "
        "from 1 to the knee and is flat after it -- which is the real argument "
        "for a window, since latency only gets worse."
    )
    return out


def to_markdown(
    report: dict[str, dict[str, Any]],
    window: str,
    *,
    deltas: dict[str, dict[str, Any]] | None = None,
    anchor: str | None = None,
    trials: list[dict[str, Any]] | None = None,
) -> str:
    def stats_of(entry: dict[str, Any], metric: str) -> dict[str, Any] | None:
        return (entry.get("stats") or {}).get(metric)

    header = (
        "| condition | i | passes | frames | p50 ms | p95 ms | p99 ms | fps "
        "| cpu ms/f | cores | pss MB | gpu % | load % | W | J/f | tj °C |"
    )
    rule = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, rule]
    for condition, entry in report.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    _fmt(entry.get("inflight"), 0),
                    str(entry["trials"]),
                    str(entry["frames"]),
                    _fmt_stats(stats_of(entry, "p50_ms"), 2),
                    _fmt_stats(stats_of(entry, "p95_ms"), 2),
                    _fmt(entry.get("p99_ms"), 2),
                    _fmt_stats(stats_of(entry, "fps"), 2),
                    _fmt_stats(stats_of(entry, "cpu_ms_f"), 2),
                    _fmt(entry.get("cores"), 2),
                    _fmt(entry.get("pss_mb"), 0),
                    _fmt(entry.get("gpu_pct"), 0),
                    _fmt(entry.get("load_pct"), 0),
                    _fmt(entry.get("power_w"), 1),
                    _fmt_stats(stats_of(entry, "j_f"), 3),
                    _fmt(entry.get("tj_max"), 1),
                ]
            )
            + " |"
        )

    out = [
        f"Window: {window}",
        "",
        "Headline values are mean ± sample std across passes (the cell width is a "
        "dispersion, not a confidence interval: with n < 6 the std itself is only "
        "good to ~40 %, so read the per-trial table below for the raw values).",
        "",
        *lines,
        "",
    ]
    out.append(
        "Stages (median ms): pre / infer / post, acquisition separately. "
        "`xport` is the ROS-only serialisation + DDS + scheduling residual and "
        "`hop` the wait in front of each busy node: at a window of one they are "
        "the same number (all residual is delivery), beyond it `hop` grows by "
        "the queueing the window introduced."
    )
    out.append("")
    out.append(
        "| condition | i | acq | pre | infer | post | xport | hop | dets/frame |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for condition, entry in report.items():
        stages = entry["stage_ms"]
        out.append(
            "| "
            + " | ".join(
                [
                    condition,
                    _fmt(entry.get("inflight"), 0),
                    _fmt(stages.get("acq_ms"), 2),
                    _fmt(stages.get("pre_ms"), 2),
                    _fmt(stages.get("infer_ms"), 2),
                    _fmt(stages.get("post_ms"), 2),
                    _fmt(stages.get("transport_ms"), 2),
                    _fmt(stages.get("hop_ms"), 2),
                    _fmt(entry.get("detections_per_frame"), 2),
                ]
            )
            + " |"
        )

    # Only cells that ran a conditioning chain have a breakdown; print the
    # section only then (and note that the op medians need not sum to `pre`,
    # because the stage total is a sum of per-frame sums, not a sum of medians).
    for key, stage, label in (
        ("pre_ops", "pre", "Stage 2 (conditioning chain)"),
        ("post_ops", "post", "Stages 4-6 (NMS / filter / coordinate)"),
    ):
        op_rows = [
            (condition, name, value)
            for condition, entry in report.items()
            for name, value in (entry["stage_ms"].get(key) or {}).items()
        ]
        if not op_rows:
            continue
        out.append("")
        out.append(
            f"{label}, median ms per op, from the records' `{key}`. "
            f"`{stage}` above is the per-frame sum, so it exceeds the sum of "
            "these medians."
        )
        out.append("")
        out.append("| condition | op | ms |")
        out.append("|---|---|---|")
        for condition, name, value in op_rows:
            out.append(f"| {condition} | {name} | {_fmt(value, 2)} |")

    out.append("")
    out.append(
        "Per-trial detail (the raw values the mean ± std above are computed from; "
        "with n < 6 read these, not the summary)"
    )
    out.append("")
    out.append("| condition | repeat | frames | p50 | p95 | fps | replays |")
    out.append("|---|---|---|---|---|---|---|")
    for condition, entry in report.items():
        for trial in entry["trials_detail"]:
            out.append(
                "| "
                + " | ".join(
                    [
                        condition,
                        str(trial["repeat"]),
                        str(trial["frames"]),
                        _fmt(trial["p50_ms"], 2),
                        _fmt(trial["p95_ms"], 2),
                        _fmt(trial["fps"], 2),
                        str(trial.get("replays")),
                    ]
                )
                + " |"
            )
        stats = entry.get("stats") or {}
        out.append(
            f"| _{condition} mean ± std_ | | | "
            f"{_fmt_stats(stats.get('p50_ms'), 2)} | "
            f"{_fmt_stats(stats.get('p95_ms'), 2)} | "
            f"{_fmt_stats(stats.get('fps'), 2)} | |"
        )

    if deltas is not None:
        passes = max((entry.get("trials", 0) for entry in report.values()), default=0)
        out.extend(delta_markdown(deltas, anchor, passes))
    if trials is not None:
        out.extend(operating_curve_markdown(report, trials))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ timeline
#: Stage stack, bottom to top. Acquisition is deliberately separate: it is the
#: decoder's cost and is excluded from the latency headline.
_TIMELINE_STAGES = (
    ("pre_ms", "preprocess"),
    ("infer_ms", "inference"),
    ("post_ms", "postprocess"),
    ("transport_ms", "transport (ROS only)"),
)


def _stack(axes, seconds: list[float], records: list[dict[str, Any]], stages) -> float:
    """Stack ``stages`` as a step area; returns the top of the stack (ms)."""
    bottom = np.zeros(len(seconds))
    for key, label in stages:
        if not all(key in record for record in records):
            continue
        values = np.asarray([record[key] for record in records], dtype=float)
        axes.fill_between(
            seconds,
            bottom,
            bottom + values,
            step="post",
            label=label,
            linewidth=0.0,
        )
        bottom += values
    return float(bottom.max()) if bottom.size else 0.0


def plot_timelines(
    root: Path,
    *,
    settle_s: float | None,
    frames: int | None,
    to_s: float | None,
    out_dir: Path,
    records_name: str,
) -> list[Path]:
    """One two-panel latency figure per condition (first usable trial).

    This is the figure the numbers cannot replace. Top: the whole per-frame
    budget (for ROS the transport residual usually dominates, so the node-side
    work is invisible there). Bottom: the same frame without transport and
    without acquisition, which is where the stages can actually be compared.
    Replay boundaries are thin verticals, the analysis window is shaded.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is present on the board
        raise SystemExit("--timeline-out needs matplotlib") from None

    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for condition_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        trial = next(
            (
                candidate
                for candidate in sorted(condition_dir.glob("r*"))
                if (candidate / records_name).is_file()
            ),
            None,
        )
        if trial is None:
            continue
        records = _rows(trial / records_name)
        if not records:
            continue

        origin = records[0]["t_mono"]
        seconds = [record["t_mono"] - origin for record in records]
        figure, (top, detail) = plt.subplots(
            2,
            1,
            figsize=(11.0, 5.6),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        try:
            _stack(top, seconds, records, _TIMELINE_STAGES)
            _stack(detail, seconds, records, _TIMELINE_STAGES[:3])
            if all("acq_ms" in record for record in records):
                detail.plot(
                    seconds,
                    [record["acq_ms"] for record in records],
                    color="0.35",
                    linewidth=0.7,
                    linestyle="--",
                    label="acquisition (excluded from latency)",
                )

            low, high = window_indices(
                records, settle_s=settle_s, frames=frames, to_s=to_s
            )
            if low < high and (low > 0 or high < len(records)):
                start, window_end = seconds[low], seconds[high - 1]
                for axes in (top, detail):
                    axes.axvspan(start, window_end, color="0.9", zorder=0)
                top.plot([], [], color="0.9", linewidth=6, label="analysis window")

            for position in (
                seconds[index]
                for index in range(1, len(records))
                if records[index].get("loop") != records[index - 1].get("loop")
            ):
                for axes in (top, detail):
                    axes.axvline(position, color="0.2", linewidth=0.4, alpha=0.35)

            top.set_ylabel("ms (whole frame)")
            detail.set_ylabel("ms (stages only)")
            detail.set_xlabel("seconds after the first recorded frame")
            top.set_title(f"{condition_dir.name} ({trial.name}): per-frame stages")
            detail.set_xlim(0.0, max(seconds[-1], 0.001))
            top.legend(loc="upper left", fontsize=7, ncols=3, framealpha=0.85)
            detail.legend(loc="upper left", fontsize=7, ncols=2, framealpha=0.85)
            for axes in (top, detail):
                axes.grid(True, axis="y", linewidth=0.3, alpha=0.4)
            figure.tight_layout()
            target = out_dir / f"timeline-{condition_dir.name}.png"
            figure.savefig(target, dpi=130)
            written.append(target)
        finally:
            plt.close(figure)
    return written


def demo_section(demo_dir: Path) -> str | None:
    """List rendered demo material, when ``bench/run.py --demo`` produced any."""
    if not demo_dir.is_dir():
        return None
    videos = sorted(demo_dir.glob("*.mp4"))
    if not videos:
        return None
    lines = [
        "",
        "Demo material (rendered offline from detections.jsonl; not measured)",
        "",
    ]
    for video in videos:
        try:
            name = video.relative_to(REPO)
        except ValueError:  # pragma: no cover - outside the repo
            name = video
        lines.append(f"* `{name}` ({video.stat().st_size / 1e6:.0f} MB)")
    return "\n".join(lines)


# -------------------------------------------------------------------- glossary
#: One entry per column the report prints, grouped by the table it appears in.
#: Nothing here may restate a number: this is what an abbreviation means and how
#: the value was obtained, so a reader does not have to read `summarise_trial`.
_GLOSSARY: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Headline table",
        (
            (
                "condition",
                "cell name `<chain>-<resolution>-<model>[-i<D>]`: the command "
                "template from `bench/scenarios.json`, plus the frames-in-flight "
                "window when it is above 1. `–` anywhere in a table means the "
                "value was not available (not measured, or not emitted by that "
                "front-end).",
            ),
            (
                "i",
                "**frames in flight**: how many unacknowledged frames the "
                "acquisition node may have outstanding. `1` is demand-driven, one "
                "frame at a time; empty for the monolith, which has no window.",
            ),
            (
                "passes",
                "`n`: how many interleaved passes this cell was measured in, i.e. "
                "the sample size behind every mean ± std in its row.",
            ),
            (
                "frames",
                "median number of frames inside the analysis window (the settle "
                "seconds are dropped, then `analysis_frames` frames are taken). "
                "These are the frames the percentiles are computed over.",
            ),
            (
                "p50 / p95 / p99 ms",
                "percentiles of `latency_ms`, the **median**, 95th and 99th "
                "percentile frame latency: preprocessing + inference + "
                "postprocessing of one frame, acquisition (decode) excluded on "
                "purpose and presentation excluded because it is debug-only. In a "
                "ROS cell the transport residual is inside it. `p50` is the "
                "typical frame, `p99` the worst 1 %.",
            ),
            (
                "fps",
                "throughput: frames in the window divided by the window's span in "
                "seconds. The pipeline's rate, not the clip's, and not the rate of "
                "a time window: frames are the unit.",
            ),
            (
                "cpu ms/f",
                "workload CPU time per frame, from `/proc`: CPU milliseconds "
                "consumed by the process group over the window divided by frames "
                "delivered. It sums every thread, so it can far exceed the wall "
                "latency of a frame.",
            ),
            (
                "cores",
                "average parallelism in use over the window: CPU time over wall "
                "time (`cpu ms/f` x `fps` / 1000, which is what the sampler's "
                "`cpu_used / span` reduces to). `1.0` saturates exactly one core; "
                "`1.5` is one and a half cores busy.",
            ),
            (
                "pss MB",
                "peak **proportional set size** of the process group in MB over the "
                "window: resident memory with shared pages divided among the "
                "processes that map them, so a four-node chain does not pay for "
                "the same library four times. Peak RSS is used when PSS is "
                "unavailable.",
            ),
            (
                "gpu %",
                "GPU duty cycle from tegrastats `GR3D_FREQ`: the share of the "
                "window the GPU spent clocked on, not occupancy. Averaged (mean, "
                "not median) because at low duty the samples are bimodal -- GPU "
                "work or no GPU work -- and a median would land on the zero mode. "
                "A latency-bound chain reads low here even while TensorRT is the "
                "bottleneck.",
            ),
            (
                "load %",
                "whole-board CPU load averaged across all cores over the window -- "
                "every process, not just the pipeline. This is the comparability "
                "column: identical code measured 27 % on a quiet desk and 95 % "
                "with a browser open, 1.4x slower for it, so cells whose `load %` "
                "differ are not comparable by absolute value (their within-pass "
                "deltas still are).",
            ),
            (
                "W",
                "board power in watts: median of the tegrastats `VDD_IN` samples "
                "over the window. One rail for the whole Jetson, so it cannot be "
                "attributed to a pipeline stage.",
            ),
            (
                "J/f",
                "joules per frame: the trapezoidal integral of board power over "
                "the window divided by the frames delivered, i.e. the energy of "
                "one frame end to end.",
            ),
            (
                "tj °C",
                "junction temperature in °C, maximum over the window: how hot the "
                "SoC got while the cell ran.",
            ),
            (
                "mean ± std",
                "mean across passes ± sample standard deviation (n − 1). A "
                "dispersion of the passes, not a confidence interval: with n < 6 "
                "the std itself is only good to ~40 %, which is why the per-trial "
                "table prints the raw values.",
            ),
        ),
    ),
    (
        "Stage table",
        (
            (
                "acq",
                "acquisition, median ms: read, decode and rotate one frame from the "
                "source. It is the decoder's cost and is excluded from `latency_ms`.",
            ),
            (
                "pre",
                "preprocessing, median ms: the conditioning chain (CLAHE, "
                "bilateral, ...) at source resolution. It means conditioning only "
                "in both front-ends -- letterbox and tensor count inside `infer` "
                "-- and the chain is the same constant in every cell "
                "(`conditioning.ops`), so the column compares like with like. An "
                "empty chain is identity and reads ~0, which is why the harness "
                "rejects a cell that skips it.",
            ),
            (
                "infer",
                "inference, median ms: the model input contract (letterbox, BGR "
                "to RGB, HWC to NCHW, dtype/scale), the engine call and "
                "`decode_predictions` with thresholding. In the ROS chain the "
                "inference node publishes decoded candidates, so compare `infer` "
                "and `post` as one block against the monolith.",
            ),
            (
                "post",
                "postprocessing, median ms: NMS over all classes, then the domain "
                "filter, then coordinate adaptation back to the input frame. The "
                "order is canonical, so this value is not reorderable.",
            ),
            (
                "xport",
                "ROS only: `transport_ms`, the residual of serialisation + DDS + "
                "scheduling, computed by the terminal node as `latency_ms - "
                "(pre_ms + infer_ms + post_ms)`. Measured at ~7 ms fixed plus ~2.2 "
                "ms/MB on this board. The monolith has no such stage.",
            ),
            (
                "hop",
                "ROS only: the measured wait in front of each busy node. It equals "
                "`xport` at a window of one (all residual is delivery); its growth "
                "beyond that is the queueing the window introduced.",
            ),
            (
                "dets/frame",
                "mean number of detections surviving the domain filter per frame. "
                "Not a performance metric: it is the check that two front-ends "
                "looked at the same picture (a wrong `rotate` shows up here).",
            ),
        ),
    ),
    (
        "Per-op tables",
        (
            (
                "op",
                "one conditioning operator of the chain (`clahe`, `bilateral`, "
                "`gaussian`, `gamma`, `hist_equalization`, `median`, `unsharp`), or "
                "one of `nms` / `domain_filter` / `coordinate_adaptation`, median "
                "ms over the frames that ran it.",
            ),
            (
                "ms",
                "median of that op's own timer. The `pre`/`post` value above is the "
                "per-frame *sum* of the ops, so it exceeds the sum of these "
                "medians.",
            ),
        ),
    ),
    (
        "Per-trial detail",
        (
            (
                "repeat",
                "pass index (`r1`, `r2`, ...): the trial directory this row comes "
                "from. Passes are interleaved, so each one carries the machine "
                "state the other cells in the same pass also saw.",
            ),
            (
                "replays",
                "how many times the source clip was played end to end inside the "
                "window (`loop` range). A fast cell replays it more often, which is "
                "why the window is counted in frames: it keeps the content mix "
                "comparable.",
            ),
            (
                "_condition_ mean ± std",
                "the same pass aggregate as the headline table, repeated under the "
                "raw rows it was computed from.",
            ),
        ),
    ),
    (
        "Deltas vs the anchor",
        (
            (
                "Δp50 / Δp95 / Δfps / Δcpu ms/f / ΔJ/f",
                "within-pass difference: this cell minus the anchor cell *in the "
                "same pass*, reported as mean ± 95 % CI over the paired passes. "
                "Differencing inside a pass cancels the machine state the two "
                "trials shared (~8 % drift over a session), which is why a claim "
                "is made from this column and not from two headline means.",
            ),
            (
                "*",
                "the 95 % CI excludes zero -- the only kind of difference worth a "
                "sentence. No star means the effect is inside the noise floor.",
            ),
            (
                "passes",
                "how many paired passes the delta rests on: only passes in which "
                "both this cell and the anchor produced a usable trial.",
            ),
        ),
    ),
    (
        "Frames-in-flight curve",
        (
            (
                "fill",
                "realized window over requested: median `inflight` at publish "
                "divided by `i`. Below 0.9 is a warning -- the acquisition node "
                "never filled the window, so the cell did not measure what its "
                "name says.",
            ),
            (
                "L·X/i",
                "Little's law on the closed loop, `p50 x fps / 1000 / i`: ~1 when "
                "the window is genuinely full, reading slightly under 1 because "
                "`p50` excludes acquisition.",
            ),
            (
                "drops",
                "gaps in the `seq` sequence inside the window. Must be 0: a "
                "non-zero value means a topic history was shallower than the "
                "window and silently evicted an unacknowledged sample.",
            ),
            (
                "Δfps / Δp50 (curve)",
                "within-pass difference against the *same cell* at `i=1` -- the "
                "local zero of the curve, not the global anchor. `p50` is the "
                "price of the window, `fps` and `J/f` the dividend.",
            ),
        ),
    ),
)


def glossary_markdown() -> str:
    """One section explaining every column the report prints.

    Appended unconditionally: the report is a document that gets read months
    later and out of context, so the units and the estimators have to travel with
    the numbers they describe.
    """
    lines = [
        "",
        "## Glossary",
        "",
        "Every column the tables above print, in the order it appears. "
        "The metric dictionary with the measurement policy behind it is "
        "`bench/README.md` and `bench/METHOD.md`; this is the short version, "
        "next to the numbers it explains.",
    ]
    for title, entries in _GLOSSARY:
        lines.append("")
        lines.append(f"**{title}**")
        lines.append("")
        lines.append("| column | meaning |")
        lines.append("|---|---|")
        for column, meaning in entries:
            lines.append(f"| `{column}` | {meaning} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=REPO / "results/bench")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path(__file__).resolve().with_name("scenarios.json"),
        help="read the default settle/frames window from this scenario's `trial` block",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=None,
        help="seconds after the first frame treated as settling (default: trial.settle_s)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="frames of analysis window after the settling (default: trial.analysis_frames)",
    )
    parser.add_argument(
        "--from",
        dest="from_s",
        type=float,
        default=None,
        help="legacy: start the window this many seconds after the first frame",
    )
    parser.add_argument(
        "--to",
        dest="to_s",
        type=float,
        default=None,
        help="end the window this many seconds after the first frame",
    )
    parser.add_argument(
        "--anchor",
        default=None,
        help="cell every other cell is differenced against within each pass "
        "(default: the scenario's `anchor`)",
    )
    parser.add_argument("--records", default="records.jsonl")
    parser.add_argument("--sampler", default="sampler.csv")
    parser.add_argument("--md", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--timeline-out",
        type=Path,
        default=None,
        help="write one stage-stacked latency figure per condition into this directory",
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=REPO / "results/demo",
        help="list rendered demo videos from this directory in the report (if any)",
    )
    args = parser.parse_args(argv)

    scenario: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    try:
        scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
        defaults = dict(scenario.get("trial") or {})
    except (OSError, ValueError):
        defaults = {}
    settle_s = args.from_s if args.from_s is not None else args.settle
    if settle_s is None:
        settle_s = defaults.get("settle_s")
    window_frames = args.frames if args.frames is not None else defaults.get("analysis_frames")

    trials = collect(
        args.dir,
        settle_s=settle_s,
        frames=window_frames,
        to_s=args.to_s,
        records_name=args.records,
        sampler_name=args.sampler,
    )
    if not trials:
        raise SystemExit(
            f"no trials with {args.records} found under {args.dir}"
            " (if the window warnings above are empty, the directory has no trial output)"
        )
    failed = [trial for trial in trials if trial.get("failed")]
    for trial in failed:
        print(
            f"warning: {trial['condition']}/{trial['repeat']} "
            f"{trial.get('reason') or 'exited non-zero'}; excluded",
            file=sys.stderr,
        )

    report = aggregate(trials)
    if not report:
        raise SystemExit("no usable trials: every trial was excluded (see warnings above)")
    spans = [trial["window_s"] for trial in trials if trial.get("window_s")]
    span_note = f" (median span {statistics.median(spans):.1f}s)" if spans else ""
    if window_frames:
        window = (
            f"first {int(window_frames)} frames after {settle_s or 0:g}s of settling"
            + span_note
        )
    else:
        window = (
            f"{settle_s or 0:g}s → "
            + ("end" if args.to_s is None else f"{args.to_s:g}s")
            + " after the first frame"
        )
    for trial in trials:
        if trial.get("window_short"):
            print(
                f"warning: {trial['condition']}/{trial['repeat']} yielded "
                f"{trial['frames']} of {trial['requested_frames']} window frames "
                "(trial stopped too early: lower the fps hint or raise the ceiling)",
                file=sys.stderr,
            )
    for trial in trials:
        if trial.get("drops"):
            print(
                f"warning: {trial['condition']}/{trial['repeat']} has "
                f"{trial['drops']} gap(s) in the seq set: a frame was lost in the "
                "graph (a topic history shallower than the window evicts samples)",
                file=sys.stderr,
            )
        if trial.get("window_not_full"):
            print(
                f"warning: {trial['condition']}/{trial['repeat']} only realized "
                f"a window of {trial.get('inflight_real')} of "
                f"{trial.get('inflight_requested')} frames (fill "
                f"{(trial.get('fill') or 0):.2f}): the acquisition node never "
                "filled it, so this cell did not measure what its name says",
                file=sys.stderr,
            )
    anchor = args.anchor or scenario.get("anchor")
    deltas = paired_deltas(trials, anchor)
    if args.anchor and anchor not in report:
        print(
            f"warning: anchor {anchor!r} has no usable pass; no deltas reported",
            file=sys.stderr,
        )
    markdown = to_markdown(report, window, deltas=deltas, anchor=anchor, trials=trials)
    extra = demo_section(args.demo_dir)
    if extra:
        markdown += extra + "\n"
    markdown += glossary_markdown()
    print(markdown)

    if args.timeline_out is not None:
        written = plot_timelines(
            args.dir,
            settle_s=settle_s,
            frames=window_frames,
            to_s=args.to_s,
            out_dir=args.timeline_out,
            records_name=args.records,
        )
        for path in written:
            print(f"timeline -> {path}", file=sys.stderr)
        if not written:
            print("no trials to plot", file=sys.stderr)

    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown, encoding="utf-8")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"anchor": anchor, "deltas": deltas, "conditions": report}
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
