#!/usr/bin/env python3
"""Aggregate benchmark trials into one comparable report.

Reads the artefacts written by ``bench/run.py``::

    <root>/<condition>/r<N>/records.jsonl   per-frame stream from the workload
    <root>/<condition>/r<N>/sampler.csv     board + workload telemetry
    <root>/<condition>/r<N>/run.meta.json   command, exit code, machine state

and prints (and optionally writes) a markdown report.

Window selection happens *here*, not during the run: ``--from``/``--to`` are
seconds relative to the first recorded frame, so the same data can be
re-examined with a different steady-state window without repeating the
measurement. The default is the whole trial.

Reported per condition: latency percentiles, throughput, per-stage medians,
CPU milliseconds per frame (and how many cores that is), PSS, GPU duty cycle,
board power and joules per frame, peak temperature -- plus the spread across
repeats, which is the noise floor any difference must beat.

Usage::

    bench/analyze.py                       # whole trials
    bench/analyze.py --from 30 --to 240    # steady-state window
    bench/analyze.py --md results/report.md --json results/report.json
    bench/analyze.py --timeline-out results/report/timeline
"""

from __future__ import annotations

import argparse
import csv
import json
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
        result["gpu_util_pct"] = _median(row.get("gpu_util_pct") for row in samples)
        result["cpu_load_pct"] = _median(row.get("cpu_load_pct") for row in samples)
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
            for stage in ("acq_ms", "pre_ms", "infer_ms", "post_ms", "transport_ms")
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


def to_markdown(
    report: dict[str, dict[str, Any]],
    window: str,
    *,
    deltas: dict[str, dict[str, Any]] | None = None,
    anchor: str | None = None,
) -> str:
    def stats_of(entry: dict[str, Any], metric: str) -> dict[str, Any] | None:
        return (entry.get("stats") or {}).get(metric)

    header = (
        "| condition | passes | frames | p50 ms | p95 ms | p99 ms | fps "
        "| cpu ms/f | cores | pss MB | gpu % | W | J/f | tj °C |"
    )
    rule = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, rule]
    for condition, entry in report.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
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
        "`xport` is the ROS-only serialisation + DDS + scheduling residual."
    )
    out.append("")
    out.append("| condition | acq | pre | infer | post | xport | dets/frame |")
    out.append("|---|---|---|---|---|---|---|")
    for condition, entry in report.items():
        stages = entry["stage_ms"]
        out.append(
            "| "
            + " | ".join(
                [
                    condition,
                    _fmt(stages.get("acq_ms"), 2),
                    _fmt(stages.get("pre_ms"), 2),
                    _fmt(stages.get("infer_ms"), 2),
                    _fmt(stages.get("post_ms"), 2),
                    _fmt(stages.get("transport_ms"), 2),
                    _fmt(entry.get("detections_per_frame"), 2),
                ]
            )
            + " |"
        )

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
    anchor = args.anchor or scenario.get("anchor")
    deltas = paired_deltas(trials, anchor)
    if args.anchor and anchor not in report:
        print(
            f"warning: anchor {anchor!r} has no usable pass; no deltas reported",
            file=sys.stderr,
        )
    markdown = to_markdown(report, window, deltas=deltas, anchor=anchor)
    extra = demo_section(args.demo_dir)
    if extra:
        markdown += extra + "\n"
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
