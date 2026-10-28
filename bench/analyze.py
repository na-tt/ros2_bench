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


# ---------------------------------------------------------------------- trials
def summarise_trial(
    trial_dir: Path,
    *,
    from_s: float | None,
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

    t_first = records[0]["t_mono"]
    start = t_first + (from_s or 0.0)
    end = t_first + to_s if to_s is not None else float("inf")
    window = [record for record in records if start <= record["t_mono"] <= end]
    if not window:
        return None

    latency = [record["latency_ms"] for record in window]
    frames = len(window)
    span = window[-1]["t_mono"] - window[0]["t_mono"]
    throughput = frames / span if span > 0 else 0.0

    result: dict[str, Any] = {
        "condition": trial_dir.parent.name,
        "repeat": trial_dir.name,
        "frames": frames,
        "window_s": round(span, 3),
        "replays": max(record["loop"] for record in window) - window[0]["loop"] + 1,
        "latency": _percentiles(latency),
        "fps": throughput,
        "detections_per_frame": sum(record["n_det"] for record in window) / frames,
        "stage_ms": {
            key: _median(record[key] for record in window)
            for key in ("acq_ms", "pre_ms", "infer_ms", "post_ms")
        },
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
            result["cpu_ms_per_frame"] = cpu_used / frames
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
            result["joules_per_frame"] = energy / frames
    return result


def collect(
    root: Path,
    *,
    from_s: float | None,
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
            from_s=from_s,
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
            "spread": {},
        }
        for name, getter in SCALARS:
            values = [getter(trial) for trial in group]
            values = [value for value in values if value is not None]
            entry[name] = _median(values)
            if len(values) > 1:
                low, high = min(values), max(values)
                entry["spread"][name] = (
                    (high - low) / statistics.median(values) * 100.0
                    if statistics.median(values)
                    else None
                )
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
            for stage in ("acq_ms", "pre_ms", "infer_ms", "post_ms")
        }
        report[condition] = entry
    return report


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    return f"{float(value):.{digits}f}"


def to_markdown(report: dict[str, dict[str, Any]], window: str) -> str:
    header = (
        "| condition | trials | frames | p50 | p95 | p99 | fps | cpu ms/f | cores "
        "| pss MB | gpu % | W | J/f | tj °C |"
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
                    _fmt(entry.get("p50_ms"), 2),
                    _fmt(entry.get("p95_ms"), 2),
                    _fmt(entry.get("p99_ms"), 2),
                    _fmt(entry.get("fps"), 2),
                    _fmt(entry.get("cpu_ms_f"), 2),
                    _fmt(entry.get("cores"), 2),
                    _fmt(entry.get("pss_mb"), 0),
                    _fmt(entry.get("gpu_pct"), 0),
                    _fmt(entry.get("power_w"), 1),
                    _fmt(entry.get("j_f"), 2),
                    _fmt(entry.get("tj_max"), 1),
                ]
            )
            + " |"
        )

    out = [f"Window: {window}", "", *lines, ""]
    out.append("Stages (median ms): pre / infer / post, acquisition separately")
    out.append("")
    out.append("| condition | acq | pre | infer | post | dets/frame |")
    out.append("|---|---|---|---|---|---|")
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
                    _fmt(entry.get("detections_per_frame"), 2),
                ]
            )
            + " |"
        )

    out.append("")
    out.append("Per-trial detail (spread across repeats is the noise floor)")
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
        spread = entry["spread"]
        if spread:
            out.append(
                f"| _{condition} spread_ | | | "
                f"{_fmt(spread.get('p50_ms'), 1)}% | {_fmt(spread.get('p95_ms'), 1)}% | "
                f"{_fmt(spread.get('fps'), 1)}% | |"
            )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=REPO / "results/bench")
    parser.add_argument(
        "--from", dest="from_s", type=float, default=None,
        help="start of the analysis window, seconds after the first frame",
    )
    parser.add_argument(
        "--to", dest="to_s", type=float, default=None,
        help="end of the analysis window, seconds after the first frame",
    )
    parser.add_argument("--records", default="records.jsonl")
    parser.add_argument("--sampler", default="sampler.csv")
    parser.add_argument("--md", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    trials = collect(
        args.dir,
        from_s=args.from_s,
        to_s=args.to_s,
        records_name=args.records,
        sampler_name=args.sampler,
    )
    if not trials:
        raise SystemExit(f"no trials with {args.records} found under {args.dir}")
    failed = [trial for trial in trials if trial.get("failed")]
    for trial in failed:
        print(
            f"warning: {trial['condition']}/{trial['repeat']} exited non-zero; excluded",
            file=sys.stderr,
        )

    report = aggregate(trials)
    window = (
        f"{args.from_s or 0:g}s → "
        + ("end" if args.to_s is None else f"{args.to_s:g}s")
        + " after the first frame"
    )
    markdown = to_markdown(report, window)
    print(markdown)

    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown, encoding="utf-8")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
