#!/usr/bin/env python3
"""Board + workload telemetry sampler.

Samples two independent sources into one CSV on a single monotonic clock, so it
can be joined with the workload's per-frame records (``t_mono``) afterwards:

* **board** -- ``tegrastats``: system RAM, per-core CPU load/freq, GPU duty
  cycle (``GR3D_FREQ``), power rails (``VDD_IN``, ``VDD_CPU_GPU_CV``,
  ``VDD_SOC``) and thermal zones. This is whole-board data: it includes the
  desktop session, which is wanted -- the noise floor is part of the
  measurement.
* **workload** -- ``/proc``: cumulative CPU milliseconds and PSS/RSS summed
  over the process group of the workload. Cumulative ticks are tracked as a
  per-PID high-water mark, so a workload that forks (``ros2 launch``) or exits
  mid-run does not lose accounted time.

No pinning, no priority changes, no GC meddling: the sampler only reads.

Usage::

    bench/sampler.py --out run/sampler.csv --pgid 4242 --interval-ms 100
    bench/sampler.py --out idle.csv --duration 30          # board only

Stops on SIGINT/SIGTERM, ``--duration``, or when ``--pgid`` disappears.

Caveats worth knowing when reading the CSV:

* ``tegrastats`` lines carry a 1-second-resolution wall-clock prefix, which is
  useless for alignment; the ``t_mono`` column is when *this* process read the
  line, so a board column can be up to one ``--interval-ms`` older than the
  timestamp next to it. At 100 ms that is well below anything in the analysis.
* ``GR3D_FREQ`` is a duty cycle, not occupancy. A latency-bound TensorRT chain
  legitimately shows a low number.
* Power rails are board-level, so energy can only be attributed per frame
  end-to-end, never per pipeline stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

FIELDS = (
    # --- board (tegrastats) ---
    "t_mono",
    "ram_used_mb",
    "ram_total_mb",
    "cpu_load_pct",
    "cpu_freq_mhz",
    "gpu_util_pct",
    "vdd_in_mw",
    "vdd_cpu_gpu_cv_mw",
    "vdd_soc_mw",
    "tj_c",
    "cpu_c",
    "gpu_c",
    "soc_c",
    # --- workload (/proc) ---
    "work_pids",
    "work_cpu_ms",
    "work_rss_mb",
    "work_pss_mb",
    "work_peak_rss_mb",
    "work_peak_pss_mb",
)

_CLK_TCK = os.sysconf("SC_CLK_TCK")

_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_CPU_LIST = re.compile(r"CPU \[([^\]]*)\]")
_CPU_CORE = re.compile(r"(\d+)%@(\d+)")
_GPU = re.compile(r"GR3D_FREQ (\d+)%")
_VDD_IN = re.compile(r"(?:VDD_IN|POM_5V_IN) (\d+)mW")
_VDD_CV = re.compile(r"VDD_CPU_GPU_CV (\d+)mW")
_VDD_SOC = re.compile(r"VDD_SOC (\d+)mW")
_TJ = re.compile(r"tj@([\d.]+)C")
_CPU_TEMP = re.compile(r"[cC][pP][uU]@([\d.]+)C")
_GPU_TEMP = re.compile(r"[gG][pP][uU]@([\d.]+)C")
_SOC_TEMP = re.compile(r"soc0@([\d.]+)C")


# ------------------------------------------------------------------ tegrastats
def parse_tegrastats(line: str) -> dict[str, float]:
    """Extract the fields we keep from one ``tegrastats`` line.

    Missing fields are simply absent -- ``tegrastats`` output differs between
    JetPack releases and board models, and a sampler that crashes because
    ``EMC_FREQ`` is gone is worse than one that reports fewer columns.
    """
    out: dict[str, float] = {}

    match = _RAM.search(line)
    if match:
        out["ram_used_mb"] = int(match.group(1))
        out["ram_total_mb"] = int(match.group(2))

    match = _CPU_LIST.search(line)
    if match:
        cores = _CPU_CORE.findall(match.group(1))
        if cores:
            out["cpu_load_pct"] = round(
                sum(int(load) for load, _ in cores) / len(cores), 2
            )
            out["cpu_freq_mhz"] = round(
                sum(int(freq) for _, freq in cores) / len(cores), 1
            )

    for key, pattern in (
        ("gpu_util_pct", _GPU),
        ("vdd_in_mw", _VDD_IN),
        ("vdd_cpu_gpu_cv_mw", _VDD_CV),
        ("vdd_soc_mw", _VDD_SOC),
        ("tj_c", _TJ),
        ("cpu_c", _CPU_TEMP),
        ("gpu_c", _GPU_TEMP),
        ("soc_c", _SOC_TEMP),
    ):
        match = pattern.search(line)
        if match:
            out[key] = float(match.group(1))
    return out


class TegrastatsReader(threading.Thread):
    """Drain ``tegrastats`` stdout into a queue without blocking shutdown."""

    def __init__(self, interval_ms: int) -> None:
        super().__init__(daemon=True)
        self.lines: "queue.Queue[str]" = queue.Queue()
        self._process: subprocess.Popen[str] | None = None
        self.available = False
        try:
            self._process = subprocess.Popen(
                ["tegrastats", "--interval", str(int(interval_ms))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self.available = True
        except OSError as exc:  # not a Jetson, or tegrastats not installed
            print(f"sampler: tegrastats unavailable ({exc}); board columns empty", file=sys.stderr)

    def run(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            self.lines.put(line)

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()


# ---------------------------------------------------------------------- /proc
def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _proc_stat(pid: int) -> tuple[int, int] | None:
    """``(pgrp, cpu_ticks)`` for ``pid``; ``comm`` may contain spaces/parens."""
    text = _read(f"/proc/{pid}/stat")
    if text is None:
        return None
    fields = text[text.rfind(")") + 2 :].split()
    if len(fields) < 13:
        return None
    try:
        return int(fields[2]), int(fields[11]) + int(fields[12])
    except (IndexError, ValueError):
        return None


def _proc_memory(pid: int) -> tuple[float, float]:
    """``(rss_mb, pss_mb)`` for ``pid``; PSS accounts for shared pages."""
    pss_kb = 0.0
    rollup = _read(f"/proc/{pid}/smaps_rollup")
    if rollup:
        for line in rollup.splitlines():
            if line.startswith("Pss:"):
                pss_kb = float(line.split()[1])
                break
    rss_kb = 0.0
    status = _read(f"/proc/{pid}/status")
    if status:
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = float(line.split()[1])
                break
    if rss_kb == 0.0:
        statm = _read(f"/proc/{pid}/statm")
        if statm:
            rss_kb = float(statm.split()[1]) * (os.sysconf("SC_PAGE_SIZE") / 1024.0)
    return rss_kb / 1024.0, pss_kb / 1024.0


class WorkloadSampler:
    """Cumulative CPU and memory of a process group, resilient to forks/exits."""

    def __init__(self, pgid: int | None = None, pids: tuple[int, ...] = ()) -> None:
        self.pgid = pgid
        self.explicit = set(pids)
        self._peak_ticks: dict[int, int] = {}
        self._peak_rss_mb = 0.0
        self._peak_pss_mb = 0.0
        self.pids: list[int] = []

    def members(self) -> list[int]:
        pids = set(self.explicit)
        if self.pgid is not None:
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                stat = _proc_stat(int(entry.name))
                if stat is not None and stat[0] == self.pgid:
                    pids.add(int(entry.name))
        return sorted(pids)

    def sample(self) -> dict[str, float]:
        self.pids = self.members()
        rss_mb = 0.0
        pss_mb = 0.0
        for pid in self.pids:
            stat = _proc_stat(pid)
            if stat is not None:
                self._peak_ticks[pid] = max(self._peak_ticks.get(pid, 0), stat[1])
            rss, pss = _proc_memory(pid)
            rss_mb += rss
            pss_mb += pss
        self._peak_rss_mb = max(self._peak_rss_mb, rss_mb)
        self._peak_pss_mb = max(self._peak_pss_mb, pss_mb)
        return {
            "work_pids": len(self.pids),
            "work_cpu_ms": round(sum(self._peak_ticks.values()) * 1000.0 / _CLK_TCK, 3),
            "work_rss_mb": round(rss_mb, 2),
            "work_pss_mb": round(pss_mb, 2),
            "work_peak_rss_mb": round(self._peak_rss_mb, 2),
            "work_peak_pss_mb": round(self._peak_pss_mb, 2),
        }


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="CSV to write")
    parser.add_argument(
        "--pgid", type=int, default=None, help="process group of the workload"
    )
    parser.add_argument(
        "--pid", type=int, action="append", default=[], help="extra PID (repeatable)"
    )
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument(
        "--duration", type=float, default=None, help="stop after this many seconds"
    )
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def _handle(signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    tegra = TegrastatsReader(args.interval_ms)
    tegra.start()
    workload = WorkloadSampler(args.pgid, tuple(args.pid))

    t_start = time.monotonic()
    samples = 0
    saw_board = False
    timeout = max(args.interval_ms / 1000.0 * 1.5, 0.05)

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        while not stop.is_set():
            if args.duration is not None and (time.monotonic() - t_start) >= args.duration:
                break
            try:
                line = tegra.lines.get(timeout=timeout)
            except queue.Empty:
                line = None
            if line is not None:
                saw_board = True
            t_mono = time.monotonic()
            row: dict[str, object] = {"t_mono": round(t_mono, 6)}
            if line is not None:
                row.update(parse_tegrastats(line))
            row.update(workload.sample())
            writer.writerow(row)
            handle.flush()
            samples += 1
            # Board-only mode: keep a steady cadence when tegrastats is silent.
            if line is None:
                time.sleep(min(args.interval_ms / 1000.0, 0.2))

    t_end = time.monotonic()
    tegra.stop()

    meta = {
        "t_start_mono": round(t_start, 6),
        "t_end_mono": round(t_end, 6),
        "interval_ms": args.interval_ms,
        "pgid": args.pgid,
        "pids_seen": workload.pids,
        "samples": samples,
        "board_source": "tegrastats" if saw_board else None,
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
    }
    Path(args.out).with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"sampler: {samples} samples -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
