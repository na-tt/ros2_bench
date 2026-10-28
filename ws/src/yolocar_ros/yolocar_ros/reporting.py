"""
Per-frame records and the run summary for the ROS 2 front-end.

This mirrors ``yolocar.main`` (the monolith's benchmark loop) so both
front-ends emit the same file contract: ``records.jsonl`` with one JSON object
per frame and ``summary.json`` with the same keys. The ROS front-end adds two
fields the monolith cannot have -- ``transport_ms`` per record and a
``transport`` bucket in ``stage_ms`` -- because only a distributed chain spends
time between stages.

It is deliberately free of ROS imports: the postprocessing node feeds it plain
Python values, which keeps it unit-testable without a middleware context.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
from typing import Iterable, Mapping

import cv2
import numpy as np

from yolocar.config import YoloCarConfig


# ------------------------------------------------------------------- records
class RecordWriter:
    """
    One JSON object per frame (JSONL), a no-op when no path was given.

    Each line is flushed immediately: the postprocessing node acknowledges a
    frame only after its record is on disk, so a killed run still leaves a
    complete prefix of the stream.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._file = None
        self.open_file()

    def open_file(self) -> 'RecordWriter':
        if self.path is not None and self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open('w', encoding='utf-8')
        return self

    def write(self, record: Mapping[str, object]) -> None:
        if self._file is not None:
            self._file.write(json.dumps(dict(record)) + '\n')
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> 'RecordWriter':
        return self.open_file()

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# --------------------------------------------------------------------- stats
def latency_stats(values_ms: Iterable[float]) -> dict[str, float]:
    """Mean/std/median/p95/min/max, or an empty dict when nothing was recorded."""
    values = np.asarray(list(values_ms), dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)),
        'min': float(values.min()),
        'max': float(values.max()),
    }


def detection_summary(
    frames: int, per_frame: list[int], by_class: dict[str, int]
) -> dict[str, object]:
    total = int(sum(per_frame))
    return {
        'total': total,
        'per_frame_mean': (total / frames) if frames else 0.0,
        'per_frame_max': int(max(per_frame)) if per_frame else 0,
        'by_class': dict(sorted(by_class.items(), key=lambda item: -item[1])),
    }


# ---------------------------------------------------------------- provenance
def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding='utf-8', errors='replace').strip()
    except OSError:
        return None


def _git_commit() -> str | None:
    """Short HEAD sha of the checkout this module was imported from, if any."""
    for directory in (Path.cwd(), Path(__file__).resolve().parents[4]):
        try:
            result = subprocess.run(
                ['git', '-C', str(directory), 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def _file_identity(path: Path) -> dict[str, object] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return {'path': str(path), 'bytes': stat.st_size, 'mtime': stat.st_mtime}


def provenance(config: YoloCarConfig, source: str) -> dict[str, object]:
    """Enough context to tell whether two runs are comparable at all."""
    return {
        'frontend': 'ros2',
        'python': platform.python_version(),
        'opencv': cv2.__version__,
        'numpy': np.__version__,
        'git_commit': _git_commit(),
        'board': _read_text('/proc/device-tree/model'),
        'cpu_governor': _read_text(
            '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
        ),
        'engine': _file_identity(Path(config.model_path)),
        'source': source,
    }


# ------------------------------------------------------------------- summary
def build_summary(
    *,
    config: YoloCarConfig,
    source: str,
    frames: int,
    replays: int,
    events: Mapping[str, float],
    interrupted: bool,
    wall_time_s: float,
    latency_ms: list[float],
    acquisition_ms: list[float],
    preprocess_ms: list[float],
    inference_ms: list[float],
    postprocess_ms: list[float],
    transport_ms: list[float],
    detections_per_frame: list[int],
    detections_by_class: dict[str, int],
    records_path: Path | None,
) -> dict[str, object]:
    """Assemble the ``summary.json`` payload (same shape as the monolith)."""
    fps = (frames / wall_time_s) if frames and wall_time_s > 0 else 0.0
    summary: dict[str, object] = {
        'pipeline': 'yolocar',
        'frontend': 'ros2',
        'model': str(config.model_path),
        'source': source,
        'frames': frames,
        'loops': config.loops,
        'replays': replays,
        'duration_s': config.duration_s,
        'interrupted': interrupted,
        'wall_time_s': wall_time_s,
        #: No presentation layer in the node graph, so the raw rate *is* the
        #: comparable rate; kept for contract parity with the monolith.
        'benchmark_time_s': wall_time_s,
        'fps': fps,
        'fps_wall': fps,
        'latency_ms': latency_stats(latency_ms),
        'stage_ms': {
            'acquisition': latency_stats(acquisition_ms),
            'preprocess': latency_stats(preprocess_ms),
            'inference': latency_stats(inference_ms),
            'postprocess': latency_stats(postprocess_ms),
            'presentation': {},
            #: ROS-only residual: capture -> detections minus the three stage
            #: timings, i.e. serialisation, transport and scheduling.
            'transport': latency_stats(transport_ms),
        },
        'events': {name: round(value, 6) for name, value in events.items()},
        'detections': detection_summary(
            frames, detections_per_frame, detections_by_class
        ),
        'provenance': provenance(config, source),
        'records': None if records_path is None else str(records_path),
        'config': {
            'conf_threshold': config.conf_threshold,
            'iou_threshold': config.iou_threshold,
            'image_size': config.image_size,
            'target_class_ids': list(config.target_class_ids),
            'agnostic_nms': config.agnostic_nms,
            'device': config.device,
            'rotate': config.rotate,
            'max_frames': config.max_frames,
        },
    }
    return summary


def write_summary(path: Path | None, summary: Mapping[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(summary), indent=2) + '\n', encoding='utf-8')
