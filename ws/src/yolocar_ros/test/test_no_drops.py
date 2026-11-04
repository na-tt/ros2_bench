"""
Hardware test: a windowed chain loses no frame and changes no detection.

This starts the real graph (``ros2 launch``, four processes, real engine) because
the two things the window can break are only visible end to end:

* **Contiguity.** With ``inflight`` frames resident, every data topic's history
  depth has to be at least the window. A shorter one evicts an unacked sample
  even under ``RELIABLE``; the credit is released by a later ack anyway, so the
  graph keeps running and the loss survives only as a hole in the sequence.
  Nothing at runtime complains -- so the records are checked for exactly
  ``{0 .. published-1}``, and that the tail is not short either (the acquisition
  node drains the window before the terminal node writes the summary).
* **Identity.** Running two frames concurrently must not change what the
  pipeline finds. Same clip, same frames, different window: the per-frame
  detection counts must be identical.

Skipped when the engine, the clip, ``ros2`` or TensorRT are unavailable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytest.importorskip('tensorrt')

REPO = Path(__file__).resolve().parents[4]
CLIP = REPO / 'assets/video/0000f77c-6257be58_180p.mov'
ENGINE = REPO / 'assets/models/yolov8n_fp16.engine'
YOLOCAR_SRC = REPO / 'monolith/yolocar/src'

#: Small enough to be a fast test, large enough that a window of 3 is genuinely
#: filled and the drain at the end has frames to wait for.
FRAMES = 24


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(YOLOCAR_SRC)
    if env.get('PYTHONPATH'):
        pythonpath = pythonpath + os.pathsep + env['PYTHONPATH']
    env['PYTHONPATH'] = pythonpath
    return env


def _run(out: Path, inflight: int) -> tuple[list[dict], dict]:
    """Launch the four-node chain for ``FRAMES`` frames at the given window."""
    if shutil.which('ros2') is None:
        pytest.skip('ros2 is not on PATH')
    if not (CLIP.is_file() and ENGINE.is_file()):
        pytest.skip('benchmark assets are not present')
    out.mkdir(parents=True, exist_ok=True)
    records = out / 'records.jsonl'
    summary = out / 'summary.json'
    command = [
        'ros2',
        'launch',
        'yolocar_ros',
        'bench.launch.py',
        f'model:={ENGINE}',
        f'source:={CLIP}',
        'rotate:=0',
        'loops:=1',
        'duration:=-1',
        f'max_frames:={FRAMES}',
        f'inflight:={inflight}',
        f'records:={records}',
        f'summary:={summary}',
        'log_level:=WARNING',
    ]
    result = subprocess.run(
        command,
        cwd=REPO,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert records.is_file(), (
        'no records.jsonl\n'
        f'--- launch stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
    rows = [
        json.loads(line)
        for line in records.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    return rows, json.loads(summary.read_text(encoding='utf-8'))


def test_a_windowed_chain_loses_no_frame(tmp_path: Path) -> None:
    rows, summary = _run(tmp_path / 'i3', inflight=3)

    # Nothing was evicted by a too-shallow topic history, and nothing was left
    # in the graph when the summary was written.
    assert [row['seq'] for row in rows] == list(range(len(rows)))
    assert len(rows) == FRAMES

    # The window was actually opened, not flattened back to one frame.
    outstanding = [row['inflight'] for row in rows]
    assert max(outstanding) > 1
    assert all(1 <= value <= 3 for value in outstanding)

    assert all('hop_ms' in row for row in rows)
    assert summary['provenance']['inflight'] == 3
    assert summary['provenance']['shape'] == 'acq|pre|infer|post'


def test_the_window_does_not_change_the_detections(tmp_path: Path) -> None:
    """Concurrency must be a change of *when*, never of *what*."""
    one, _ = _run(tmp_path / 'i1', inflight=1)
    three, _ = _run(tmp_path / 'i3', inflight=3)

    assert [row['seq'] for row in one] == [row['seq'] for row in three]
    assert [row['n_det'] for row in one] == [row['n_det'] for row in three]
    assert any(row['n_det'] for row in one), 'a comparison over no detections is empty'
