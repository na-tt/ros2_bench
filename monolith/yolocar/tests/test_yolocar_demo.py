"""Tests for the offline renderer (sidecar join, timing quantiser, composite).

The clip is synthesised in ``tmp_path`` and the sidecars are written by the real
writer, so these run without a GPU, an engine or the shipped footage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
cv2 = pytest.importorskip("cv2")

from yolocar.demo import (  # noqa: E402
    Output,
    Side,
    _hud_lines,
    load_side,
    locked_schedule,
    merged_schedule,
    quantise,
    render,
    single_schedule,
    validate,
    walk,
)
from yolocar.detections import DetectionLog, build_header  # noqa: E402
from yolocar.postprocessing import Detection  # noqa: E402

SIZE = (64, 48)
FPS = 30.0
CAR = 2


# ---------------------------------------------------------------- fixtures
def _clip(path: Path, frames: int = 8) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((SIZE[1], SIZE[0], 3), 20 + index * 5, np.uint8))
    writer.release()
    return path


def _sidecar(
    path: Path,
    clip: Path,
    *,
    frames: int = 8,
    dt: float = 0.025,
    rotate: int = 0,
    source: Path | str | None = None,
    boxes: bool = True,
    producer: str = "test",
) -> Path:
    header = build_header(
        source=clip if source is None else source,
        rotate=rotate,
        model="engine",
        image_size=640,
        conf_threshold=0.25,
        iou_threshold=0.45,
        agnostic_nms=False,
        class_ids=(0, 2),
        producer=producer,
    )
    detection = Detection(
        x_min=4.0, y_min=4.0, x_max=40.0, y_max=36.0,
        confidence=0.5, class_id=CAR, class_name="car",
    )
    with DetectionLog(path, header=header) as log:
        for seq in range(frames):
            log.write(
                seq=seq,
                loop=0,
                t_mono=100.0 + seq * dt,
                detections=[detection] if boxes else [],
            )
    return path


def _records(path: Path, frames: int = 8, *, transport: bool = False) -> Path:
    lines = []
    for seq in range(frames):
        record = {
            "seq": seq,
            "loop": 0,
            "t_mono": 100.0 + seq * 0.025,
            "latency_ms": 17.0,
            "pre_ms": 4.0,
            "infer_ms": 8.0,
            "post_ms": 5.0,
        }
        if transport:
            record["transport_ms"] = 60.0
        lines.append(__import__("json").dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- quantiser
def test_quantise_uniform_deltas_is_exact() -> None:
    times = [index * 0.025 for index in range(40)]
    counts = quantise(times, 120.0)
    assert counts == [3] * 40
    # Cumulative rounding: the emitted total matches the elapsed time exactly.
    assert sum(counts) == round((times[-1] + 0.025) * 120.0)


def test_quantise_keeps_at_least_one_frame_per_sample() -> None:
    assert quantise([0.0, 0.0, 0.0], 120.0) == [1, 1, 1]


def test_quantise_bounds_the_pacing_error() -> None:
    times = [0.0, 0.033, 0.041, 0.19, 0.2, 0.35]
    counts = quantise(times, 120.0)
    before = 0
    for time_s, count in zip(times, counts):
        # Before sample k the emitted count must match its own timestamp to
        # within half a base frame (plus the one-frame floor of a clamped gap).
        assert abs(before / 120.0 - time_s) <= 1.0 / 120.0
        before += count


def test_quantise_clamps_a_pathological_gap() -> None:
    counts = quantise([0.0, 60.0], 120.0)
    assert counts[0] == 10 * 120  # MAX_GAP_S, not a 60 s freeze


def test_quantise_rejects_a_non_positive_base_rate() -> None:
    with pytest.raises(ValueError, match="base fps"):
        quantise([0.0, 1.0], 0.0)


# ---------------------------------------------------------------- schedules
def test_single_schedule_source_mode_is_one_frame_each(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    side = load_side(f"run={_sidecar(tmp_path / 'dets.jsonl', clip)}")
    schedule = single_schedule(side, timing="source", base_fps=120.0)
    assert [entry.repeats for entry in schedule] == [1] * 8
    assert [entry.reqs for entry in schedule] == [(seq,) for seq in range(8)]


def test_single_schedule_wall_mode_follows_the_recorded_clock(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    side = load_side(f"run={_sidecar(tmp_path / 'dets.jsonl', clip, dt=0.1)}")
    schedule = single_schedule(side, timing="wall", base_fps=120.0)
    assert [entry.repeats for entry in schedule] == [12] * 8


def test_merged_schedule_is_as_long_as_the_slower_run(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    fast = load_side(f"fast={_sidecar(tmp_path / 'a.jsonl', clip, dt=0.025)}")
    slow = load_side(f"slow={_sidecar(tmp_path / 'b.jsonl', clip, dt=0.100)}")
    schedule = merged_schedule([fast, slow], base_fps=120.0)
    # The composite lasts exactly as long as the slowest panel (8 frames at
    # 100 ms, plus one held frame), not as long as the fast one (0.2 s).
    slow_alone = sum(
        entry.repeats for entry in single_schedule(slow, timing="wall", base_fps=120.0)
    )
    assert sum(entry.repeats for entry in schedule) == pytest.approx(slow_alone, abs=1)
    assert slow_alone > 3 * sum(
        entry.repeats for entry in single_schedule(fast, timing="wall", base_fps=120.0)
    )
    # Both panels are always present once the run has started.
    assert all(all(req is not None for req in entry.reqs) for entry in schedule)
    # Requests are monotone per panel.
    for index in (0, 1):
        reqs = [entry.reqs[index] for entry in schedule]
        assert reqs == sorted(reqs)


def test_locked_schedule_uses_the_common_frames(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    left = load_side(f"left={_sidecar(tmp_path / 'a.jsonl', clip, frames=8)}")
    right = load_side(f"right={_sidecar(tmp_path / 'b.jsonl', clip, frames=5)}")
    schedule = locked_schedule([left, right])
    assert [entry.reqs for entry in schedule] == [(seq, seq) for seq in range(5)]


# ---------------------------------------------------------------- sidecar io
def test_load_side_parses_label_and_records(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    dets = _sidecar(tmp_path / "dets.jsonl", clip)
    records = _records(tmp_path / "records.jsonl")
    side = load_side(f"monolith={dets},{records}")
    assert side.label == "monolith"
    assert side.producer == "test"
    assert side.timings[3]["latency_ms"] == 17.0
    assert side.times[0] == 0.0
    assert side.times[4] == pytest.approx(0.1)


def test_load_side_filters_by_loop_and_frame_cap(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    side = load_side(f"run={_sidecar(tmp_path / 'd.jsonl', clip)}", max_frames=3)
    assert side.order == [0, 1, 2]
    with pytest.raises(ValueError, match="no frames with loop=1"):
        load_side(f"run={_sidecar(tmp_path / 'd.jsonl', clip)}", loop=1)


@pytest.mark.parametrize("spec", ["nodots", "=dets.jsonl", "label="])
def test_load_side_rejects_a_malformed_spec(spec: str) -> None:
    with pytest.raises(ValueError):
        load_side(spec)


def test_validate_rejects_a_different_clip(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    other = _clip(tmp_path / "other.mp4")
    side = load_side(f"run={_sidecar(tmp_path / 'd.jsonl', clip, source=other)}")
    with pytest.raises(ValueError, match="produced from"):
        validate(side, source=clip, rotate=0)


def test_validate_rejects_a_rotation_mismatch(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    side = load_side(f"run={_sidecar(tmp_path / 'd.jsonl', clip, rotate=180)}")
    with pytest.raises(ValueError, match="rotate=180"):
        validate(side, source=clip, rotate=0)


# --------------------------------------------------------------------- walk
def _fake_source(clip: Path):
    from yolocar.acquisition import open_source

    return open_source(clip, 0, loops=1)


def test_walk_emits_one_frame_per_schedule_entry(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    side = load_side(f"run={_sidecar(tmp_path / 'd.jsonl', clip)}")
    schedule = single_schedule(side, timing="wall", base_fps=120.0)
    source = _fake_source(clip)
    seen: list[tuple[int, ...]] = []
    try:
        emitted = walk(
            [side], schedule, source, SIZE,
            hud=False, scale=1.0,
            emit=lambda _frame, entry: seen.append(entry.reqs),
        )
    finally:
        source.close()
    assert emitted == len(schedule)
    assert seen == [entry.reqs for entry in schedule]


def test_walk_serves_both_panels_from_one_decode_pass(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    left = load_side(f"left={_sidecar(tmp_path / 'a.jsonl', clip, dt=0.025)}")
    right = load_side(f"right={_sidecar(tmp_path / 'b.jsonl', clip, dt=0.075)}")
    schedule = merged_schedule([left, right], base_fps=120.0)
    source = _fake_source(clip)
    frames: list[np.ndarray] = []
    try:

        def emit(frame: np.ndarray, _entry: Output) -> None:
            frames.append(frame.copy())

        emitted = walk(
            [left, right], schedule, source, SIZE,
            hud=True, scale=0.5, emit=emit,
        )
    finally:
        source.close()
    assert emitted == len(schedule)
    assert frames and frames[0].shape == (SIZE[1] // 2, SIZE[0], 3)


def test_walk_never_draws_the_hud_onto_a_cached_canvas(tmp_path: Path) -> None:
    """A panel reused by several entries must not accumulate captions."""
    clip = _clip(tmp_path / "clip.mp4")
    fast = load_side(f"fast={_sidecar(tmp_path / 'a.jsonl', clip, dt=0.025)}")
    slow = load_side(f"slow={_sidecar(tmp_path / 'b.jsonl', clip, dt=0.100)}")
    schedule = merged_schedule([fast, slow], base_fps=120.0)
    frames: list[np.ndarray] = []
    source = _fake_source(clip)
    try:
        walk(
            [fast, slow], schedule, source, SIZE,
            hud=True, scale=1.0,
            emit=lambda frame, _entry: frames.append(frame.copy()),
        )
    finally:
        source.close()

    # Entries where the fast panel kept its frame: its half must be identical,
    # i.e. the HUD was drawn on a copy of the cached canvas, not on it.
    compared = 0
    for index in range(1, len(schedule)):
        if schedule[index].reqs[0] == schedule[index - 1].reqs[0]:
            left = SIZE[0]
            assert np.array_equal(frames[index][:, :left], frames[index - 1][:, :left])
            compared += 1
    assert compared > 5


def test_walk_drains_entries_the_log_does_not_cover(tmp_path: Path) -> None:
    """A truncated log (or --max-frames) must not drop the tail silently."""
    clip = _clip(tmp_path / "clip.mp4", frames=8)
    side = load_side(f"run={_sidecar(tmp_path / 'd.jsonl', clip, frames=4)}")
    # Ask for frames the sidecar does not have: the walk must still emit them.
    schedule = [Output(reqs=(seq,), t=seq * 0.025, repeats=1) for seq in range(8)]
    source = _fake_source(clip)
    emitted = 0
    try:
        emitted = walk(
            [side], schedule, source, SIZE,
            hud=False, scale=1.0, emit=lambda _f, _e: None,
        )
    finally:
        source.close()
    assert emitted == 8


# --------------------------------------------------------------------- hud
def test_hud_reports_latency_and_transport(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    dets = _sidecar(tmp_path / "d.jsonl", clip)
    side = load_side(f"ros-N4-p2c1={dets},{_records(tmp_path / 'r.jsonl', transport=True)}")
    lines = _hud_lines(side, 3, side.rows(3), side.times[3])
    assert lines[0].startswith("ros-N4-p2c1  seq 3")
    assert "lat 17.0 ms" in lines[1]
    assert "xport 60.0 ms" in lines[2]


def test_hud_without_records_still_labels_the_panel(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    side = load_side(f"run={_sidecar(tmp_path / 'd.jsonl', clip)}")
    lines = _hud_lines(side, 0, side.rows(0), side.times[0])
    assert len(lines) == 2
    assert "det 1" in lines[0]


def test_hud_marks_a_panel_that_has_finished(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    side = load_side(f"run={_sidecar(tmp_path / 'd.jsonl', clip)}")
    running = _hud_lines(side, 7, side.rows(7), side.times[7])
    assert "finished" not in running[-1]
    done = _hud_lines(side, 7, side.rows(7), side.times[7], finished_at=0.175)
    assert "finished (run ended at 0.2 s)" in done[-1]


# ------------------------------------------------------------------- render
def test_render_writes_a_video(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    dets = _sidecar(tmp_path / "d.jsonl", clip, dt=0.1)
    out = tmp_path / "out.mp4"
    summary = render(
        [load_side(f"run={dets}")],
        source_path=clip,
        rotate=0,
        timing="wall",
        base_fps=60.0,
        hud=True,
        scale=1.0,
        play=False,
        out=out,
    )
    assert out.exists()
    assert summary["output_frames"] == 8 * 6  # 100 ms frames at 60 fps
    assert summary["timing"] == "wall"
    capture = cv2.VideoCapture(str(out))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == summary["output_frames"]
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == SIZE[0]
    finally:
        capture.release()


def test_render_compare_stacks_the_panels(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    left = _sidecar(tmp_path / "a.jsonl", clip)
    right = _sidecar(tmp_path / "b.jsonl", clip, dt=0.075)
    out = tmp_path / "compare.mp4"
    summary = render(
        [load_side(f"left={left}"), load_side(f"right={right}")],
        source_path=clip,
        rotate=0,
        timing="wall",
        base_fps=60.0,
        hud=True,
        scale=0.5,
        play=False,
        out=out,
        align="wall",
    )
    assert summary["panels"] == ["left", "right"]
    assert summary["align"] == "wall"
    capture = cv2.VideoCapture(str(out))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == SIZE[0]
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == summary["output_frames"]
    finally:
        capture.release()


def test_render_align_seq_is_lockstep_cfr(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    left = _sidecar(tmp_path / "a.jsonl", clip)
    right = _sidecar(tmp_path / "b.jsonl", clip, frames=5)
    summary = render(
        [load_side(f"left={left}"), load_side(f"right={right}")],
        source_path=clip,
        rotate=0,
        timing="wall",
        base_fps=60.0,
        hud=False,
        scale=1.0,
        play=False,
        out=tmp_path / "lockstep.mp4",
        align="seq",
    )
    assert summary["output_frames"] == 5
    assert summary["timing"] == "source"


def test_render_until_trims_the_output_timeline(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    dets = _sidecar(tmp_path / "d.jsonl", clip, dt=0.1)
    full = render(
        [load_side(f"run={dets}")], source_path=clip, rotate=0, timing="wall",
        base_fps=60.0, hud=False, scale=1.0, play=False, out=tmp_path / "full.mp4",
    )
    trimmed = render(
        [load_side(f"run={dets}")], source_path=clip, rotate=0, timing="wall",
        base_fps=60.0, hud=False, scale=1.0, play=False, out=tmp_path / "short.mp4",
        until=0.35,
    )
    assert trimmed["output_frames"] < full["output_frames"]
    assert trimmed["video_duration_s"] <= 0.35 + 1e-6
    assert trimmed["until"] == 0.35


def test_render_requires_an_output_without_play(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    side = load_side(f"run={_sidecar(tmp_path / 'd.jsonl', clip)}")
    with pytest.raises(ValueError, match="--out is required"):
        render(
            [side],
            source_path=clip, rotate=0, timing="source", base_fps=60.0,
            hud=False, scale=1.0, play=False, out=None,
        )


def test_render_refuses_a_sidecar_from_another_clip(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    other = _clip(tmp_path / "other.mp4")
    side = load_side(f"run={_sidecar(tmp_path / 'd.jsonl', other)}")
    with pytest.raises(ValueError, match="produced from"):
        render(
            [side], source_path=clip, rotate=0, timing="source", base_fps=60.0,
            hud=False, scale=1.0, play=False, out=tmp_path / "o.mp4",
        )
