"""Render a finished run: detections sidecar + the clip -> an annotated video.

The pipeline's product is a detection list, so the only honest way to *see* a
run is to replay its detections over the frames it actually saw. This module
does that offline, for the monolith and for the 4-node ROS 2 chain alike, from
the artifacts both front-ends can write:

    <run>/detections.jsonl   seq -> final detections (yolocar.detections)
    <run>/records.jsonl      seq -> stage timings (the benchmark contract)

Why offline: encoding a 1280x720 frame with ``mp4v`` costs ~16 ms on this
board, i.e. about the whole pipeline, and a viewer node in the ROS chain would
mean a 2.7 MB/frame fan-out plus a process competing with the measured stages.
Rendering afterwards costs the measurement nothing.

The join is exact, never timestamp-based: ``seq`` is a global frame counter and
a file source re-decodes in the same order, so ``seq`` identifies the frame
unambiguously. The sidecar's header is validated against ``--source`` and
``--rotate``; a mismatch is an error, not a picture with wrong boxes.

Timing (``--timing``)
    ``wall``   quantised VFR: CFR at ``--base-fps`` with each frame repeated by
               its own duration, so the video lasts as long as the run did and
               the ROS chain's slower pacing is visible.
    ``source`` CFR at the clip's rate: what a slide deck wants, but it hides
               the throughput deficit.

Examples::

    python -m yolocar.demo --source assets/video/0000f77c-6257be58.mov \\
        --rotate 180 --detections results/demo/ros-N4-p2c1/detections.jsonl \\
        --records results/demo/ros-N4-p2c1/records.jsonl \\
        --out results/demo/ros-N4-p2c1.mp4 --hud

    python -m yolocar.demo --source assets/video/0000f77c-6257be58.mov \\
        --rotate 180 --compare \\
            monolith=results/demo/monolith/detections.jsonl,results/demo/monolith/records.jsonl \\
            ros-N4-p2c1=results/demo/ros-N4-p2c1/detections.jsonl,results/demo/ros-N4-p2c1/records.jsonl \\
        --out results/demo/compare.mp4 --hud

    python -m yolocar.demo ... --play      # window, paced by the real timestamps
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .acquisition import FrameSource, open_source
from .config import class_name
from .detections import DetectionRow, FrameDetections, iter_records, read_header
from .postprocessing import Detection
from .render import FrameSink, draw_detections, draw_hud, open_sink

LOGGER = logging.getLogger("yolocar.demo")

#: Longer gaps than this are clamped when quantising, so a stalled run cannot
#: turn a 5 s hiccup into a 10 s freeze at 120 fps.
MAX_GAP_S = 10.0


# ---------------------------------------------------------------------- sides
@dataclass
class Side:
    """One run to render: its detections, its timings, its label."""

    label: str
    header: dict[str, object]
    frames: dict[int, FrameDetections]
    order: list[int]
    #: ``seq -> seconds since this run's first frame`` (the panel's own clock).
    times: dict[int, float] = field(default_factory=dict)
    #: ``seq -> per-frame record`` from ``records.jsonl``, when supplied.
    timings: dict[int, dict[str, object]] = field(default_factory=dict)

    def rows(self, seq: int) -> tuple[DetectionRow, ...]:
        record = self.frames.get(seq)
        return record.rows if record is not None else ()

    @property
    def producer(self) -> str:
        return str(self.header.get("producer", "run"))


def _read_records(path: Path | None) -> dict[int, dict[str, object]]:
    """Per-frame timings keyed by ``seq``; empty when no file was given."""
    timings: dict[int, dict[str, object]] = {}
    if path is None:
        return timings
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                timings[int(record["seq"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return timings


def _absolute_source(value: object) -> str:
    path = Path(str(value))
    try:
        return str(path.resolve())
    except OSError:  # pragma: no cover - resolve() tolerates missing files
        return str(path)


def load_side(
    spec: str, *, loop: int | None = None, max_frames: int | None = None
) -> Side:
    """Parse ``LABEL=detections.jsonl[,records.jsonl]`` into a :class:`Side`."""
    try:
        label, rest = spec.split("=", 1)
    except ValueError as exc:
        raise ValueError(
            f"expected LABEL=DETECTIONS_JSONL[,RECORDS_JSONL], got {spec!r}"
        ) from exc
    label = label.strip()
    if not label:
        raise ValueError(f"missing label before '=': {spec!r}")
    parts = [part.strip() for part in rest.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"missing detections log: {spec!r}")

    detections_path = Path(parts[0])
    header = read_header(detections_path)
    records = list(iter_records(detections_path))
    if not records:
        raise ValueError(f"{detections_path}: the detections log has no frames")
    if loop is not None:
        records = [record for record in records if record.loop == loop]
        if not records:
            raise ValueError(f"{detections_path}: no frames with loop={loop}")
    if max_frames is not None:
        records = records[: int(max_frames)]

    first = records[0].t_mono
    return Side(
        label=label,
        header=header,
        frames={record.seq: record for record in records},
        order=[record.seq for record in records],
        times={record.seq: max(record.t_mono - first, 0.0) for record in records},
        timings=_read_records(Path(parts[1]) if len(parts) > 1 else None),
    )


def validate(side: Side, *, source: object, rotate: int) -> None:
    """Refuse to draw a sidecar onto frames it was not produced from."""
    header_source = _absolute_source(side.header.get("source", ""))
    wanted_source = _absolute_source(source)
    if header_source != wanted_source and Path(header_source).name != Path(
        wanted_source
    ).name:
        raise ValueError(
            f"{side.label}: detections were produced from "
            f"{side.header.get('source')!r}, not {str(source)!r}"
        )
    header_rotate = int(side.header.get("rotate", 0) or 0)
    if header_rotate != int(rotate):
        raise ValueError(
            f"{side.label}: detections were produced with rotate={header_rotate}, "
            f"but --rotate={rotate}: boxes would land in the wrong orientation"
        )


# ------------------------------------------------------------------- schedule
@dataclass(frozen=True)
class Output:
    """One output frame: which source seq each panel shows, and for how long."""

    reqs: tuple[int | None, ...]
    t: float
    repeats: int = 1


def quantise(
    times: Sequence[float], base_fps: float, tail: float | None = None
) -> list[int]:
    """How many CFR frames each sample occupies at ``base_fps``.

    Cumulative rounding, so pacing error stays bounded by half a base frame
    instead of drifting: before sample ``k`` the emitted count is
    ``round((t_k - t_0) * base_fps)``. Gaps are clamped to ``MAX_GAP_S`` and
    every sample keeps at least one frame.

    ``tail`` is how long the last sample is held (default: the median gap), so
    a composite can end with the slowest panel's frame duration.
    """
    if not times:
        return []
    if base_fps <= 0:
        raise ValueError(f"base fps must be positive, got {base_fps!r}")
    deltas = [max(0.0, times[i + 1] - times[i]) for i in range(len(times) - 1)]
    if tail is None:
        tail = statistics.median(deltas) if deltas else 1.0 / base_fps
    edges = list(times[1:]) + [times[-1] + max(float(tail), 0.0)]

    counts: list[int] = []
    previous = float(times[0])
    total = 0.0
    emitted = 0
    for edge in edges:
        total += min(max(float(edge) - previous, 0.0), MAX_GAP_S)
        previous = float(edge)
        target = int(round(total * base_fps))
        count = max(1, target - emitted)
        counts.append(count)
        emitted += count
    return counts


def single_schedule(side: Side, *, timing: str, base_fps: float) -> list[Output]:
    """One panel, paced by its own clock or at the clip's rate."""
    seqs = list(side.order)
    times = [side.times[seq] for seq in seqs]
    if timing == "source":
        return [Output(reqs=(seq,), t=time_s) for seq, time_s in zip(seqs, times)]
    counts = quantise(times, base_fps)
    return [
        Output(reqs=(seq,), t=time_s, repeats=count)
        for seq, time_s, count in zip(seqs, times, counts)
    ]


def merged_schedule(sides: Sequence[Side], *, base_fps: float) -> list[Output]:
    """Union of the panels' own clocks; each panel shows its newest frame.

    Both runs replay the same clip from ``seq`` 0, so one wall-clock timeline
    shows the monolith and the ROS chain at their real paces: the slower panel
    visibly stutters and the composite lasts as long as the slower run.
    """
    events: list[tuple[float, int, int]] = []
    for index, side in enumerate(sides):
        for seq in side.order:
            events.append((side.times[seq], index, seq))
    events.sort(key=lambda item: (item[0], item[1]))

    current: list[int | None] = [None] * len(sides)
    entries: list[Output] = []
    index = 0
    while index < len(events):
        time_s = events[index][0]
        # Everything that lands on the same instant becomes one frame, so a
        # panel is never shown with a stale `None` because the other side's
        # event at the same timestamp was processed one iteration later.
        while index < len(events) and events[index][0] == time_s:
            _, side_index, seq = events[index]
            current[side_index] = seq
            index += 1
        reqs = tuple(current)
        if entries and reqs == entries[-1].reqs:
            continue  # same picture, later timestamp: nothing to add
        entries.append(Output(reqs=reqs, t=time_s))

    counts = quantise([entry.t for entry in entries], base_fps, tail=_side_tail(sides))
    return [
        Output(reqs=entry.reqs, t=entry.t, repeats=count)
        for entry, count in zip(entries, counts)
    ]


def _side_tail(sides: Sequence[Side]) -> float:
    """Frame duration of the slowest panel, i.e. how long the last state holds."""
    tails = []
    for side in sides:
        times = [side.times[seq] for seq in side.order]
        deltas = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        tails.append(statistics.median(deltas) if deltas else 0.0)
    return max(tails) if tails else 0.0


def locked_schedule(sides: Sequence[Side]) -> list[Output]:
    """Panels in lockstep by source frame (``--align seq``): a quality compare."""
    common = set(sides[0].frames)
    for side in sides[1:]:
        common &= set(side.frames)
    if not common:
        raise ValueError("the compared runs share no frame (same clip, same loop?)")
    return [Output(reqs=tuple(seq for _ in sides), t=0.0) for seq in sorted(common)]


# --------------------------------------------------------------------- render
def _detections(rows: Sequence[DetectionRow]) -> list[Detection]:
    return [
        Detection(
            x_min=float(row[0]),
            y_min=float(row[1]),
            x_max=float(row[2]),
            y_max=float(row[3]),
            confidence=float(row[4]),
            class_id=int(row[5]),
            class_name=class_name(int(row[5])),
        )
        for row in rows
    ]


def _hud_lines(
    side: Side,
    seq: int,
    rows: Sequence[DetectionRow],
    elapsed: float,
    *,
    finished_at: float | None = None,
) -> list[str]:
    """Self-documenting caption: what this panel is, and what it cost."""
    record = side.timings.get(seq)
    head = f"{side.label}  seq {seq}"
    if record is not None and "loop" in record:
        head += f"  loop {record['loop']}"
    lines = [f"{head}  det {len(rows)}"]
    if record is not None and "latency_ms" in record:
        lines.append(
            "lat {lat:.1f} ms  pre {pre:.1f} infer {inf:.1f} post {post:.1f}".format(
                lat=float(record["latency_ms"]),
                pre=float(record.get("pre_ms", 0.0)),
                inf=float(record.get("infer_ms", 0.0)),
                post=float(record.get("post_ms", 0.0)),
            )
        )
    tail = f"t {elapsed:.2f} s"
    if record is not None and "transport_ms" in record:
        tail += f"  xport {float(record['transport_ms']):.1f} ms"
    if finished_at is not None:
        # The panel has no frames left while the composite still runs (it
        # finished sooner): say so instead of looking frozen for no reason.
        tail += f"  finished (run ended at {finished_at:.1f} s)"
    lines.append(tail)
    return lines


def _panel(image: np.ndarray, side: Side, seq: int) -> np.ndarray:
    """Detections only: the HUD is added *after* scaling, so it stays legible."""
    return draw_detections(image, _detections(side.rows(seq)))


def _scale_panel(canvas: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return canvas
    height, width = canvas.shape[:2]
    return cv2.resize(
        canvas,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _compose(
    canvases: Sequence[np.ndarray | None],
    sides: Sequence[Side],
    entry: Output,
    frame_size: tuple[int, int],
    scale: float,
    hud: bool,
) -> np.ndarray:
    """One output frame: per-panel resize, HUD, horizontal concatenation."""
    height = max(1, int(round(frame_size[1] * scale)))
    width = max(1, int(round(frame_size[0] * scale)))
    rendered: list[np.ndarray] = []
    for index, side in enumerate(sides):
        req = entry.reqs[index]
        canvas = canvases[index]
        if canvas is None or req is None:
            rendered.append(np.zeros((height, width, 3), dtype=np.uint8))
            continue
        panel = _scale_panel(canvas, scale)
        if hud:
            if panel is canvas:
                # A cached canvas is reused by later entries: never draw the
                # HUD onto it, or the captions accumulate.
                panel = canvas.copy()
            run_end = side.times[side.order[-1]]
            finished_at = (
                run_end if (req == side.order[-1] and entry.t > run_end + 0.05) else None
            )
            draw_hud(
                panel,
                _hud_lines(side, req, side.rows(req), side.times[req], finished_at=finished_at),
            )
        rendered.append(panel)
    return rendered[0] if len(rendered) == 1 else np.hstack(rendered)


def walk(
    sides: Sequence[Side],
    schedule: Sequence[Output],
    source: FrameSource,
    frame_size: tuple[int, int],
    *,
    hud: bool,
    scale: float,
    emit: Callable[[np.ndarray, Output], None],
) -> int:
    """Decode the source once, render each panel when its frame comes up, emit.

    Requests are monotone per panel, so a single decode cursor serves all of
    them: when the cursor reaches a panel's next request that panel is rendered
    once and reused until its next request. Two panels therefore cost one
    decode pass, not two. Returns the number of emitted output frames.
    """
    requests: list[list[int]] = [[] for _ in sides]
    for entry in schedule:
        for index, req in enumerate(entry.reqs):
            if req is None:
                continue
            if not requests[index] or requests[index][-1] != req:
                requests[index].append(req)

    canvases: list[np.ndarray | None] = [None] * len(sides)
    pointers = [0] * len(sides)
    served = [-1] * len(sides)
    position = 0
    emitted = 0

    def ready(entry: Output) -> bool:
        return all(
            req is None or req <= served[index]
            for index, req in enumerate(entry.reqs)
        )

    def flush() -> None:
        nonlocal position, emitted
        while position < len(schedule) and ready(schedule[position]):
            emit(
                _compose(
                    canvases, sides, schedule[position], frame_size, scale, hud
                ),
                schedule[position],
            )
            emitted += 1
            position += 1

    for frame in source:
        if position >= len(schedule):
            # Everything the schedule asked for has been emitted: stop decoding
            # (the clip usually runs longer than the artefact needs).
            break
        for index in range(len(sides)):
            while (
                pointers[index] < len(requests[index])
                and requests[index][pointers[index]] < frame.seq
            ):
                # The log has no entry for this frame (or it was filtered out):
                # move past the request and keep the newest canvas.
                pointers[index] += 1
            if (
                pointers[index] < len(requests[index])
                and requests[index][pointers[index]] == frame.seq
            ):
                canvases[index] = _panel(frame.image, sides[index], frame.seq)
                pointers[index] += 1
            if pointers[index]:
                served[index] = requests[index][pointers[index] - 1]
        flush()

    # The source can end before every request was satisfied (a truncated log,
    # or --max-frames); emit what we have rather than dropping the tail.
    while position < len(schedule):
        emit(_compose(canvases, sides, schedule[position], frame_size, scale, hud), schedule[position])
        emitted += 1
        position += 1
    return emitted


# ------------------------------------------------------------------ rendering
def _sink_size(frame_size: tuple[int, int], panels: int, scale: float) -> tuple[int, int]:
    return (
        max(1, int(round(frame_size[0] * scale))) * panels,
        max(1, int(round(frame_size[1] * scale))),
    )


def render(
    sides: Sequence[Side],
    *,
    source_path: Path | str,
    rotate: int,
    timing: str,
    base_fps: float,
    hud: bool,
    scale: float,
    play: bool,
    out: Path | None,
    align: str = "wall",
    until: float | None = None,
    title: str = "yolocar.demo",
) -> dict[str, object]:
    """Render ``sides`` to ``out``, or to a window when ``play`` is set."""
    source = open_source(source_path, rotate, loops=1)
    for side in sides:
        try:
            validate(side, source=source_path, rotate=rotate)
        except ValueError:
            source.close()
            raise
    frame_size = source.size
    if not frame_size or not frame_size[0] or not frame_size[1]:
        source.close()
        raise ValueError(f"{source_path}: the decoder did not report a frame size")
    source_fps = source.fps or 30.0

    if len(sides) > 1:
        schedule = (
            locked_schedule(sides)
            if align == "seq"
            else merged_schedule(sides, base_fps=base_fps)
        )
    else:
        schedule = single_schedule(sides[0], timing=timing, base_fps=base_fps)
    if not schedule:
        source.close()
        raise ValueError("nothing to render: the detections log has no frames")

    # Lockstep is necessarily CFR; `--timing source` is CFR by construction.
    locked = len(sides) > 1 and align == "seq"
    output_fps = source_fps if (locked or timing == "source") else base_fps
    # For a lockstep compare, pace at the clip rate even though the schedule
    # carries no timestamps of its own.
    if locked:
        schedule = [
            Output(reqs=entry.reqs, t=index / output_fps)
            for index, entry in enumerate(schedule)
        ]

    if until is not None and until > 0:
        # Trim the output timeline (seconds of the render, not of the clip):
        # useful when the interesting part is the two runs diverging, not the
        # freeze after the faster one has finished. The cut is exact, so the
        # last entry does not overrun it.
        kept: list[Output] = []
        total = 0
        for entry in schedule:
            if entry.t > until:
                break
            kept.append(entry)
            total += entry.repeats
        limit = max(1, int(round(until * output_fps)))
        if kept and total > limit:
            last = kept[-1]
            kept[-1] = Output(
                reqs=last.reqs, t=last.t, repeats=max(1, last.repeats - (total - limit))
            )
        if not kept:
            source.close()
            raise ValueError(f"--until {until:g}s leaves no frames to render")
        schedule = kept

    sink: FrameSink | None = None
    size = _sink_size(frame_size, len(sides), scale)
    if not play:
        if out is None:
            source.close()
            raise ValueError("--out is required unless --play is given")
        sink = open_sink(Path(out), output_fps, size)
        LOGGER.info(
            "rendering %d output frames (%d panels) at %.1f fps -> %s",
            sum(entry.repeats for entry in schedule),
            len(sides),
            output_fps,
            out,
        )

    started = time.monotonic()
    shown = 0
    quit_early = False

    def emit(canvas: np.ndarray, entry: Output) -> None:
        nonlocal shown, quit_early
        if quit_early:
            return
        if play:
            deadline = started + entry.t
            delay = int(round(max(deadline - time.monotonic(), 0.0) * 1e3))
            cv2.imshow(title, canvas)
            key = cv2.waitKey(max(1, delay)) & 0xFF
            if key in (27, ord("q")):
                quit_early = True
            shown += 1
            return
        assert sink is not None
        for _ in range(entry.repeats):
            sink.write(canvas)
            shown += 1

    try:
        emitted = walk(
            sides,
            schedule,
            source,
            frame_size,
            hud=hud,
            scale=scale,
            emit=emit,
        )
    finally:
        source.close()
        if sink is not None:
            sink.close()
        if play:
            try:
                cv2.destroyAllWindows()
            except cv2.error:  # pragma: no cover - headless environment
                pass

    duration_s = time.monotonic() - started
    summary: dict[str, object] = {
        "panels": [side.label for side in sides],
        "schedule_entries": len(schedule),
        "output_frames": shown,
        "walked_frames": emitted,
        "output_fps": round(output_fps, 3),
        "timing": "source" if (locked or timing == "source") else timing,
        "align": align if len(sides) > 1 else None,
        "hud": hud,
        "scale": scale,
        "video_duration_s": round(sum(entry.repeats for entry in schedule) / output_fps, 3),
        "until": until,
        "render_time_s": round(duration_s, 3),
        "out": None if out is None else str(out),
        "interrupted": quit_early,
    }
    LOGGER.info(
        "rendered %d frames (%d entries) in %.1f s -> %s",
        shown,
        len(schedule),
        duration_s,
        out if out is not None else "window",
    )
    return summary


# ------------------------------------------------------------------------ cli
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolocar.demo",
        description=(
            "Render detections from a finished run (detections.jsonl) onto the "
            "clip it processed. Offline: nothing here runs inside a measured "
            "pipeline."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("assets/video/0000f77c-6257be58.mov"),
        help="the clip the run processed (re-decoded here, in the same order)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        choices=(0, 90, 180, 270),
        help="must match the rotation the run used (validated against the header)",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help="detections.jsonl written by --detections (single-panel mode)",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="records.jsonl from the same run: supplies the HUD's timings",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        default=None,
        help=(
            "side-by-side mode: LABEL=DETECTIONS_JSONL[,RECORDS_JSONL] twice "
            "(e.g. monolith=... ros-N4-p2c1=...)"
        ),
    )
    parser.add_argument(
        "--align",
        choices=("wall", "seq"),
        default="wall",
        help=(
            "composite alignment: wall = each panel at its own pace (the "
            "overhead becomes visible); seq = lockstep by clip frame (quality "
            "compare, CFR at the clip rate)"
        ),
    )
    parser.add_argument(
        "--timing",
        choices=("wall", "source"),
        default="wall",
        help=(
            "wall = quantised VFR at --base-fps, so the video lasts as long as "
            "the run did; source = CFR at the clip rate (hides the pacing)"
        ),
    )
    parser.add_argument(
        "--base-fps",
        type=float,
        default=120.0,
        help="CFR base rate for --timing wall (pacing granularity: 1/base fps)",
    )
    parser.add_argument("--hud", dest="hud", action="store_true", default=True)
    parser.add_argument(
        "--no-hud", dest="hud", action="store_false", help="draw boxes only"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="resize each panel (default: 1.0 single, 0.5 in --compare)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=None,
        help="only render frames from this replay index (0 = first pass)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="render at most this many frames per panel",
    )
    parser.add_argument(
        "--until",
        type=float,
        default=None,
        help="stop the output timeline after this many seconds (of the render, not the clip)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="show a window paced by the recorded timestamps instead of writing a file",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output video (required without --play)"
    )
    parser.add_argument("--label", default=None, help="panel label (single mode)")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.compare:
            sides = [
                load_side(spec, loop=args.loop, max_frames=args.max_frames)
                for spec in args.compare
            ]
        elif args.detections:
            spec = f"{args.label or 'run'}={args.detections}"
            if args.records:
                spec += f",{args.records}"
            sides = [load_side(spec, loop=args.loop, max_frames=args.max_frames)]
        else:
            parser.error("either --detections or --compare is required")
    except ValueError as exc:
        parser.error(str(exc))

    for side in sides:
        try:
            validate(side, source=args.source, rotate=args.rotate)
        except ValueError as exc:
            parser.error(str(exc))

    scale = args.scale
    if scale is None:
        scale = 0.5 if len(sides) > 1 else 1.0
    if scale < 0.1:
        parser.error("--scale must be >= 0.1")
    if len(sides) > 1 and args.detections:
        parser.error("--compare replaces --detections")

    summary = render(
        sides,
        source_path=args.source,
        rotate=int(args.rotate),
        timing=str(args.timing),
        base_fps=float(args.base_fps),
        hud=bool(args.hud),
        scale=float(scale),
        play=bool(args.play),
        out=Path(args.out) if args.out else None,
        align=str(args.align),
        until=None if args.until is None else float(args.until),
        title="yolocar.demo " + " | ".join(side.label for side in sides),
    )
    LOGGER.info("demo summary: %s", json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
