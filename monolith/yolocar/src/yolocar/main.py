"""Standalone single-process entry point / benchmark for the yolocar pipeline.

Usage examples::

    python -m yolocar.main --source assets/video/0000f77c-6257be58.mov --rotate 180
    python -m yolocar.main --source 0 --draw
    python -m yolocar.main --source assets/video --loops 10 --duration 600 \\
        --records results/monolith.jsonl --json results/monolith.json

This script is the "monolith" baseline: one process, no middleware, no image
serialisation. It contains no pipeline logic -- sources come from
:mod:`yolocar.acquisition`, stages from :mod:`yolocar.processor`, annotation
from :mod:`yolocar.render`. Only the CLI, the timing loop and the summary live
here.

Measurement policy
------------------
Nothing is discarded at runtime. Every frame is written to ``--records`` as one
JSON object, tagged with its monotonic timestamp, replay index (``loop``) and
per-stage duration; the steady-state window is chosen afterwards from that
stream. There is deliberately no warmup gate: startup and settling are visible
in the records (``events`` in the summary) instead of being assumed away.

Clocks are not pinned, the garbage collector is not touched and the desktop
session keeps running: the machine's natural noise floor is part of what is
being measured. Compare conditions with interleaved trials, not back-to-back
blocks, or DVFS/thermal drift will be attributed to whichever ran second.

Annotation (``--draw``, ``--output``) is **debug only**. Drawing and video
encoding are wall-clock work that no node in the measured chain performs, so the
presentation layer is timed into its own bucket and subtracted from the
reported throughput (``fps``). ``fps_wall`` keeps the raw figure.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .acquisition import FrameSource, open_source
from .config import DEFAULT_OUTPUT_FPS, ROAD_CLASS_IDS, YoloCarConfig
from .postprocessing import Detection
from .processor import YoloCarProcessor
from .render import draw_detections, open_sink

LOGGER = logging.getLogger("yolocar")

PROGRESS_EVERY = 50

#: Stages that make up the measured pipeline latency (acquisition excluded).
PIPELINE_STAGES = ("preprocess", "inference", "postprocess")

_SENTINEL = object()


# ---------------------------------------------------------------- provenance
def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _git_commit() -> str | None:
    """Short HEAD sha of the checkout this module was imported from, if any."""
    for directory in (Path.cwd(), Path(__file__).resolve().parents[4]):
        try:
            result = subprocess.run(
                ["git", "-C", str(directory), "rev-parse", "--short", "HEAD"],
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
    return {"path": str(path), "bytes": stat.st_size, "mtime": stat.st_mtime}


def _provenance(config: YoloCarConfig) -> dict[str, object]:
    """Enough context to tell whether two runs are comparable at all."""
    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "git_commit": _git_commit(),
        "board": _read_text("/proc/device-tree/model"),
        "cpu_governor": _read_text(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "engine": _file_identity(Path(config.model_path)),
        "source": str(config.source),
    }


# ------------------------------------------------------------------- records
class _RecordWriter:
    """One JSON object per frame (JSONL); a no-op when no path was given."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._file = None

    def __enter__(self) -> "_RecordWriter":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", encoding="utf-8")
        return self

    def write(self, record: dict[str, object]) -> None:
        if self._file is not None:
            self._file.write(json.dumps(record) + "\n")

    def __exit__(self, *exc_info: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


# ----------------------------------------------------------------- benchmark
def _latency_stats(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {}
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _detection_summary(
    frames: int, per_frame: list[int], by_class: dict[str, int]
) -> dict[str, object]:
    total = int(sum(per_frame))
    return {
        "total": total,
        "per_frame_mean": (total / frames) if frames else 0.0,
        "per_frame_max": int(max(per_frame)) if per_frame else 0,
        "by_class": dict(sorted(by_class.items(), key=lambda item: -item[1])),
    }


def run_benchmark(config: YoloCarConfig) -> dict[str, object]:
    """Run the full pipeline over ``config.source`` and return summary stats."""
    source: FrameSource = open_source(
        config.source, config.rotate, loops=config.loops
    )
    LOGGER.info(
        "source: %s (kind=%s, fps=%.1f, rotate=%d, loops=%d)",
        source.description,
        source.kind,
        source.fps,
        config.rotate,
        config.loops,
    )

    processor = YoloCarProcessor(config)
    sink = (
        None
        if config.output_path is None
        else open_sink(
            Path(config.output_path),
            source.fps or DEFAULT_OUTPUT_FPS,
            source.size,
            image_source=source.kind == "images",
        )
    )
    show = bool(config.draw)

    acquisition_ms: list[float] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    presentation_ms: list[float] = []
    frame_ms: list[float] = []
    detections_per_frame: list[int] = []
    detections_by_class: dict[str, int] = {}
    frames = 0
    replays = 0
    interrupted = False
    events: dict[str, float] = {}
    records = _RecordWriter(config.records_path)

    t_start = time.monotonic()
    events["start"] = t_start

    with records:
        try:
            processor.load()
            events["engine_loaded"] = time.monotonic()
            # The measurement clock starts when the engine is ready: startup is
            # reported separately and must not shrink a --duration budget.
            t_measure_start = events["engine_loaded"]

            iterator = iter(source)
            while True:
                if config.max_frames is not None and frames >= config.max_frames:
                    break
                if (
                    config.duration_s is not None
                    and (time.monotonic() - t_measure_start) >= config.duration_s
                ):
                    break

                # Stage 1 (including the rotation correction) is timed
                # separately: it is the decoder's cost, not the pipeline's.
                t_acq = time.monotonic()
                frame = next(iterator, _SENTINEL)
                if frame is _SENTINEL:
                    break
                t0 = time.monotonic()
                if frames == 0:
                    events["first_frame"] = t0
                replays = frame.loop + 1

                tensor, letterbox = processor.preprocess(frame.image)
                t1 = time.monotonic()
                outputs = processor.infer(tensor)
                t2 = time.monotonic()
                detections: list[Detection] = processor.postprocess(outputs, letterbox)
                t3 = time.monotonic()

                present = 0.0
                if sink is not None or show:
                    t_present = time.monotonic()
                    annotated = (
                        frame.image if not detections else draw_detections(frame.image, detections)
                    )
                    if sink is not None:
                        sink.write(annotated)
                    if show:
                        try:
                            cv2.imshow("yolocar", annotated)
                            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                                LOGGER.info("display: quitting on user request")
                                break
                        except cv2.error as exc:  # headless environment
                            LOGGER.warning(
                                "cannot show frames (%s); disabling display", exc
                            )
                            show = False
                    present = (time.monotonic() - t_present) * 1e3
                    presentation_ms.append(present)

                acquired = (t0 - t_acq) * 1e3
                preprocessed = (t1 - t0) * 1e3
                inferred = (t2 - t1) * 1e3
                postprocessed = (t3 - t2) * 1e3

                acquisition_ms.append(acquired)
                preprocess_ms.append(preprocessed)
                inference_ms.append(inferred)
                postprocess_ms.append(postprocessed)
                frame_ms.append(preprocessed + inferred + postprocessed)

                detections_per_frame.append(len(detections))
                for detection in detections:
                    detections_by_class[detection.class_name] = (
                        detections_by_class.get(detection.class_name, 0) + 1
                    )
                frames += 1

                records.write(
                    {
                        "seq": frame.seq,
                        "loop": frame.loop,
                        "t_mono": round(t_acq, 6),
                        "acq_ms": round(acquired, 4),
                        "pre_ms": round(preprocessed, 4),
                        "infer_ms": round(inferred, 4),
                        "post_ms": round(postprocessed, 4),
                        "present_ms": round(present, 4),
                        "latency_ms": round(preprocessed + inferred + postprocessed, 4),
                        "n_det": len(detections),
                    }
                )

                if frames % PROGRESS_EVERY == 0:
                    LOGGER.info(
                        "frame %d (loop %d): %.1f ms (acq %.1f / pre %.1f / infer %.1f / post %.1f), %d dets",
                        frames,
                        frame.loop,
                        frame_ms[-1],
                        acquisition_ms[-1],
                        preprocess_ms[-1],
                        inference_ms[-1],
                        postprocess_ms[-1],
                        len(detections),
                    )
            wall_time = time.monotonic() - t_measure_start
        except KeyboardInterrupt:
            LOGGER.warning("interrupted by user; reporting partial results")
            interrupted = True
            wall_time = float("nan")
        finally:
            events["end"] = time.monotonic()
            source.close()
            if sink is not None:
                sink.close()
            processor.close()
            if show:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

    presentation_time_s = sum(presentation_ms) / 1e3
    benchmark_time_s = (
        max(wall_time - presentation_time_s, 0.0)
        if np.isfinite(wall_time)
        else float("nan")
    )

    summary: dict[str, object] = {
        "pipeline": "yolocar",
        "model": str(config.model_path),
        "source": source.description,
        "frames": frames,
        #: Replay cap from the config, and how many replays actually completed.
        "loops": config.loops,
        "replays": replays,
        "duration_s": config.duration_s,
        "interrupted": interrupted,
        "wall_time_s": wall_time,
        #: Wall time with the debug-only presentation layer removed: the figure
        #: to compare across front-ends.
        "benchmark_time_s": benchmark_time_s,
        "fps": (frames / benchmark_time_s) if frames and benchmark_time_s > 0 else 0.0,
        #: Raw wall-clock rate, i.e. including annotation/encoding.
        "fps_wall": (frames / wall_time) if frames and wall_time > 0 else 0.0,
        "latency_ms": _latency_stats(frame_ms),
        "stage_ms": {
            "acquisition": _latency_stats(acquisition_ms),
            "preprocess": _latency_stats(preprocess_ms),
            "inference": _latency_stats(inference_ms),
            "postprocess": _latency_stats(postprocess_ms),
            #: Debug only: annotation + encoding, excluded from ``fps``.
            "presentation": _latency_stats(presentation_ms),
        },
        #: Monotonic timestamps (seconds, same clock as the records): startup and
        #: settling are observed afterwards, never declared up front.
        "events": {name: round(value, 6) for name, value in events.items()},
        "detections": _detection_summary(frames, detections_per_frame, detections_by_class),
        "provenance": _provenance(config),
        "records": None if config.records_path is None else str(config.records_path),
        "config": {
            "conf_threshold": config.conf_threshold,
            "iou_threshold": config.iou_threshold,
            "image_size": config.image_size,
            "target_class_ids": list(config.target_class_ids),
            "agnostic_nms": config.agnostic_nms,
            "device": config.device,
            "rotate": config.rotate,
            "max_frames": config.max_frames,
        },
    }
    LOGGER.info("\n%s", format_summary(summary))
    if config.json_path is not None:
        target = Path(config.json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("summary written to %s", target)
    return summary


def format_summary(summary: dict[str, object]) -> str:
    """Human readable multi-line report."""
    latency = summary.get("latency_ms") or {}
    stages = summary.get("stage_ms") or {}
    detections = summary.get("detections") or {}
    events = summary.get("events") or {}
    replays = int(summary.get("replays") or 1)
    frames_line = f"frames   {summary.get('frames')}"
    if replays > 1:
        frames_line += f" ({replays} replays)"
    lines = [
        "=" * 62,
        f"yolocar  model={summary.get('model')}",
        f"source   {summary.get('source')}",
        frames_line,
        f"throughput {float(summary.get('fps') or 0.0):.2f} fps"
        f" (benchmark window {float(summary.get('benchmark_time_s') or 0.0):.1f} s)",
    ]
    if latency:
        lines.append(
            "latency  mean {mean:.2f} ms | median {median:.2f} | p95 {p95:.2f} "
            "| min {min:.2f} | max {max:.2f}".format(**latency)  # type: ignore[arg-type]
        )
    acquisition = stages.get("acquisition") or {}
    if acquisition:
        lines.append(
            f"  {'acquisition':<11} mean {acquisition['mean']:.2f} ms "
            "(decoder cost, excluded from latency)"
        )
    for stage in PIPELINE_STAGES:
        stats = stages.get(stage) or {}
        if stats:
            share = stats["mean"] / max(latency.get("mean", 1.0), 1e-9) * 100
            lines.append(f"  {stage:<11} mean {stats['mean']:.2f} ms ({share:.1f}%)")
    presentation = stages.get("presentation") or {}
    if presentation:
        lines.append(
            f"  {'presentation':<11} mean {presentation['mean']:.2f} ms "
            f"(debug only, excluded from throughput; wall-clock {float(summary.get('fps_wall') or 0.0):.2f} fps)"
        )
    if detections:
        lines.append(
            f"detections  {detections.get('total')} total, "
            f"{float(detections.get('per_frame_mean') or 0.0):.2f} per frame"
        )
        by_class = detections.get("by_class") or {}
        if by_class:
            lines.append(
                "  by class  "
                + ", ".join(f"{name}={count}" for name, count in by_class.items())
            )
    if events.get("engine_loaded"):
        startup = events["engine_loaded"] - events["start"]
        lines.append(f"startup  engine loaded after {startup:.2f} s")
    if summary.get("records"):
        lines.append(f"records  {summary['records']}")
    lines.append("=" * 62)
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolocar",
        description="YOLO road-object detection benchmark (TensorRT, single process).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=YoloCarConfig.model_path,
        help="TensorRT engine to deserialize",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=YoloCarConfig.source,
        help="video file, image, image directory, or camera index",
    )
    parser.add_argument("--conf", type=float, default=YoloCarConfig.conf_threshold)
    parser.add_argument("--iou", type=float, default=YoloCarConfig.iou_threshold)
    parser.add_argument("--imgsz", type=int, default=YoloCarConfig.image_size)
    parser.add_argument("--device", type=int, default=YoloCarConfig.device)
    parser.add_argument(
        "--classes",
        default=",".join(str(c) for c in ROAD_CLASS_IDS),
        help="comma separated COCO ids kept by the domain filter",
    )
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--max-det", type=int, default=YoloCarConfig.max_detections)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--loops",
        type=int,
        default=YoloCarConfig.loops,
        help="replay the source this many times (long runs from a short clip)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after this many seconds of wall time",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="write one JSON object per frame here (JSONL)",
    )
    parser.add_argument("--draw", action="store_true", help="annotate and show frames")
    parser.add_argument(
        "--output", type=Path, default=None, help="write annotated frames here"
    )
    parser.add_argument("--json", type=Path, default=None, help="write summary JSON here")
    parser.add_argument(
        "--rotate",
        type=int,
        default=YoloCarConfig.rotate,
        choices=(0, 90, 180, 270),
        help="clockwise rotation (degrees) applied to every frame at acquisition",
    )
    parser.add_argument("--no-pinned", action="store_true", help="disable page-locked host memory")
    parser.add_argument(
        "--trt-log-level",
        default=YoloCarConfig.trt_log_level,
        choices=("verbose", "info", "warning", "error"),
    )
    parser.add_argument("--log-level", default=YoloCarConfig.log_level)
    return parser


def config_from_args(args: argparse.Namespace) -> YoloCarConfig:
    class_ids = tuple(
        int(value) for value in str(args.classes).split(",") if value.strip() != ""
    )
    return YoloCarConfig(
        model_path=Path(args.model),
        device=int(args.device),
        image_size=int(args.imgsz),
        pinned_host_memory=not args.no_pinned,
        trt_log_level=args.trt_log_level,
        source=args.source,
        max_frames=args.max_frames,
        loops=int(args.loops),
        duration_s=args.duration,
        records_path=Path(args.records) if args.records else None,
        draw=bool(args.draw),
        output_path=Path(args.output) if args.output else None,
        json_path=Path(args.json) if args.json else None,
        rotate=int(args.rotate),
        conf_threshold=float(args.conf),
        iou_threshold=float(args.iou),
        agnostic_nms=bool(args.agnostic_nms),
        max_detections=int(args.max_det),
        target_class_ids=class_ids or ROAD_CLASS_IDS,
        log_level=args.log_level,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = config_from_args(args)
    except ValueError as exc:  # e.g. a class id outside the COCO range
        parser.error(str(exc))
    LOGGER.debug("config: %s", config)
    run_benchmark(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
