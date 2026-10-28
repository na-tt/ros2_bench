"""Standalone single-process entry point / benchmark for the yolocar pipeline.

Usage examples::

    python -m yolocar.main --source assets/videos/clip.mp4
    python -m yolocar.main --source 0 --draw
    python -m yolocar.main --source assets/images --json /tmp/yolocar.json
    yolocar --source assets/videos/clip.mp4 --output /tmp/annotated.mp4

This script is the "monolith" baseline: one process, no middleware, no image
serialisation. The ROS 2 node in ``ws/src/yolocar_ros`` reuses
:class:`yolocar.processor.YoloCarProcessor` and only adds transport.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np

from .config import ROAD_CLASS_IDS, YoloCarConfig
from .processor import Detection, YoloCarProcessor, draw_detections

LOGGER = logging.getLogger("yolocar")

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm"}
)
VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm"})
DEFAULT_OUTPUT_FPS = 30.0
PROGRESS_EVERY = 50


# ------------------------------------------------------------------- sources
@dataclass
class FrameSource:
    """Thin iterator wrapper so ``main`` can report fps/size and release."""

    kind: str
    frames: Iterator[np.ndarray]
    fps: float
    size: tuple[int, int] | None  # (width, height)
    description: str
    release: Callable[[], None] | None = None

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.frames

    def close(self) -> None:
        if self.release is not None:
            self.release()
            self.release = None


def _iter_video(capture: "cv2.VideoCapture") -> Iterator[np.ndarray]:
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            return
        yield frame


def _iter_images(paths: Sequence[Path]) -> Iterator[np.ndarray]:
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            LOGGER.warning("skipping unreadable image: %s", path)
            continue
        yield image


def _capture_size(capture: "cv2.VideoCapture") -> tuple[int, int] | None:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return (width, height) if width and height else None


def _capture_fps(capture: "cv2.VideoCapture") -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps and np.isfinite(fps) and fps > 0 else DEFAULT_OUTPUT_FPS


#: Rotation code per supported clockwise angle; ``0`` means "leave as is".
_ROTATE_CODES: dict[int, int | None] = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _normalize_rotation(rotate: int) -> int:
    """Validate a clockwise rotation angle and return it in ``[0, 360)``."""
    angle = int(rotate) % 360
    if angle not in _ROTATE_CODES:
        raise ValueError(
            f"unsupported rotation {rotate!r}; expected one of 0, 90, 180, 270"
        )
    return angle


def _rotate_frames(
    frames: Iterator[np.ndarray], rotate: int
) -> Iterator[np.ndarray]:
    code = _ROTATE_CODES[rotate]
    for frame in frames:
        yield frame if code is None else cv2.rotate(frame, code)


def _rotated(source: FrameSource, rotate: int) -> FrameSource:
    """Wrap ``source`` so that every frame is rotated clockwise by ``rotate``."""
    if rotate == 0:
        return source
    size = source.size
    if size is not None and rotate in (90, 270):
        size = (size[1], size[0])
    return FrameSource(
        kind=source.kind,
        frames=_rotate_frames(source.frames, rotate),
        fps=source.fps,
        size=size,
        description=source.description,
        release=source.release,
    )


def open_source(source: Path | str, rotate: int = 0) -> FrameSource:
    """Open a video file, an image (directory), or a camera index.

    ``rotate`` (clockwise degrees: 0, 90, 180 or 270) is applied to every frame
    as it is read, i.e. inside the acquisition stage. Use it for clips whose
    container rotation flag the decoder ignores -- OpenCV exposes it as
    ``CAP_PROP_ORIENTATION_META`` but does not apply it for every backend, so
    frames can arrive upside down (which also degrades detection quality).
    """
    angle = _normalize_rotation(rotate)
    return _rotated(_open_raw_source(source), angle)


def _open_raw_source(source: Path | str) -> FrameSource:
    """Open the source without applying any rotation."""
    text = str(source)
    path = Path(text).expanduser()

    if path.is_dir():
        paths = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"no images found in {path}")
        return FrameSource(
            kind="images",
            frames=_iter_images(paths),
            fps=DEFAULT_OUTPUT_FPS,
            size=None,
            description=f"{path} ({len(paths)} images)",
        )

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            return FrameSource(
                kind="images",
                frames=_iter_images([path]),
                fps=DEFAULT_OUTPUT_FPS,
                size=None,
                description=str(path),
            )
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        return FrameSource(
            kind="video",
            frames=_iter_video(capture),
            fps=_capture_fps(capture),
            size=_capture_size(capture),
            description=str(path),
            release=capture.release,
        )

    try:
        index = int(text)
    except ValueError:
        raise FileNotFoundError(f"source not found: {source}") from None

    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera index {index}")
    return FrameSource(
        kind="camera",
        frames=_iter_video(capture),
        fps=_capture_fps(capture),
        size=_capture_size(capture),
        description=f"camera:{index}",
        release=capture.release,
    )


# ------------------------------------------------------------------- writers
class _VideoWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer for {path}")
        self.path = path

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


class _ImageWriter:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.count = 0

    def write(self, frame: np.ndarray) -> None:
        target = self.directory / f"frame_{self.count:06d}.jpg"
        cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        self.count += 1

    def close(self) -> None:
        return None


def _open_writer(config: YoloCarConfig, source: FrameSource):
    if config.output_path is None:
        return None
    target = Path(config.output_path)
    if source.kind == "images" and target.suffix.lower() not in VIDEO_SUFFIXES:
        return _ImageWriter(target)
    size = source.size
    if size is None:
        LOGGER.warning("frame size unknown; pass a video/camera source to write a video")
        return _ImageWriter(target)
    return _VideoWriter(target, source.fps or DEFAULT_OUTPUT_FPS, size)


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
    source = open_source(config.source, config.rotate)
    LOGGER.info(
        "source: %s (kind=%s, fps=%.1f, rotate=%d)",
        source.description,
        source.kind,
        source.fps,
        config.rotate,
    )

    processor = YoloCarProcessor(config)
    writer = _open_writer(config, source)
    show = bool(config.draw)

    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    frame_ms: list[float] = []
    detections_per_frame: list[int] = []
    detections_by_class: dict[str, int] = {}
    frames = 0
    interrupted = False

    try:
        processor.load()
        processor.warmup()

        start = time.perf_counter()
        for frame in source:
            if config.max_frames is not None and frames >= config.max_frames:
                break

            t0 = time.perf_counter()
            preprocessed = processor.preprocess(frame)
            t1 = time.perf_counter()
            outputs = processor.infer(preprocessed)
            t2 = time.perf_counter()
            detections: list[Detection] = processor.postprocess(outputs, preprocessed)
            t3 = time.perf_counter()

            preprocess_ms.append((t1 - t0) * 1e3)
            inference_ms.append((t2 - t1) * 1e3)
            postprocess_ms.append((t3 - t2) * 1e3)
            frame_ms.append((t3 - t0) * 1e3)

            detections_per_frame.append(len(detections))
            for detection in detections:
                detections_by_class[detection.class_name] = (
                    detections_by_class.get(detection.class_name, 0) + 1
                )
            frames += 1

            if writer is not None or show:
                annotated = (
                    frame
                    if not detections
                    else draw_detections(frame, detections)
                )
                if writer is not None:
                    writer.write(annotated)
                if show:
                    try:
                        cv2.imshow("yolocar", annotated)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            LOGGER.info("display: quitting on user request")
                            break
                    except cv2.error as exc:  # headless environment
                        LOGGER.warning("cannot show frames (%s); disabling display", exc)
                        show = False

            if frames % PROGRESS_EVERY == 0:
                LOGGER.info(
                    "frame %d: %.1f ms (pre %.1f / infer %.1f / post %.1f), %d dets",
                    frames,
                    frame_ms[-1],
                    preprocess_ms[-1],
                    inference_ms[-1],
                    postprocess_ms[-1],
                    len(detections),
                )
        wall_time = time.perf_counter() - start
    except KeyboardInterrupt:
        LOGGER.warning("interrupted by user; reporting partial results")
        interrupted = True
        wall_time = float("nan")
    finally:
        source.close()
        if writer is not None:
            writer.close()
        processor.close()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    summary: dict[str, object] = {
        "pipeline": "yolocar",
        "model": str(config.model_path),
        "source": source.description,
        "frames": frames,
        "warmup_iterations": config.warmup_iterations,
        "interrupted": interrupted,
        "wall_time_s": wall_time,
        "fps": (frames / wall_time) if frames and wall_time > 0 else 0.0,
        "latency_ms": _latency_stats(frame_ms),
        "stage_ms": {
            "preprocess": _latency_stats(preprocess_ms),
            "inference": _latency_stats(inference_ms),
            "postprocess": _latency_stats(postprocess_ms),
        },
        "detections": _detection_summary(frames, detections_per_frame, detections_by_class),
        "config": {
            "conf_threshold": config.conf_threshold,
            "iou_threshold": config.iou_threshold,
            "image_size": config.image_size,
            "target_class_ids": list(config.target_class_ids),
            "agnostic_nms": config.agnostic_nms,
            "device": config.device,
            "rotate": config.rotate,
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
    lines = [
        "=" * 62,
        f"yolocar  model={summary.get('model')}",
        f"source   {summary.get('source')}",
        f"frames   {summary.get('frames')} (warmup {summary.get('warmup_iterations')})",
        f"throughput {float(summary.get('fps') or 0.0):.2f} fps",
    ]
    if latency:
        lines.append(
            "latency  mean {mean:.2f} ms | median {median:.2f} | p95 {p95:.2f} "
            "| min {min:.2f} | max {max:.2f}".format(**latency)  # type: ignore[arg-type]
        )
    for stage in ("preprocess", "inference", "postprocess"):
        stats = stages.get(stage) or {}
        if stats:
            lines.append(
                f"  {stage:<11} mean {stats['mean']:.2f} ms ({stats['mean'] / max(latency.get('mean', 1.0), 1e-9) * 100:.1f}%)"
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
        "--warmup",
        type=int,
        default=YoloCarConfig.warmup_iterations,
        help="dummy inferences before measuring",
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
        warmup_iterations=int(args.warmup),
        pinned_host_memory=not args.no_pinned,
        trt_log_level=args.trt_log_level,
        source=args.source,
        max_frames=args.max_frames,
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
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = config_from_args(args)
    LOGGER.debug("config: %s", config)
    run_benchmark(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
