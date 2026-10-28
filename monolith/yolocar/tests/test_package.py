"""Scaffolding smoke tests: the package must be importable and wired up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import yolocar
from yolocar.config import YoloCarConfig
from yolocar.main import build_parser, open_source
from yolocar.processor import YoloCarProcessor

cv2 = pytest.importorskip("cv2")


def _write_clip(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    """Write ``frames`` as a lossy-but-dependable MJPG/avi clip."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        pytest.skip("no OpenCV video writer backend for MJPG/avi")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _marker_frame() -> np.ndarray:
    """Black 120x160 frame with a white square in the top-left corner."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[0:20, 0:20] = 255
    return frame


def _brightness(frame: np.ndarray, y: int, x: int) -> float:
    return float(frame[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3].mean())


def test_version() -> None:
    assert yolocar.__version__


def test_processor_constructs_with_default_config() -> None:
    processor = YoloCarProcessor()
    assert isinstance(processor.config, YoloCarConfig)


def test_cli_parser_accepts_source() -> None:
    args = build_parser().parse_args(["--source", "assets/videos"])
    assert str(args.source) == "assets/videos"


def test_cli_parser_accepts_rotation() -> None:
    assert build_parser().parse_args([]).rotate == 0
    assert build_parser().parse_args(["--rotate", "180"]).rotate == 180


@pytest.mark.parametrize(
    ("rotate", "shape", "corner"),
    [
        (0, (120, 160), (10, 10)),  # marker stays top-left
        (90, (160, 120), (10, 110)),  # top-left -> top-right
        (180, (120, 160), (109, 149)),  # top-left -> bottom-right
        (270, (160, 120), (149, 10)),  # top-left -> bottom-left
    ],
)
def test_open_source_rotates_frames(
    tmp_path: Path, rotate: int, shape: tuple[int, int], corner: tuple[int, int]
) -> None:
    path = tmp_path / "clip.avi"
    _write_clip(path, [_marker_frame()])
    source = open_source(path, rotate=rotate)
    frame = next(iter(source))
    source.close()
    assert frame.shape[:2] == shape
    assert _brightness(frame, *corner) > 200.0


def test_open_source_swaps_size_for_quarter_turns(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    _write_clip(path, [_marker_frame()])
    assert open_source(path).size == (160, 120)
    assert open_source(path, rotate=90).size == (120, 160)
    assert open_source(path, rotate=180).size == (160, 120)


def test_open_source_rejects_unsupported_rotation() -> None:
    with pytest.raises(ValueError, match="unsupported rotation"):
        open_source("assets/videos/does-not-exist.mov", rotate=45)
