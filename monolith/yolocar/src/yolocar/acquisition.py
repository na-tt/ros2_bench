"""Stage 1: frame acquisition -- files, image directories, cameras.

Produces a :class:`FrameSource`: an iterator of :class:`Frame` values plus the
metadata a writer needs (fps, frame size, kind) and a ``close()`` that releases
the capture. A ROS 2 acquisition node can reuse this module to publish a video
file as a topic; only the transport differs.

Each :class:`Frame` carries a monotonic ``seq`` (across replays) and a ``loop``
index, so a long run built by replaying a short clip stays analysable: latency
drift between loop 1 and loop 20 is exactly the steady-state signal the
benchmark wants, and it is invisible if replays are silently stitched together.

**Rotation lives here on purpose.** ``--rotate`` corrects clips whose container
rotation flag the decoder ignores (OpenCV exposes it as
``CAP_PROP_ORIENTATION_META`` but does not apply it on every backend). That is a
defect of the decoder, not of the model's input contract, so it is applied to
the frame as it is read and *before* anything is timed as preprocessing or
inference. Consequence worth remembering: detections are expressed in the
**rotated** frame's pixel coordinates, which is also the coordinate space of
every annotated frame written out.

The one-frame transform is exported as :func:`rotate_frame` so a node that
receives frames from somewhere else (a camera driver, ``cv_bridge``) can apply
the same correction without going through a :class:`FrameSource`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np

from .config import DEFAULT_OUTPUT_FPS, IMAGE_SUFFIXES

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------- frame
@dataclass(frozen=True)
class Frame:
    """One acquired frame plus the identity a benchmark record needs."""

    image: np.ndarray
    seq: int  # monotonic across replays
    loop: int  # 0-based replay index


# ---------------------------------------------------------------------- source
@dataclass
class FrameSource:
    """Thin iterator wrapper so callers can report fps/size and release."""

    kind: str  # "video" | "images" | "camera"
    frames: Iterator[Frame]
    fps: float
    size: tuple[int, int] | None  # (width, height)
    description: str
    release: Callable[[], None] | None = None

    def __iter__(self) -> Iterator[Frame]:
        return self.frames

    def close(self) -> None:
        if self.release is not None:
            self.release()
            self.release = None


def _validate_loops(loops: int) -> int:
    count = int(loops)
    if count < 1:
        raise ValueError(f"loops must be >= 1, got {loops!r}")
    return count


def iter_video(capture: "cv2.VideoCapture", loops: int = 1) -> Iterator[Frame]:
    """Yield decoded frames, rewinding the file ``loops`` times.

    Rewinding seeks back to the first keyframe instead of reopening the
    container, so a replay does not repay decoder initialisation. If the
    backend refuses to seek, the replay stops rather than silently yielding
    nothing for the remaining loops.
    """
    count = _validate_loops(loops)
    seq = 0
    for loop in range(count):
        if loop and not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            LOGGER.warning("source cannot rewind; stopping after %d loops", loop)
            return
        produced = 0
        while True:
            ok, image = capture.read()
            if not ok or image is None:
                break
            yield Frame(image=image, seq=seq, loop=loop)
            seq += 1
            produced += 1
        if loop and produced == 0:
            LOGGER.warning("replay %d produced no frames; stopping", loop)
            return


def iter_images(paths: Sequence[Path], loops: int = 1) -> Iterator[Frame]:
    """Yield the readable images of ``paths``, replaying the list ``loops`` times."""
    count = _validate_loops(loops)
    seq = 0
    for loop in range(count):
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                LOGGER.warning("skipping unreadable image: %s", path)
                continue
            yield Frame(image=image, seq=seq, loop=loop)
            seq += 1


def capture_size(capture: "cv2.VideoCapture") -> tuple[int, int] | None:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return (width, height) if width and height else None


def capture_fps(capture: "cv2.VideoCapture") -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps and np.isfinite(fps) and fps > 0 else DEFAULT_OUTPUT_FPS


# -------------------------------------------------------------------- rotation
#: Rotation code per supported clockwise angle; ``0`` means "leave as is".
ROTATE_CODES: dict[int, int | None] = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def normalize_rotation(rotate: int) -> int:
    """Validate a clockwise rotation angle and return it in ``[0, 360)``."""
    angle = int(rotate) % 360
    if angle not in ROTATE_CODES:
        raise ValueError(
            f"unsupported rotation {rotate!r}; expected one of 0, 90, 180, 270"
        )
    return angle


def rotate_frame(image: np.ndarray, rotate: int) -> np.ndarray:
    """Rotate a single frame clockwise by ``rotate`` degrees (0/90/180/270)."""
    code = ROTATE_CODES[normalize_rotation(rotate)]
    return image if code is None else cv2.rotate(image, code)


def iter_rotated(frames: Iterator[Frame], rotate: int) -> Iterator[Frame]:
    code = ROTATE_CODES[normalize_rotation(rotate)]
    for frame in frames:
        if code is None:
            yield frame
        else:
            yield Frame(
                image=cv2.rotate(frame.image, code), seq=frame.seq, loop=frame.loop
            )


def rotated(source: FrameSource, rotate: int) -> FrameSource:
    """Wrap ``source`` so that every frame is rotated clockwise by ``rotate``."""
    angle = normalize_rotation(rotate)
    if angle == 0:
        return source
    size = source.size
    if size is not None and angle in (90, 270):
        size = (size[1], size[0])
    return FrameSource(
        kind=source.kind,
        frames=iter_rotated(source.frames, angle),
        fps=source.fps,
        size=size,
        description=source.description,
        release=source.release,
    )


# --------------------------------------------------------------------- opening
def open_source(source: Path | str, rotate: int = 0, loops: int = 1) -> FrameSource:
    """Open a video file, an image (directory), or a camera index.

    ``loops`` replays the source that many times (cameras are already open
    ended and must not be looped). ``rotate`` (clockwise degrees: 0, 90, 180 or
    270) is applied to every frame as it is read, i.e. inside the acquisition
    stage. See the module docstring for why the correction belongs here.
    """
    angle = normalize_rotation(rotate)
    return rotated(open_raw_source(source, loops=loops), angle)


def open_raw_source(source: Path | str, loops: int = 1) -> FrameSource:
    """Open the source without applying any rotation."""
    count = _validate_loops(loops)
    text = str(source)
    path = Path(text).expanduser()

    if path.is_dir():
        paths = sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"no images found in {path}")
        return FrameSource(
            kind="images",
            frames=iter_images(paths, count),
            fps=DEFAULT_OUTPUT_FPS,
            size=None,
            description=f"{path} ({len(paths)} images)",
        )

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            return FrameSource(
                kind="images",
                frames=iter_images([path], count),
                fps=DEFAULT_OUTPUT_FPS,
                size=None,
                description=str(path),
            )
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        return FrameSource(
            kind="video",
            frames=iter_video(capture, count),
            fps=capture_fps(capture),
            size=capture_size(capture),
            description=str(path),
            release=capture.release,
        )

    try:
        index = int(text)
    except ValueError:
        raise FileNotFoundError(f"source not found: {source}") from None

    if count > 1:
        raise ValueError("cannot loop a camera source; it is already open ended")
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera index {index}")
    return FrameSource(
        kind="camera",
        frames=iter_video(capture, count),
        fps=capture_fps(capture),
        size=capture_size(capture),
        description=f"camera:{index}",
        release=capture.release,
    )
