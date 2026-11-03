"""Detections sidecar: a per-frame log of what the pipeline finally found.

The pipeline's product is a detection list, so the only artifact that can show
what it did is the list itself. This module writes (and reads back) a small
JSONL file next to ``records.jsonl``:

.. code-block:: text

    {"kind":"detections","v":1,"source":"assets/video/x.mov","rotate":180,...}
    {"seq":0,"loop":0,"t_mono":12345.678,"det":[[96.4,412.9,180.2,512.0,0.87,2]]}
    {"seq":1,"loop":0,"t_mono":12345.706,"det":[]}

Rows are ``[x1, y1, x2, y2, score, class_id]`` of the *final* detections --
after NMS, after the domain filter, with the coordinates already adapted to the
original frame -- i.e. exactly what a presentation stage would draw. Nothing
about the pipeline's internals (letterbox geometry, raw candidates) leaves the
process: this is a sink, not an interface.

The header line is not decoration: it lets the renderer *validate* the source
and rotation instead of trusting whoever typed the command. Rendering a
sidecar against the wrong rotation draws boxes in the wrong orientation and
looks like a detector bug.

Deliberately stdlib-only (no OpenCV, no numpy, no ROS): the ROS 2
postprocessing node writes this file, and it must not gain a heavy import for
a debug artifact. The file is written line by line with an immediate flush, so
a killed run still leaves a complete prefix, like ``records.jsonl``.

This artifact is **not** part of the benchmark contract: a front-end that never
writes one is still compliant, and ``records.jsonl``/``summary.json`` are
untouched by its presence.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

#: Schema version of the sidecar. Readers reject anything newer.
DETECTIONS_VERSION = 1

#: Value of the header's ``kind``, so a stray JSONL file is not mistaken for it.
HEADER_KIND = "detections"

#: Pixel coordinates are rounded to a tenth of a pixel, scores to a thousandth.
_COORD_DIGITS = 1
_SCORE_DIGITS = 3

#: One detection row: ``(x1, y1, x2, y2, score, class_id)``.
DetectionRow = tuple[float, float, float, float, float, int]


class DetectionLike(Protocol):
    """What the writer needs from a detection (``Detection`` satisfies it)."""

    @property
    def xyxy(self) -> tuple[float, float, float, float]: ...

    @property
    def confidence(self) -> float: ...

    @property
    def class_id(self) -> int: ...


@dataclass(frozen=True)
class FrameDetections:
    """One frame's detections, as read back from the sidecar."""

    seq: int
    loop: int
    t_mono: float
    rows: tuple[DetectionRow, ...]

    def __len__(self) -> int:
        return len(self.rows)


# --------------------------------------------------------------------- header
def build_header(
    *,
    source: object,
    rotate: int,
    model: object,
    image_size: int,
    conf_threshold: float,
    iou_threshold: float,
    agnostic_nms: bool,
    class_ids: Sequence[int],
    producer: str,
) -> dict[str, Any]:
    """Assemble the header that makes an artifact self-describing."""
    return {
        "kind": HEADER_KIND,
        "v": DETECTIONS_VERSION,
        "source": str(source),
        "rotate": int(rotate),
        "model": str(model),
        "imgsz": int(image_size),
        "conf": float(conf_threshold),
        "iou": float(iou_threshold),
        "agnostic": bool(agnostic_nms),
        "classes": [int(value) for value in class_ids],
        "producer": str(producer),
    }


# ---------------------------------------------------------------------- write
def _round_row(detection: DetectionLike) -> list[float | int]:
    x1, y1, x2, y2 = (round(float(value), _COORD_DIGITS) for value in detection.xyxy)
    return [
        x1,
        y1,
        x2,
        y2,
        round(float(detection.confidence), _SCORE_DIGITS),
        int(detection.class_id),
    ]


class DetectionLog:
    """Per-frame detections -> JSONL; a no-op when no path was given."""

    def __init__(
        self, path: Path | None, header: Mapping[str, Any] | None = None
    ) -> None:
        self.path = path
        self.header = dict(header) if header is not None else None
        self._file: TextIO | None = None

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def open_file(self) -> DetectionLog:
        """Create the file and write the header; a no-op when there is no path."""
        if self.path is not None and self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", encoding="utf-8")
            if self.header is not None:
                self._write_line(self.header)
        return self

    def __enter__(self) -> DetectionLog:
        return self.open_file()

    def write(
        self,
        *,
        seq: int,
        loop: int,
        t_mono: float,
        detections: Sequence[DetectionLike],
    ) -> None:
        """Append one frame; cheap enough to leave on (~40 B per detection)."""
        if self._file is None:
            return
        self._write_line(
            {
                "seq": int(seq),
                "loop": int(loop),
                "t_mono": round(float(t_mono), 6),
                "det": [_round_row(detection) for detection in detections],
            }
        )

    def _write_line(self, payload: Mapping[str, Any]) -> None:
        assert self._file is not None
        self._file.write(json.dumps(dict(payload), separators=(",", ":")) + "\n")
        # Flush per line: a killed run leaves a complete prefix (same contract
        # as records.jsonl).
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ----------------------------------------------------------------------- read
def read_header(path: Path) -> dict[str, Any]:
    """First line of a sidecar, validated. Raises ``ValueError`` when unusable."""
    with Path(path).open("r", encoding="utf-8") as handle:
        first = handle.readline()
    if not first.strip():
        raise ValueError(f"{path}: empty detections log")
    try:
        header = json.loads(first)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: header line is not JSON ({exc})") from exc
    if not isinstance(header, dict) or header.get("kind") != HEADER_KIND:
        raise ValueError(f"{path}: not a detections log (missing kind={HEADER_KIND!r})")
    version = header.get("v")
    if not isinstance(version, int) or version > DETECTIONS_VERSION:
        raise ValueError(
            f"{path}: unsupported detections log version {version!r} "
            f"(this build reads v{DETECTIONS_VERSION})"
        )
    return header


def iter_records(path: Path) -> Iterator[FrameDetections]:
    """Yield the per-frame entries of a sidecar (the header is skipped)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        read_header(Path(path))  # validate before yielding anything
        handle.readline()  # drop the header line
        for number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                rows = tuple(
                    (
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        int(row[5]),
                    )
                    for row in payload["det"]
                )
                yield FrameDetections(
                    seq=int(payload["seq"]),
                    loop=int(payload["loop"]),
                    t_mono=float(payload["t_mono"]),
                    rows=rows,  # type: ignore[arg-type]
                )
            except (
                json.JSONDecodeError,
                IndexError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(f"{path}:{number}: malformed detections entry ({exc})") from exc


def load(path: Path) -> tuple[dict[str, Any], list[FrameDetections]]:
    """Header plus every frame, for callers that need to walk the log twice."""
    header = read_header(Path(path))
    return header, list(iter_records(Path(path)))
