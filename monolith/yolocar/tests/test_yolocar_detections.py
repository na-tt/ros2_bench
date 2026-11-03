"""Tests for the detections sidecar (writer, reader, header validation).

No GPU, no engine, no OpenCV: the sidecar is stdlib-only on purpose, because
the ROS 2 postprocessing node writes it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolocar.detections import (
    DETECTIONS_VERSION,
    DetectionLog,
    FrameDetections,
    build_header,
    iter_records,
    load,
    read_header,
)
from yolocar.postprocessing import Detection

CAR = 2


def _detection(x1: float, y1: float, x2: float, y2: float, conf: float, cls: int = CAR) -> Detection:
    return Detection(
        x_min=x1, y_min=y1, x_max=x2, y_max=y2, confidence=conf, class_id=cls,
        class_name="car",
    )


def _header() -> dict[str, object]:
    return build_header(
        source="assets/video/x.mov",
        rotate=180,
        model="assets/models/yolov8n_fp16.engine",
        image_size=640,
        conf_threshold=0.25,
        iou_threshold=0.45,
        agnostic_nms=False,
        class_ids=(0, 2, 7),
        producer="test",
    )


# ---------------------------------------------------------------------- write
def test_disabled_log_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "detections.jsonl"
    with DetectionLog(None) as log:
        assert not log.enabled
        log.write(seq=0, loop=0, t_mono=1.0, detections=[_detection(1, 2, 3, 4, 0.5)])
    assert not target.exists()


def test_header_is_written_once_and_frames_follow(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "detections.jsonl"
    with DetectionLog(target, header=_header()) as log:
        log.write(seq=0, loop=0, t_mono=10.0, detections=[_detection(1.04, 2.06, 3.0, 4.9, 0.5)])
        log.write(seq=1, loop=0, t_mono=10.033, detections=[])

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    header = json.loads(lines[0])
    assert header["kind"] == "detections"
    assert header["v"] == DETECTIONS_VERSION
    assert header["rotate"] == 180
    assert header["classes"] == [0, 2, 7]

    first = json.loads(lines[1])
    assert first["seq"] == 0 and first["loop"] == 0
    # Coordinates to 0.1 px, scores to 0.001, class id as an int.
    assert first["det"] == [[1.0, 2.1, 3.0, 4.9, 0.5, CAR]]
    assert json.loads(lines[2])["det"] == []


def test_each_line_is_flushed_immediately(tmp_path: Path) -> None:
    """A killed run must leave a readable prefix, like records.jsonl."""
    target = tmp_path / "detections.jsonl"
    log = DetectionLog(target, header=_header()).__enter__()
    try:
        log.write(seq=0, loop=0, t_mono=1.0, detections=[])
        assert len(list(iter_records(target))) == 1
    finally:
        log.close()


def test_write_preserves_seq_loop_and_timestamp(tmp_path: Path) -> None:
    target = tmp_path / "detections.jsonl"
    with DetectionLog(target, header=_header()) as log:
        log.write(seq=17, loop=3, t_mono=1234.5678901, detections=[])
    record = next(iter_records(target))
    assert (record.seq, record.loop) == (17, 3)
    assert record.t_mono == pytest.approx(1234.567890, abs=1e-6)


# ----------------------------------------------------------------------- read
def test_round_trip_keeps_every_row(tmp_path: Path) -> None:
    target = tmp_path / "detections.jsonl"
    detections = [
        _detection(10, 20, 30, 40, 0.9, cls=0),
        _detection(50, 60, 70, 80, 0.1, cls=7),
    ]
    with DetectionLog(target, header=_header()) as log:
        log.write(seq=0, loop=0, t_mono=1.0, detections=detections)

    header, records = load(target)
    assert header["producer"] == "test"
    assert len(records) == 1
    assert records[0].rows == ((10.0, 20.0, 30.0, 40.0, 0.9, 0), (50.0, 60.0, 70.0, 80.0, 0.1, 7))
    assert len(records[0]) == 2
    assert isinstance(records[0], FrameDetections)


def test_reader_rejects_a_foreign_jsonl_file(tmp_path: Path) -> None:
    target = tmp_path / "records.jsonl"
    target.write_text('{"seq": 0, "loop": 0, "t_mono": 1.0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a detections log"):
        read_header(target)


def test_reader_rejects_an_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "detections.jsonl"
    target.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty detections log"):
        read_header(target)


def test_reader_rejects_a_newer_schema_version(tmp_path: Path) -> None:
    target = tmp_path / "detections.jsonl"
    header = {"kind": "detections", "v": DETECTIONS_VERSION + 1}
    target.write_text(json.dumps(header) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported detections log version"):
        read_header(target)


def test_reader_reports_the_line_of_a_malformed_entry(tmp_path: Path) -> None:
    target = tmp_path / "detections.jsonl"
    with DetectionLog(target, header=_header()) as log:
        log.write(seq=0, loop=0, t_mono=1.0, detections=[])
    with target.open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 1, "loop": 0, "t_mono": 2.0, "det": [[1,2]]}\n')
    with pytest.raises(ValueError, match=r":3: malformed detections entry"):
        list(iter_records(target))
