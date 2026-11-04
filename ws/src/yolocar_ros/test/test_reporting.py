"""Tests for the records/summary writer shared with the benchmark harness."""

import json
from pathlib import Path

import pytest

from yolocar.config import ROAD_CLASS_IDS, YoloCarConfig
from yolocar.detections import build_header, DetectionLog, iter_records
from yolocar.postprocessing import Detection

from yolocar_ros.reporting import (
    build_summary,
    detection_summary,
    latency_stats,
    RecordWriter,
    write_summary,
)


def _config(records: Path | None = None, summary: Path | None = None) -> YoloCarConfig:
    return YoloCarConfig(
        model_path=Path('assets/models/yolov8n_fp16.engine'),
        source='assets/video/clip.mov',
        records_path=records,
        json_path=summary,
        loops=2,
        duration_s=10.0,
    )


def test_latency_stats_of_empty_series_is_empty() -> None:
    assert latency_stats([]) == {}


def test_latency_stats_reports_the_contract_keys() -> None:
    stats = latency_stats([1.0, 2.0, 3.0, 4.0])
    assert set(stats) == {'mean', 'std', 'median', 'p95', 'min', 'max'}
    assert stats['mean'] == pytest.approx(2.5)
    assert stats['median'] == pytest.approx(2.5)


def test_detection_summary_counts_and_classes() -> None:
    summary = detection_summary(4, [1, 2, 3, 4], {'car': 8, 'person': 2})
    assert summary['total'] == 10
    assert summary['per_frame_mean'] == pytest.approx(2.5)
    assert summary['per_frame_max'] == 4
    assert list(summary['by_class']) == ['car', 'person']


def test_record_writer_flushes_each_line(tmp_path: Path) -> None:
    path = tmp_path / 'records.jsonl'
    writer = RecordWriter(path)
    writer.write({'seq': 0, 'n_det': 2})
    # The postprocessing node acks a frame only after its line is on disk, so
    # the data must be visible before close().
    assert json.loads(path.read_text(encoding='utf-8').splitlines()[0])['seq'] == 0
    writer.write({'seq': 1, 'n_det': 0})
    writer.close()

    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert [row['seq'] for row in rows] == [0, 1]


def test_record_writer_without_path_is_a_no_op() -> None:
    writer = RecordWriter(None)
    writer.write({'seq': 0})
    writer.close()


def test_detections_sidecar_stays_in_lockstep_with_records(tmp_path: Path) -> None:
    """Both files the postprocessing node writes must agree on seq."""
    records = RecordWriter(tmp_path / 'records.jsonl')
    detections = DetectionLog(
        tmp_path / 'detections.jsonl',
        header=build_header(
            source='assets/video/clip.mov',
            rotate=180,
            model='engine',
            image_size=640,
            conf_threshold=0.25,
            iou_threshold=0.45,
            agnostic_nms=False,
            class_ids=ROAD_CLASS_IDS,
            producer='ros-N4-p2c1',
        ),
    ).open_file()
    found = Detection(
        x_min=1.0, y_min=2.0, x_max=3.0, y_max=4.0,
        confidence=0.5, class_id=2, class_name='car',
    )
    try:
        for seq in range(5):
            records.write({'seq': seq, 'n_det': 1})
            detections.write(seq=seq, loop=0, t_mono=1.0 + seq, detections=[found])
    finally:
        records.close()
        detections.close()

    stored = [record.seq for record in iter_records(tmp_path / 'detections.jsonl')]
    logged = [
        json.loads(line)['seq']
        for line in (tmp_path / 'records.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    assert stored == logged == list(range(5))


def test_build_summary_records_the_detections_sidecar_path(tmp_path: Path) -> None:
    config = _config()
    common = {
        'config': config,
        'source': 'assets/video/clip.mov',
        'frames': 1,
        'replays': 1,
        'events': {'start': 1.0, 'end': 2.0},
        'interrupted': False,
        'wall_time_s': 1.0,
        'latency_ms': [10.0],
        'acquisition_ms': [8.0],
        'preprocess_ms': [3.0],
        'inference_ms': [6.0],
        'postprocess_ms': [1.0],
        'transport_ms': [0.0],
        'detections_per_frame': [1],
        'detections_by_class': {'car': 1},
    }
    without = build_summary(**common, records_path=None)  # type: ignore[arg-type]
    assert without['detections_log'] is None

    path = tmp_path / 'detections.jsonl'
    with_it = build_summary(  # type: ignore[arg-type]
        **common, records_path=None, detections_path=path
    )
    assert with_it['detections_log'] == str(path)


def test_build_summary_matches_the_monolith_contract(tmp_path: Path) -> None:
    config = _config()
    summary = build_summary(
        config=config,
        source='assets/video/clip.mov',
        frames=100,
        replays=2,
        events={'start': 1.0, 'engine_loaded': 2.0, 'first_frame': 2.0, 'end': 12.0},
        interrupted=False,
        wall_time_s=10.0,
        latency_ms=[10.0] * 100,
        acquisition_ms=[8.0] * 100,
        preprocess_ms=[3.0] * 100,
        inference_ms=[6.0] * 100,
        postprocess_ms=[1.0] * 100,
        transport_ms=[0.0] * 100,
        detections_per_frame=[3] * 100,
        detections_by_class={'car': 300},
        records_path=tmp_path / 'records.jsonl',
    )

    assert summary['pipeline'] == 'yolocar'
    assert summary['frontend'] == 'ros2'
    assert summary['fps'] == pytest.approx(10.0)
    assert summary['fps_wall'] == pytest.approx(10.0)
    assert summary['latency_ms']['median'] == pytest.approx(10.0)
    assert summary['detections']['total'] == 300
    assert summary['config']['target_class_ids'] == list(ROAD_CLASS_IDS)
    assert set(summary['stage_ms']) == {
        'acquisition',
        'preprocess',
        'inference',
        'postprocess',
        'presentation',
        'transport',
    }
    # The file contract wants these event names, whatever the front-end.
    assert set(summary['events']) == {
        'start',
        'engine_loaded',
        'first_frame',
        'end',
    }


def test_write_summary_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / 'nested' / 'summary.json'
    write_summary(path, {'pipeline': 'yolocar'})
    assert json.loads(path.read_text(encoding='utf-8'))['pipeline'] == 'yolocar'
