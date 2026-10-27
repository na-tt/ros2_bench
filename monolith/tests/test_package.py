"""Scaffolding smoke tests: the package must be importable and wired up."""

from __future__ import annotations

import yolocar
from yolocar.config import YoloCarConfig
from yolocar.main import build_parser
from yolocar.processor import YoloCarProcessor


def test_version() -> None:
    assert yolocar.__version__


def test_processor_constructs_with_default_config() -> None:
    processor = YoloCarProcessor()
    assert isinstance(processor.config, YoloCarConfig)


def test_cli_parser_accepts_source() -> None:
    args = build_parser().parse_args(["--source", "assets/videos"])
    assert str(args.source) == "assets/videos"
