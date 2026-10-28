"""YOLO-based road-object detection pipeline (TensorRT).

Single source of truth for the yolocar pipeline logic. The ROS 2 wrapper in
``ws/src/yolocar_ros`` must only handle ROS 2 boilerplate and delegate all
inference / post-processing to :mod:`yolocar.processor`.

Modules
-------
``config``
    Frozen dataclass with every tunable plus the COCO class table.
``cuda``
    ctypes bindings to ``libcudart`` (device/host buffers, stream, copies).
``engine``
    TensorRT engine deserialization, buffers and ``infer``.
``processor``
    Letterbox preprocessing, NMS, domain filter, coordinate adaptation, drawing.
``main``
    Standalone single-process benchmark CLI.

Heavy submodules are imported lazily, so ``import yolocar`` (and the smoke
tests) work without OpenCV / TensorRT present.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "COCO_CLASS_NAMES",
    "ROAD_CLASS_IDS",
    "Detection",
    "LetterboxInfo",
    "PreprocessedFrame",
    "YoloCarConfig",
    "YoloCarProcessor",
    "class_name",
    "draw_detections",
    "nms",
    "process_frame",
]

_LAZY_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "COCO_CLASS_NAMES": ("yolocar.config", "COCO_CLASS_NAMES"),
    "ROAD_CLASS_IDS": ("yolocar.config", "ROAD_CLASS_IDS"),
    "YoloCarConfig": ("yolocar.config", "YoloCarConfig"),
    "class_name": ("yolocar.config", "class_name"),
    "Detection": ("yolocar.processor", "Detection"),
    "LetterboxInfo": ("yolocar.processor", "LetterboxInfo"),
    "PreprocessedFrame": ("yolocar.processor", "PreprocessedFrame"),
    "YoloCarProcessor": ("yolocar.processor", "YoloCarProcessor"),
    "draw_detections": ("yolocar.processor", "draw_detections"),
    "nms": ("yolocar.processor", "nms"),
    "process_frame": ("yolocar.processor", "process_frame"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_ATTRIBUTES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
