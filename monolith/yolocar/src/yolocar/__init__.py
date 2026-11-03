"""YOLO-based road-object detection pipeline (TensorRT).

Single source of truth for the yolocar pipeline logic. The ROS 2 wrapper in
``ws/src/yolocar_ros`` must only handle ROS 2 boilerplate and delegate all
inference / post-processing to these modules.

Modules
-------
``config``
    Frozen dataclass with every tunable plus the COCO class table.
``cuda``
    ctypes bindings to ``libcudart`` (device/host buffers, stream, copies).
``engine``
    TensorRT engine deserialization, buffers and ``infer``.
``acquisition``
    Stage 1: video/image/camera iterators and the rotation correction.
``preprocessing``
    Stage 2: image conditioning at source resolution (coercion + op chain).
``model_input``
    The engine input contract: letterbox, colour/spatial layout, dtype.
``inference``
    Stage 3: engine invocation and detect-head decoding.
``postprocessing``
    Stages 4-6: NMS, domain filter, coordinate adaptation.
``detections``
    Per-frame sidecar log of the final detections (artefact, not a stage).
``render``
    Optional presentation: annotation, HUD and frame sinks.
``demo``
    Offline renderer: replays a run's detections sidecar onto the clip.
``processor``
    Facade that owns the engine and wires stages 2-6 together.
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
    "OPS",
    "ROAD_CLASS_IDS",
    "Candidates",
    "Detection",
    "DetectionLog",
    "FrameSource",
    "InputSpec",
    "LetterboxInfo",
    "Op",
    "YoloCarConfig",
    "YoloCarProcessor",
    "class_name",
    "condition",
    "coordinate_adaptation",
    "decode_predictions",
    "domain_filter",
    "draw_detections",
    "draw_hud",
    "enhance",
    "letterbox",
    "nms",
    "open_source",
    "postprocess",
    "prepare",
    "process_frame",
    "rotate_frame",
    "to_detections",
    "to_tensor",
]

_LAZY_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "COCO_CLASS_NAMES": ("yolocar.config", "COCO_CLASS_NAMES"),
    "ROAD_CLASS_IDS": ("yolocar.config", "ROAD_CLASS_IDS"),
    "YoloCarConfig": ("yolocar.config", "YoloCarConfig"),
    "class_name": ("yolocar.config", "class_name"),
    "FrameSource": ("yolocar.acquisition", "FrameSource"),
    "open_source": ("yolocar.acquisition", "open_source"),
    "rotate_frame": ("yolocar.acquisition", "rotate_frame"),
    "LetterboxInfo": ("yolocar.model_input", "LetterboxInfo"),
    "letterbox": ("yolocar.model_input", "letterbox"),
    "prepare": ("yolocar.model_input", "prepare"),
    "to_tensor": ("yolocar.model_input", "to_tensor"),
    "InputSpec": ("yolocar.model_input", "InputSpec"),
    "OPS": ("yolocar.preprocessing", "OPS"),
    "Op": ("yolocar.preprocessing", "Op"),
    "condition": ("yolocar.preprocessing", "condition"),
    "enhance": ("yolocar.preprocessing", "enhance"),
    "decode_predictions": ("yolocar.inference", "decode_predictions"),
    "Candidates": ("yolocar.postprocessing", "Candidates"),
    "Detection": ("yolocar.postprocessing", "Detection"),
    "coordinate_adaptation": ("yolocar.postprocessing", "coordinate_adaptation"),
    "domain_filter": ("yolocar.postprocessing", "domain_filter"),
    "nms": ("yolocar.postprocessing", "nms"),
    "postprocess": ("yolocar.postprocessing", "postprocess"),
    "to_detections": ("yolocar.postprocessing", "to_detections"),
    "DetectionLog": ("yolocar.detections", "DetectionLog"),
    "draw_hud": ("yolocar.render", "draw_hud"),
    "draw_detections": ("yolocar.render", "draw_detections"),
    "YoloCarProcessor": ("yolocar.processor", "YoloCarProcessor"),
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
