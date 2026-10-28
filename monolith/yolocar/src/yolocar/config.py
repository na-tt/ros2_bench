"""Pipeline configuration and static tables (no logic, no side effects)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Ultralytics YOLOv8n (COCO 80) class names, index == class id.
COCO_CLASS_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

#: COCO ids kept by the domain filter: road users + road infrastructure.
#: person, bicycle, car, motorcycle, bus, train, truck, traffic light,
#: stop sign, parking meter
ROAD_CLASS_IDS: tuple[int, ...] = (0, 1, 2, 3, 5, 6, 7, 9, 11, 12)

#: Suffixes treated as still images by the acquisition stage / image sinks.
IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm"}
)
#: Suffixes treated as video containers by the acquisition stage / video sinks.
VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm"}
)
#: Fallback rate for sources that do not report a usable fps (image sequences).
DEFAULT_OUTPUT_FPS = 30.0

#: Default input name of an Ultralytics exported engine.
DEFAULT_INPUT_NAME = "images"
#: Default output name of an Ultralytics exported engine (pre-NMS head).
DEFAULT_OUTPUT_NAME = "output0"


def class_name(class_id: int) -> str:
    """Human readable COCO name, with a safe fallback for unknown ids."""
    if 0 <= class_id < len(COCO_CLASS_NAMES):
        return COCO_CLASS_NAMES[class_id]
    return f"class_{class_id}"


@dataclass(frozen=True)
class YoloCarConfig:
    """Parameters shared by the standalone script and the ROS 2 node.

    Keep every tunable here so both front-ends stay in sync.
    """

    # --- model / device -------------------------------------------------
    #: TensorRT engine (built for this device) or ``.onnx``/``.pt`` source and
    #: converted externally; see ``README.md``.
    model_path: Path = Path("assets/models/yolov8n_fp16.engine")
    #: CUDA device ordinal used by TensorRT.
    device: int = 0
    #: Engine input size (square). Only used for dynamic shapes.
    image_size: int = 640
    #: Use page-locked host memory for H2D/D2H copies (faster, preferred).
    pinned_host_memory: bool = True
    #: TensorRT logger verbosity: "warning", "info" or "verbose".
    trt_log_level: str = "warning"

    # --- source / output ------------------------------------------------
    #: File, directory of images, video, or camera index ("0").
    source: Path | str = Path("assets/videos")
    #: Cap the number of processed frames (``None`` = all).
    max_frames: int | None = None
    #: Replay the source this many times. Lets a short clip drive a long run
    #: without re-encoding video; every frame carries its ``loop`` index so
    #: drift between late and early replays stays visible.
    loops: int = 1
    #: Stop after this many seconds of wall time (``None`` = no limit). Useful
    #: for long telemetry runs whose length should not depend on clip size.
    duration_s: float | None = None
    #: Write one JSON object per frame here (JSONL). The analysis decides the
    #: steady-state window from this stream; the run itself discards nothing.
    records_path: Path | None = None
    #: Show annotated frames in a window (needs ``$DISPLAY``).
    draw: bool = False
    #: Write annotated frames here (``.mp4`` for video sources, a directory for
    #: image sources).
    output_path: Path | None = None
    #: Dump the benchmark summary as JSON here.
    json_path: Path | None = None
    #: Clockwise rotation (0/90/180/270) applied to every frame at acquisition.
    #: Needed for clips whose container rotation flag the decoder ignores.
    rotate: int = 0

    # --- post-processing ------------------------------------------------
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    #: Merge boxes across classes regardless of class id.
    agnostic_nms: bool = False
    #: Max detections kept after NMS.
    max_detections: int = 300
    #: Cap candidates before NMS (matches the ultralytics default).
    max_candidates: int = 30_000
    #: COCO ids kept by the domain filter (road users + road infrastructure
    #: by default; see ``ROAD_CLASS_IDS``).
    target_class_ids: tuple[int, ...] = ROAD_CLASS_IDS
    #: Letterbox padding value (ultralytics uses 114).
    pad_value: int = 114

    # --- logging --------------------------------------------------------
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Reject configurations that no downstream stage could honour.

        Validation lives here (not in the processor) so every front-end and
        every test constructing a config gets the same early failure, and so
        the stages can trust ``target_class_ids`` without re-checking it.
        """
        if not self.target_class_ids:
            raise ValueError("target_class_ids must contain at least one class id")
        lowest = min(self.target_class_ids)
        highest = max(self.target_class_ids)
        if lowest < 0 or highest >= len(COCO_CLASS_NAMES):
            raise ValueError(
                f"target_class_ids must be in [0, {len(COCO_CLASS_NAMES) - 1}]"
            )
        if int(self.loops) < 1:
            raise ValueError("loops must be >= 1")
        if self.duration_s is not None and float(self.duration_s) <= 0.0:
            raise ValueError("duration_s must be > 0 when set")
