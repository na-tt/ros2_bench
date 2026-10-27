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

#: COCO ids that count as "road objects"/road users for this pipeline.
#: person, bicycle, car, motorcycle, bus, truck
ROAD_CLASS_IDS: tuple[int, ...] = (0, 1, 2, 3, 5, 7)

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
    #: Engine input size (square). Only used for warmup / dynamic shapes.
    image_size: int = 640
    #: Number of dummy frames pushed through before measuring.
    warmup_iterations: int = 10
    #: Use page-locked host memory for H2D/D2H copies (faster, preferred).
    pinned_host_memory: bool = True
    #: TensorRT logger verbosity: "warning", "info" or "verbose".
    trt_log_level: str = "warning"

    # --- source / output ------------------------------------------------
    #: File, directory of images, video, or camera index ("0").
    source: Path | str = Path("assets/videos")
    #: Cap the number of processed frames (``None`` = all).
    max_frames: int | None = None
    #: Show annotated frames in a window (needs ``$DISPLAY``).
    draw: bool = False
    #: Write annotated frames here (``.mp4`` for video sources, a directory for
    #: image sources).
    output_path: Path | None = None
    #: Dump the benchmark summary as JSON here.
    json_path: Path | None = None

    # --- post-processing ------------------------------------------------
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    #: Merge boxes across classes regardless of class id.
    agnostic_nms: bool = False
    #: Max detections kept after NMS.
    max_detections: int = 300
    #: Cap candidates before NMS (matches the ultralytics default).
    max_candidates: int = 30_000
    #: COCO ids kept by the domain filter (road objects by default).
    target_class_ids: tuple[int, ...] = ROAD_CLASS_IDS
    #: Letterbox padding value (ultralytics uses 114).
    pad_value: int = 114

    # --- logging --------------------------------------------------------
    log_level: str = "INFO"
