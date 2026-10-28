"""
Message <-> numpy conversions for the four-node chain.

All pipeline logic stays in :mod:`yolocar`; this module only moves the values
between the stage functions and the wire types in ``yolocar_interfaces``. Raw
tensor payloads are host-order (little-endian) bytes, exactly what
``numpy.ndarray.tobytes()`` produces on this board.
"""

from __future__ import annotations

import numpy as np

from yolocar.postprocessing import Candidates
from yolocar.preprocessing import LetterboxInfo

from yolocar_interfaces.msg import CandidatesFrame, FrameMeta, ImageFrame, TensorFrame

BGR8 = 'bgr8'


# ------------------------------------------------------------------- metadata
def frame_meta(
    *,
    seq: int,
    loop: int,
    t_mono: float,
    acq_ms: float = 0.0,
    pre_ms: float = 0.0,
    infer_ms: float = 0.0,
) -> FrameMeta:
    """Build the per-frame identity/timing block carried through the graph."""
    meta = FrameMeta()
    meta.seq = int(seq)
    meta.loop = int(loop)
    meta.t_mono = float(t_mono)
    meta.acq_ms = float(acq_ms)
    meta.pre_ms = float(pre_ms)
    meta.infer_ms = float(infer_ms)
    return meta


def letterbox_from_message(msg: TensorFrame | CandidatesFrame) -> LetterboxInfo:
    """Rebuild the letterbox geometry stage 6 needs from its wire form."""
    return LetterboxInfo(
        scale=float(msg.scale),
        pad_x=int(msg.pad_x),
        pad_y=int(msg.pad_y),
        original_shape=(int(msg.original_height), int(msg.original_width)),
        input_shape=(int(msg.height), int(msg.width)),
    )


# ---------------------------------------------------------------------- image
def image_frame(frame: np.ndarray, meta: FrameMeta) -> ImageFrame:
    """BGR HxWx3 uint8 -> ``ImageFrame`` (pixels and metadata in one message)."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'expected an HxWx3 BGR frame, got {image.shape}')
    if image.dtype != np.uint8:
        raise ValueError(f'expected uint8 pixels, got {image.dtype}')

    msg = ImageFrame()
    msg.meta = meta
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.channels = int(image.shape[2])
    msg.step = int(image.shape[1] * image.shape[2])
    msg.encoding = BGR8
    msg.data = np.ascontiguousarray(image).tobytes()
    return msg


def frame_from_image(msg: ImageFrame) -> np.ndarray:
    """
    ``ImageFrame`` -> read-only HxWx3 BGR view.

    No copy: the stage functions only read the frame (``letterbox`` resizes
    into a new buffer), so the serialised payload is used in place.
    """
    if msg.encoding != BGR8:
        raise ValueError(f'unsupported frame encoding {msg.encoding!r}')
    if msg.channels != 3:
        raise ValueError(f'expected 3 channels, got {msg.channels}')
    if int(msg.step) != int(msg.width) * int(msg.channels):
        raise ValueError(
            f'image step {msg.step} does not match {msg.width}x{msg.channels}'
        )
    expected = int(msg.height) * int(msg.width) * int(msg.channels)
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size != expected:
        raise ValueError(
            f'image payload is {data.size} bytes, expected {expected} '
            f'({msg.width}x{msg.height}x{msg.channels})'
        )
    return data.reshape(int(msg.height), int(msg.width), int(msg.channels))


# --------------------------------------------------------------------- tensor
def tensor_frame(
    tensor: np.ndarray, meta: FrameMeta, letterbox: LetterboxInfo
) -> TensorFrame:
    """Engine-ready NCHW tensor + letterbox geometry -> ``TensorFrame``."""
    array = np.ascontiguousarray(tensor)
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 3:
        raise ValueError(f'expected a 1x3xHxW tensor, got {array.shape}')

    msg = TensorFrame()
    msg.meta = meta
    msg.height = int(array.shape[2])
    msg.width = int(array.shape[3])
    msg.dtype = array.dtype.name
    msg.scale = float(letterbox.scale)
    msg.pad_x = int(letterbox.pad_x)
    msg.pad_y = int(letterbox.pad_y)
    msg.original_height = int(letterbox.original_shape[0])
    msg.original_width = int(letterbox.original_shape[1])
    msg.data = array.tobytes()
    return msg


def tensor_from_frame(msg: TensorFrame) -> np.ndarray:
    """``TensorFrame`` -> read-only 1x3xHxW view of the engine input tensor."""
    dtype = np.dtype(msg.dtype)
    expected = 3 * int(msg.height) * int(msg.width)
    data = np.frombuffer(msg.data, dtype=dtype)
    if data.size != expected:
        raise ValueError(
            f'tensor payload is {data.size} {dtype} elements, expected {expected}'
        )
    return data.reshape(1, 3, int(msg.height), int(msg.width))


# ----------------------------------------------------------------- candidates
def candidates_frame(
    candidates: Candidates,
    meta: FrameMeta,
    letterbox: LetterboxInfo,
    engine_loaded_t: float = 0.0,
) -> CandidatesFrame:
    """``yolocar.postprocessing.Candidates`` -> ``CandidatesFrame``."""
    msg = CandidatesFrame()
    msg.meta = meta
    msg.height = int(letterbox.input_shape[0])
    msg.width = int(letterbox.input_shape[1])
    msg.scale = float(letterbox.scale)
    msg.pad_x = int(letterbox.pad_x)
    msg.pad_y = int(letterbox.pad_y)
    msg.original_height = int(letterbox.original_shape[0])
    msg.original_width = int(letterbox.original_shape[1])
    msg.boxes = np.ascontiguousarray(candidates.boxes, dtype=np.float32).reshape(-1)
    msg.scores = np.ascontiguousarray(candidates.scores, dtype=np.float32).reshape(-1)
    msg.class_ids = np.ascontiguousarray(candidates.class_ids, dtype=np.uint16).reshape(-1)
    msg.engine_loaded_t = float(engine_loaded_t)
    return msg


def candidates_from_frame(
    msg: CandidatesFrame,
) -> tuple[Candidates, LetterboxInfo]:
    """``CandidatesFrame`` -> ``(Candidates, LetterboxInfo)`` for stages 4-6."""
    boxes = np.frombuffer(msg.boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.frombuffer(msg.scores, dtype=np.float32)
    class_ids = np.frombuffer(msg.class_ids, dtype=np.uint16).astype(np.int32)
    if boxes.shape[0] != scores.shape[0] or boxes.shape[0] != class_ids.shape[0]:
        raise ValueError(
            'candidate arrays disagree: '
            f'{boxes.shape[0]} boxes, {scores.shape[0]} scores, '
            f'{class_ids.shape[0]} class ids'
        )
    candidates = Candidates(
        boxes=np.array(boxes, dtype=np.float32, copy=True),
        scores=np.array(scores, dtype=np.float32, copy=True),
        class_ids=class_ids,
    )
    return candidates, letterbox_from_message(msg)
