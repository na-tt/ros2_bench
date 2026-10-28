# yolocar

YOLO-based road-object detection (TensorRT) — the single-process baseline
benchmarked against the ROS 2 wrapper in `../../ws/src/yolocar_ros`.

This directory is a self-contained Python project: it has its own
`pyproject.toml`, an importable package under `src/yolocar/`, and its own
`tests/` suite. `monolith/` above it is just a container directory of such
projects.

## Layout

```
monolith/yolocar/
├── pyproject.toml       # name = "yolocar", src layout
├── README.md
├── src/yolocar/
│   ├── __init__.py      # lazy re-exports
│   ├── config.py        # tunables + COCO class table (no logic)
│   ├── cuda.py          # ctypes -> libcudart (buffers, stream, copies)
│   ├── engine.py        # TensorRT deserialize, buffers, infer
│   ├── processor.py     # letterbox, NMS, domain filter, coord adaptation, draw
│   ├── main.py          # standalone CLI / benchmark
│   └── py.typed
└── tests/
    ├── test_package.py           # import/wiring smoke tests
    ├── test_yolocar_pipeline.py  # letterbox / NMS / postprocess unit tests
    ├── test_yolocar_cuda.py      # CUDA bindings with fakes
    ├── test_yolocar_engine.py    # TensorRT glue with fakes
    └── test_yolocar_hardware.py  # real driver/engine, skips when unavailable
```

## Install

```bash
pip install -e ./monolith/yolocar            # numpy / opencv included
pip install -e "./monolith/yolocar[dev]"     # + pytest
```

`yolocar` needs at runtime:

* `numpy` and `opencv-python` (declared as regular dependencies),
* the TensorRT Python bindings (`import tensorrt`) — shipped with JetPack,
  imported lazily so the rest of the package works without it,
* the CUDA runtime (`libcudart`), reached through `ctypes` — **no** `pycuda`
  and **no** `cuda-python` required.

Only needed to re-export the ONNX model: `pip install -e "./monolith/yolocar[export]"`.

## Benchmark assets

Model weights and footage are shared by every pipeline in this repository and
live in the repository-root `assets/` directory, not inside this project. All
defaults (`YoloCarConfig.model_path`, `YoloCarConfig.source`) are relative
paths, resolved against the current working directory — run the commands below
from the repository root.

## Model

The default is a prebuilt FP16 engine for this device:

```
assets/models/yolov8n.pt        # ultralytics weights (reference)
assets/models/yolov8n.onnx      # export without NMS: output0 = (1, 84, 8400)
assets/models/yolov8n_fp16.engine
```

Rebuild the engine after a TensorRT / driver upgrade, or on another device
(engines are not portable across GPU architectures or TRT versions):

```bash
trtexec --onnx=assets/models/yolov8n.onnx \
        --saveEngine=assets/models/yolov8n_fp16.engine \
        --fp16 --memPoolSize=workspace:1GiB
```

## Run

```bash
python3 -m yolocar.main --help

# monolith baseline over a dashcam clip
python3 -m yolocar.main --source assets/videos/clip.mp4

# camera, annotated window, and a JSON summary for the comparison table
python3 -m yolocar.main --source 0 --draw --json results/yolocar.json

# write an annotated video
python3 -m yolocar.main --source assets/videos/clip.mp4 --output /tmp/out.mp4
```

Useful flags: `--classes 2,3,5,7` (domain filter), `--conf`, `--iou`,
`--max-frames`, `--warmup`, `--no-pinned`, `--trt-log-level`, `--rotate`.

`--rotate {0,90,180,270}` rotates every frame clockwise at acquisition. It is
needed for clips whose container rotation flag the decoder ignores: OpenCV
exposes it as `CAP_PROP_ORIENTATION_META` but does not apply it for every
backend, so frames can arrive upside down. Feeding the detector upside-down
frames measurably hurts recall, so rotate them upright before comparing
numbers. Both front-ends share the setting through `YoloCarConfig.rotate`, so
the ROS 2 node can apply the same correction.

The summary reports throughput plus a per-stage latency breakdown
(`preprocess / inference / postprocess`) so the ROS 2 overhead can be compared
against the same pipeline pieces.

## Tests

```bash
pytest monolith/yolocar              # from the repository root
cd monolith/yolocar && pytest        # or from the project root
```

The suite has two tiers. The unit tests fake the CUDA/TensorRT boundary and
always run; `test_yolocar_hardware.py` drives the real driver and engine and
skips itself when no device, bindings or engine is available. It looks for
`assets/models/yolov8n_fp16.engine` in the repository root (override with
`$YOLOCAR_ENGINE`) and can be disabled with `$YOLOCAR_SKIP_HARDWARE=1` or
`pytest -m "not hardware"`.

## Pipeline

`YoloCarProcessor` implements the stages from `AGENTS.md`:

1. **acquisition** — `yolocar.main.open_source` (video, image, image dir, camera),
   with an optional clockwise `rotate` applied per frame (`--rotate`).
2. **preprocessing** — `letterbox()` (aspect ratio preserved, pad 114,
   `INTER_LINEAR`) + `to_tensor()` (BGR→RGB, `1x3xHxW`, dtype/scale taken from
   the engine input).
3. **inference** — `TensorRTEngine.infer()` on a non-blocking stream with
   page-locked host buffers; `execute_async_v3`.
4. **NMS** — `nms()` / `multiclass_nms()`, greedy, per class (or agnostic).
5. **domain filter** — only `config.target_class_ids`
   (default `ROAD_CLASS_IDS` = person, bicycle, car, motorcycle, bus, truck).
   Applied *before* NMS so non-road candidates never reach the suppression step.
6. **coordinate adaptation** — `LetterboxInfo.to_source()` undoes scale/pad and
   clips to the original frame.

`process_frame(frame, processor)` is the module-level entry point; the four
stages are also exposed separately (`preprocess`, `infer`, `postprocess`) so
callers can time them individually.
