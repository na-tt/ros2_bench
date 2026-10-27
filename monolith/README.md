# monolith

Single-process (non-ROS) implementations of the benchmarked CV pipelines.

Each subdirectory is an independent, importable package that owns all of the
pipeline logic. ROS 2 packages in `../ws/src` import from these modules and add
only middleware boilerplate.

## Packages

| Package   | Purpose                                   | ROS 2 wrapper  |
| --------- | ----------------------------------------- | -------------- |
| `yolocar` | YOLO-based road-object detection (TensorRT) | `yolocar_ros` (todo) |

## Layout

```
monolith/
├── pyproject.toml
├── tests/
│   ├── test_package.py           # import/wiring smoke tests
│   └── test_yolocar_pipeline.py  # letterbox / NMS / postprocess unit tests
└── yolocar/
    ├── __init__.py        # lazy re-exports
    ├── config.py          # tunables + COCO class table (no logic)
    ├── cuda.py            # ctypes -> libcudart (buffers, stream, copies)
    ├── engine.py          # TensorRT deserialize, buffers, infer
    ├── processor.py       # letterbox, NMS, domain filter, coord adaptation, draw
    ├── main.py            # standalone CLI / benchmark
    └── py.typed
```

## Install

```bash
pip install -e ./monolith               # core only
pip install -e "./monolith[yolocar]"    # + numpy / opencv for the yolocar pipeline
```

`yolocar` needs at runtime:

* `numpy` and `opencv-python` (the `yolocar` extra),
* the TensorRT Python bindings (`import tensorrt`) — shipped with JetPack,
  imported lazily so the rest of the package works without it,
* the CUDA runtime (`libcudart`), reached through `ctypes` — **no** `pycuda`
  and **no** `cuda-python` required.

Only needed to re-export the ONNX model: `pip install -e "./monolith[export]"`.

## Model

The committed default is a prebuilt FP16 engine for this device:

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
`--max-frames`, `--warmup`, `--no-pinned`, `--trt-log-level`.

The summary reports throughput plus a per-stage latency breakdown
(`preprocess / inference / postprocess`) so the ROS 2 overhead can be compared
against the same pipeline pieces.

## Pipeline

`YoloCarProcessor` implements the stages from `AGENTS.md`:

1. **acquisition** — `yolocar.main.open_source` (video, image, image dir, camera).
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
