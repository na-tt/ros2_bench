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
│   ├── __init__.py        # lazy re-exports
│   ├── config.py          # tunables + COCO class table (no logic)
│   ├── cuda.py            # ctypes -> libcudart (buffers, stream, copies)
│   ├── engine.py          # TensorRT deserialize, buffers, infer
│   ├── acquisition.py     # stage 1: sources + rotation correction
│   ├── preprocessing.py   # stage 2: letterbox + BGR->RGB NCHW tensor
│   ├── inference.py       # stage 3: engine call + detect-head decoding
│   ├── postprocessing.py  # stages 4-6: NMS, domain filter, coord adaptation
│   ├── detections.py      # per-frame detections sidecar (JSONL artefact)
│   ├── render.py          # optional: annotation, HUD + frame sinks
│   ├── demo.py            # offline renderer: sidecar + clip -> video
│   ├── processor.py       # facade: owns the engine, wires stages 2-6
│   ├── main.py            # standalone CLI / benchmark
│   └── py.typed
└── tests/
    ├── test_package.py             # import/wiring smoke tests
    ├── test_yolocar_pipeline.py    # acquisition / preprocessing / inference / facade
    ├── test_yolocar_postprocessing.py  # stages 4-6 + frozen golden output
    ├── test_yolocar_detections.py  # detections sidecar writer/reader
    ├── test_yolocar_demo.py        # renderer: join, timing, composite
    ├── test_yolocar_cuda.py        # CUDA bindings with fakes
    ├── test_yolocar_engine.py      # TensorRT glue with fakes
    ├── test_yolocar_hardware.py    # real driver/engine, skips when unavailable
    └── golden/postprocess.json     # pre-split detection output (refactor safety net)
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
`--max-frames`, `--no-pinned`, `--trt-log-level`, `--rotate`.

Run-length flags for long measurements:

* `--loops N` — replay the source N times instead of re-encoding longer video.
  Each frame carries its replay index, so drift between early and late replays
  stays visible. Replays seek back to the first keyframe (no decoder re-init);
  if the backend cannot rewind, the replay stops rather than silently idling.
* `--duration S` — stop after S seconds of *measurement* time. The clock starts
  once the engine is loaded, so startup does not eat into the budget.
* `--records PATH` — write one JSON object per frame (JSONL): `seq`, `loop`,
  `t_mono`, `acq_ms`, `pre_ms`, `infer_ms`, `post_ms`, `present_ms`,
  `latency_ms`, `n_det`. Nothing is discarded at runtime; the steady-state
  window is chosen afterwards, per run.

`--rotate {0,90,180,270}` rotates every frame clockwise at acquisition. It is
needed for clips whose container rotation flag the decoder ignores: OpenCV
exposes it as `CAP_PROP_ORIENTATION_META` but does not apply it for every
backend, so frames can arrive upside down. Feeding the detector upside-down
frames measurably hurts recall, so rotate them upright before comparing
numbers. Both front-ends share the setting through `YoloCarConfig.rotate`, so
the ROS 2 node can apply the same correction.

The summary reports throughput plus a per-stage latency breakdown
(`acquisition / preprocess / inference / postprocess`) so the ROS 2 overhead can
be compared against the same pipeline pieces. `latency_ms` covers stages 2-6
only; `acquisition` is reported separately because it is decoder cost (frame
grab + rotation), not pipeline cost, and a ROS 2 acquisition node pays it in a
different place.

Annotation is **debug only**. `--draw` / `--output` cost real wall-clock time
(drawing plus video encoding), which no node in the measured chain performs, so
that work is timed into `stage_ms.presentation` and subtracted from the reported
throughput:

* `fps` — the benchmark figure, presentation excluded; use this one.
* `fps_wall` — raw wall-clock rate including annotation/encoding.
* `benchmark_time_s` vs `wall_time_s` — the same split for total time.

To benchmark, either omit `--output`/`--draw` or just read `fps`: both are
correct, and annotating a run no longer changes the number you compare.

For showing what the pipeline found — in the monolith *or* in the ROS 2 chain —
use the sidecar + offline renderer below instead: it costs the run nothing and
is the only route that also works for the four-node chain.

## Seeing a run (demo material)

A run can write its final detections next to `records.jsonl`:

```bash
python3 -m yolocar.main --source assets/video/0000f77c-6257be58.mov \
    --rotate 180 --loops 1 \
    --records results/demo/monolith/records.jsonl \
    --detections results/demo/monolith/detections.jsonl \
    --json results/demo/monolith/summary.json
```

`detections.jsonl` is one header line plus one line per frame:

```json
{"kind":"detections","v":1,"source":"assets/video/...","rotate":180,"model":"...","imgsz":640,"conf":0.25,"iou":0.45,"agnostic":false,"classes":[0,1,2,...],"producer":"monolith"}
{"seq":0,"loop":0,"t_mono":12345.678,"det":[[96.4,412.9,180.2,512.0,0.87,2]]}
```

Rows are `[x1, y1, x2, y2, score, class_id]` of the *final* detections, in the
original frame's pixels — exactly what a presentation stage would draw. The
header is validated by the renderer, so a stale file or a rotation mismatch is
an error instead of a picture with wrong boxes. Writing it costs ~0.01 ms per
frame (it is a side artifact, not a stage: `records.jsonl` and `summary.json`
are unchanged, and a front-end that never writes one is still compliant).

Rendering happens **offline**, which matters: encoding a 1280x720 frame with
`mp4v` costs ~16 ms on this board, i.e. about the whole pipeline.

```bash
# one run -> one video, with a HUD (stage timings, frame index, transport cost)
python3 -m yolocar.demo \
    --source assets/video/0000f77c-6257be58.mov --rotate 180 \
    --detections results/demo/monolith/detections.jsonl \
    --records    results/demo/monolith/records.jsonl \
    --out        results/demo/monolith.mp4 --timing wall --hud

# two runs -> one composite, each panel at its own pace
python3 -m yolocar.demo --source assets/video/0000f77c-6257be58.mov --rotate 180 \
    --compare monolith=results/demo/monolith/detections.jsonl,results/demo/monolith/records.jsonl \
              ros-4node=results/demo/ros-4node/detections.jsonl,results/demo/ros-4node/records.jsonl \
    --out results/demo/compare.mp4 --timing wall --hud

# watch a finished run at its real pace (no file, no viewer node needed)
python3 -m yolocar.demo ... --play
```

Key options:

* `--timing wall` (default) — quantised VFR: CFR at `--base-fps` (120) with each
  frame repeated by its own duration, so the video lasts as long as the run did.
  A run at 9 fps therefore produces a video ~4x longer than one at 37 fps, which
  is the point: the overhead becomes visible. `--timing source` writes CFR at the
  clip's rate instead (shorter, hides the pacing) for slide decks.
* `--align wall|seq` (compare mode) — `wall` shows each panel at its own clock
  (the slower one stutters and keeps going after the faster one has finished);
  `seq` puts both on the same clip frame (detection-quality comparison, CFR).
* `--until S` — cut the *output* timeline after S seconds; handy to show the
  divergence without the freeze that follows.
* `--loop N`, `--max-frames N`, `--scale F`, `--no-hud`.

The join is `seq`, never a timestamp: a file source re-decodes in the same order,
so `seq` identifies the frame exactly. Camera sources cannot be re-decoded, so
their sidecars are refused (a frame tap would be the follow-up if that is ever
needed).

### Measurement policy

There is no warmup stage and no runtime discard. `processor.warmup()` exists for
interactive use but the benchmark never calls it: startup and settling are
recorded (`events` in the summary, the first frames in the record stream) and
the steady-state window is selected during analysis. Engine load time is
reported as `startup` and excluded from both latency and throughput.

Clocks are not pinned, the garbage collector is not touched and the desktop
session keeps running, so the machine's natural noise floor is part of the
measurement. That means **compare conditions with interleaved trials**
(A/B/A/B), not back-to-back blocks: with unpinned DVFS a thermal or clock drift
inside the run lands on whichever condition ran second. In a 120 s run of this
pipeline the per-replay median already moves from 16.7 ms to 18.1 ms (~8%), the
same order as the run-to-run spread, so a sequential A-then-B comparison would
be measuring the drift rather than the difference.

The summary also carries `provenance` (python/opencv/numpy versions, git sha,
board, CPU governor, engine size/mtime) so two runs can be checked for
comparability before their numbers are compared.

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

Each stage of the `AGENTS.md` spec lives in its own module, and every
postprocessing stage is a pure function so a ROS 2 chain can either call them
back to back (short chain) or put a node between them (long chain).

1. **acquisition** — `yolocar.acquisition`: `open_source()` (video, image, image
   dir, camera) yields BGR frames; an optional clockwise `rotate` (`--rotate`)
   is applied here, because it corrects a decoder defect rather than the
   model's input contract. `rotate_frame()` is exported for nodes that receive
   frames from elsewhere. Note that detections are therefore expressed in the
   *rotated* frame's pixels.
2. **preprocessing** — `yolocar.preprocessing.preprocess()`: `letterbox()`
   (aspect ratio preserved, pad 114, `INTER_LINEAR`) + `to_tensor()` (BGR→RGB,
   `1x3xHxW`, dtype/scale taken from the engine input). Returns
   `(tensor, LetterboxInfo)` — the two things the downstream stages need.
3. **inference** — `yolocar.inference`: `infer()` runs
   `TensorRTEngine.infer()` on a non-blocking stream with page-locked host
   buffers; `decode_predictions()` splits the raw head into `(4, N)` boxes and
   `(nc, N)` class scores. Kept separate on purpose: the raw head is ~2.8 MB
   per frame, the decoded candidates a few kB, so the ROS 2 topic boundary can
   sit on either side of the decode.
4. **NMS** — `yolocar.postprocessing.nms()`: greedy, per class (or agnostic),
   capped by `max_detections`. Runs on *all* classes, before the domain filter.
5. **domain filter** — `yolocar.postprocessing.domain_filter()`: keeps only
   `config.target_class_ids` (default `ROAD_CLASS_IDS` = person, bicycle, car,
   motorcycle, bus, train, truck, traffic light, stop sign, parking meter).
   Order matters and follows the spec: with `--agnostic-nms` a high-scoring
   non-road box can suppress an overlapping road box before being dropped.
6. **coordinate adaptation** —
   `yolocar.postprocessing.coordinate_adaptation()`: `LetterboxInfo.to_source()`
   undoes scale/pad and clips to the original frame.

Stages 4-6 exchange `yolocar.postprocessing.Candidates` (parallel numpy arrays:
boxes, scores, class ids) instead of a list of objects, so slicing stays
vectorised and the batch maps onto `vision_msgs/Detection2DArray` when the long
chain needs to serialise it. `to_detections()` converts to the presentation
type as the last step.

`YoloCarProcessor` (`yolocar.processor`) is the single-process facade: it owns
the engine and calls the stage functions. `postprocess()` in that module is
nothing but `nms → domain_filter → coordinate_adaptation` in a row;
`process_frame(frame, processor)` is the module-level entry point. The stage
functions can be called and timed individually.
