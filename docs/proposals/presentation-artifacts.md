# Proposal: detections sidecar + offline renderer ("see the pipeline run")

**Status:** implemented. Decisions below are the ones that were signed off; the
*As built* section records what changed during implementation.
**Scope:** shows what the pipeline *found*, for the monolith and for the 4-node
ROS 2 chain, without touching any measured condition.

## Problem

`yolocar`'s product is a detection list (post-NMS, post-domain-filter,
coordinates adapted to the original frame). Neither front-end produces anything
a human can look at, so a review cannot tell a working chain from a broken one.

Presenting *inside* the ROS chain is the obvious route and the wrong one:

| option | why not |
| --- | --- |
| 5th "view" node subscribing `ImageFrame` + `CandidatesFrame` | 2.7 MB/frame DDS fan-out to an extra subscriber, and a whole process's CPU stolen from the four measured stages on the same board. It must also sit in the ack chain or it silently drops frames. |
| append the frame to `CandidatesFrame` | grows the measured wire payload by 2.7 MB — changes the thing being benchmarked |
| `message_filters` timestamp sync | exists in Jazzy's rclpy, but approximate sync mis-pairs or drops frames when latency varies (70–85 ms here). `seq` is an exact join. |
| record frames inside the measured run | `mp4v` at 1280×720 costs **~16 ms/frame**, i.e. about the whole 17 ms pipeline, plus CPU/power contention with the other processes: a different condition, not the baseline |

Also relevant (measured on this board, 1280×720, real clip frames): `avc1`
(OpenCV's libx264) 9.3 ms p50 but 451 ms max and a 2.4 s flush; JPEG sequence
10.6 ms/frame and ~245 MB per replay; `imshow` ~1–2 ms in-process but the
compositor's cost lands on the board invisibly. The system `ffmpeg` has no
libx264 and `h264_nvenc` fails (`EINVAL`); GStreamer has no `nvv4l2h264enc`
(no DeepStream), only `x264enc`/`openh264enc`.

## Decision

Export the detections as a small side artifact next to `records.jsonl`, then
render visuals **offline** from `source + detections (+ records)`. One renderer
serves both front-ends; the measured chain is untouched.

### Recorded decisions

1. **Demo material comes from separate runs**, not from the measured trials
   (`bench/run.py --demo`, a separate invocation). The deciding argument is run
   *shape*: measured trials are `--loops 1000 --duration 120` (mid-clip starts,
   3.4 replays for the monolith, ~1 for ROS), while a demo wants one clean
   replay from `seq` 0. The sidecar's ~0.01 ms/frame is not the reason.
2. **Separate videos *and* a side-by-side composite.**
3. **Composite aligns on wall time** (`--align wall`): the composite timeline is
   the union of both runs' real timestamps, so each panel shows its own latest
   frame and the ROS panel visibly stutters while the composite runs ~3.4×
   longer. `--align seq` (lockstep by clip frame, CFR) stays available for pure
   detection-quality comparison.
4. **`--timing wall` is the default**: quantised VFR — CFR at `--base-fps 120`
   with each source frame repeated `round(Δt × 120)` times. Honest pacing to
   ±4 ms, plays in any player, no VFR container needed. `--timing source` (CFR
   at the clip rate) is available for slide-friendly output, at the price of
   hiding the throughput deficit.
5. **`--play` is included**: the renderer paced by `t_mono` in an OpenCV window.
   It is the only "live" surface the ROS chain can have without a viewer node —
   a finished ROS run can be watched at its true pace. Demo only; `present_ms`
   stays 0 in every recorded run.
6. **Latency timeline plot is included**, in `bench/analyze.py` (it is a view of
   the record stream, and `analyze.py` already owns the analysis window).
7. **`--demo` is a separate command**, never coupled to a measurement run.
8. **A demo run is exactly one replay of the clip** (`--loops 1`, no
   `--duration`, no frame cap) for both the monolith and the ROS chain.

## Component A — detections sidecar

New artifact written *next to* `records.jsonl`, opt-in, disabled by default.
Implemented in `yolocar.detections` (stdlib only, so the ROS node does not pull
`cv2` in through it).

`detections.jsonl` — a header line, then one line per frame:

```json
{"kind":"detections","v":1,"source":"assets/video/0000f77c-6257be58.mov","rotate":180,
 "model":"assets/models/yolov8n_fp16.engine","imgsz":640,"conf":0.25,"iou":0.45,
 "agnostic":false,"classes":[0,1,2,3,5,7,9,11,12],"producer":"ros-4node"}
{"seq":0,"loop":0,"t_mono":12345.678,"det":[[96.4,412.9,180.2,512.0,0.87,2]]}
{"seq":1,"loop":0,"t_mono":12345.706,"det":[]}
```

* `det` rows are `[x1, y1, x2, y2, score, class_id]` — the *final* detections,
  i.e. exactly what a presentation stage would draw (coordinates already in the
  original frame). No letterbox geometry, no raw candidates.
* Coordinates rounded to 0.1 px, scores to 0.001. ~40 B per detection, ~300 KB
  per replay at ~5 detections/frame.
* The header lets the renderer **validate** source and rotation instead of
  trusting the operator (a rotate mismatch draws boxes in the wrong
  orientation), and carries the model path and thresholds for provenance.
* Written per frame with an immediate flush, like `RecordWriter`, so a killed
  run leaves a complete prefix.
* **Contract impact: none.** `records.jsonl` and `summary.json` are untouched,
  so `bench/analyze.py`, the metric dictionary and the golden postprocess
  fixture are unaffected. A front-end that does not emit the sidecar remains
  compliant.

Wired by `--detections PATH` (monolith, `YoloCarConfig.detections_path`) and by
a `detections` parameter (ROS `postprocessing` node, `detections:=` launch arg).

## Component B — offline renderer

`python -m yolocar.demo` (`monolith/yolocar/src/yolocar/demo.py`; drawing and
the HUD live in `yolocar.render`).

```
# one run -> one video
python -m yolocar.demo \
  --source assets/video/0000f77c-6257be58.mov --rotate 180 \
  --detections results/demo/ros-4node/detections.jsonl \
  --records    results/demo/ros-4node/records.jsonl \
  --out        results/demo/ros-4node.mp4 \
  --timing wall --hud

# two runs -> one composite (align wall: each panel at its own pace)
python -m yolocar.demo \
  --source assets/video/0000f77c-6257be58.mov --rotate 180 \
  --compare monolith=results/demo/monolith/detections.jsonl,results/demo/monolith/records.jsonl \
            ros-4node=results/demo/ros-4node/detections.jsonl,results/demo/ros-4node/records.jsonl \
  --out results/demo/compare-monolith-vs-ros-4node.mp4 \
  --timing wall --hud
```

**Join semantics.** Iterate `open_source(...)` once: decode order is
deterministic and `seq` is a global counter across replays, so `seq` is an exact
key — no timestamp matching, no drops. A header mismatch on `source`/`rotate` is
a hard error. A camera sidecar (`kind` != file replay) is refused: the source
cannot be re-decoded.

**Timing modes**

| `--timing` | behaviour | plays where |
| --- | --- | --- |
| `wall` (default) | quantised VFR: CFR at `--base-fps` (120), `round(Δt × base_fps)` copies per frame; the total length equals the run's real duration | anywhere |
| `source` | CFR at the clip's fps; looks real-time and smooth, so it hides the throughput deficit | anywhere |

The quantiser clamps to at least one frame per input frame, uses the median Δ
for the final frame, and keeps the cumulative error bounded by half a base
frame.

**Composite (`--compare`).** Decode the source once in `seq` order; keep one
canvas per panel (~5.5 MB); walk the merged event list (both runs' `t_mono`
relative to each run's first frame) and emit at the base rate. Because each
side's frames are monotonic in `seq`, a single decode cursor serves both panels.
`--align seq` instead emits one output frame per source frame with both panels
in lockstep, at CFR.

**HUD** (`--hud`): label, `seq`/`loop`, `latency_ms`, the per-stage split,
`n_det`, and `transport_ms` when the run is a ROS run. Makes a recording
self-documenting rather than needing a caption.

**`--play`**: same join, but shows frames in a window paced by the recorded
deltas instead of writing a file.

## Demo orchestration (`bench/run.py --demo`)

The harness stays command-template driven — it never imports pipeline code.
`bench/scenarios.json` gains an `artifacts` block with one `run` + one `render`
template per condition, and one `compare` template:

```
results/demo/
├── run.json                       # separate from results/bench/run.json
├── <condition>/                   # records + detections + summary, one replay
└── <condition>.mp4
results/demo/compare-monolith-vs-ros-4node.mp4
```

`--demo` is a separate invocation from a measurement (`bench/run.py --demo
--conditions monolith,ros-4node`), supports `--dry-run`, and cannot fail a
measurement run because it never runs inside one. `monolith-ctrl` is included:
two visually identical control videos are the noise floor, made visible.

## Analysis side

`bench/analyze.py --timeline-out DIR` writes one stage-stacked latency figure
per condition (x = seconds from first frame, y = latency, one band per stage,
`transport` stacked on top for ROS, analysis window shaded). matplotlib 3.6.3 is
present on the board.

## As built

Implemented as described; deviations and additions, all additive:

* **`--until S`** on the renderer: trims the *output* timeline (the last entry
  is cut exactly at S), for the case where the interesting part is the two runs
  diverging rather than the freeze after the faster one has finished.
* **"finished" marker** in the HUD: once a panel has no frames left, its caption
  says `finished (run ended at N s)` — otherwise 3/4 of the composite is a
  motionless panel for no visible reason.
* **HUD is drawn after scaling**, so downscaled panels keep legible text, and it
  is drawn once per output entry (not per repeated frame).
* **Two-panel timeline figure.** A single stacked plot is unreadable for ROS,
  where transport dominates by 4-5x: the top panel is the whole frame, the
  bottom one is the stages without transport and without acquisition.
* **Composite tail** is the slowest panel's frame duration, so the composite
  lasts as long as the slower run rather than as long as the median gap.
* **`summary.json` gained a `detections_log` key** (path or `null`). Extra keys
  are allowed by the contract; `analyze.py` ignores it.
* **`bench/run.py --demo` skips the compare step** when a condition it needs was
  not selected or did not complete, instead of failing the whole pass.

Measured on this board, one full `--demo` pass (10 min 29 s):

| | run | video | render |
| --- | --- | --- | --- |
| `monolith` | 1217 frames, 37.1 fps, p50 15.7 ms | 3940 frames, 120 fps, 32.8 s, 98 MB | 63 s |
| `monolith-ctrl` | 1217 frames, 37.0 fps, p50 16.1 ms | 32.9 s, 99 MB | 61 s |
| `ros-4node` | 1217 frames, 8.9 fps, p50 94.8 ms | 16327 frames, 136.1 s, 223 MB | 177 s |
| `compare` (`--align wall`) | — | 16327 frames, 136.1 s, 188 MB | 119 s |

The 4.1x longer ROS video is the headline: same clip, same detections, visibly
slower pace, with the monolith panel marked `finished` for the remaining 103 s.

## Files

| file | change |
| --- | --- |
| `monolith/yolocar/src/yolocar/detections.py` | **new**: `DetectionLog` writer, `read_header`/`iter_records` reader, rounding, header schema |
| `monolith/yolocar/src/yolocar/render.py` | `draw_hud` |
| `monolith/yolocar/src/yolocar/config.py` | `detections_path` |
| `monolith/yolocar/src/yolocar/main.py` | `--detections`, open/write/close the log |
| `monolith/yolocar/src/yolocar/demo.py` | **new**: join, timing, HUD, sinks, `--compare`, `--play` |
| `monolith/yolocar/src/yolocar/__init__.py` | lazy exports for the new public names |
| `ws/src/yolocar_ros/yolocar_ros/postprocessing_node.py` | `detections` parameter + log |
| `ws/src/yolocar_ros/launch/bench.launch.py` | `detections:=` argument |
| `bench/scenarios.json` | `artifacts` block |
| `bench/run.py` | `--demo` |
| `bench/analyze.py` | `--timeline-out` |
| `bench/README.md`, `monolith/yolocar/README.md`, `AGENTS.md` | artifact schema, recipes, module map |

Unchanged: `yolocar_interfaces`, the stage modules, the
`records.jsonl`/`summary.json` contract, `analyze.py` metrics, `sampler.py`, the
golden postprocess fixture, and every measured condition command.

## Tests

* `DetectionLog`: header written once, rounding, `seq` preserved, disabled → no
  file, flush-per-line.
* Reader: header round-trip, version mismatch rejected, missing/extra keys.
* Renderer join: fake source + handcrafted sidecar → frame count, `seq` mapping,
  `--loop`/`--max-frames` honoured, rotate/source mismatch raises, camera
  sidecar refused.
* Timing quantiser: repeats sum to `round(Σ Δt × base)` ±1, per-frame error
  ≤ 1/(2·base), `Δt ≤ 0` clamped to 1.
* Composite: `--align wall` length equals the slower run, `--align seq` length
  equals the frame count; panel selection matches the merge.
* HUD: contains `latency_ms`, and `transport_ms` for ROS records.
* ROS: `test_reporting.py` asserts the detections sidecar and `records.jsonl`
  stay in lockstep per `seq` (both are written from the same callback, sidecar
  first) and that `summary.json` carries the sidecar path.
* All existing tests stay green, especially `tests/golden/postprocess.json`.
  Final counts: 147 monolith (was 109), 22 passed + 1 skipped ROS (was 20 + 1).

## Risks

1. **Non-deterministic decodes** — a file source re-decodes identically; a
   camera cannot. The renderer refuses camera provenance with a clear error
   (a frame tap in the acquisition node would be the follow-up, out of scope).
2. **Rotation/letterbox mismatch** — header validation turns a silent visual bug
   into an error.
3. **Stale sidecar** — the header carries model/thresholds, and the HUD prints
   the label, so a mismatched artifact is visible rather than assumed.
4. **Output codec** — mp4v plays everywhere and its 16 ms/frame is irrelevant
   offline; a slide-deck H.264 would need OpenCV's `avc1` path or gst
   `x264enc`, not the system ffmpeg.
