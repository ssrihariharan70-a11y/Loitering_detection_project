# Testing Report

Original smoke-test environment: Python 3.12, numpy 2.5, OpenCV 5.0, ultralytics 8.4.211, Linux container (CPU only).
Regression suite and provided-video rerun: Windows, Python 3.14.

## 1. Unit and logic tests — PASS (32/32)

Command: `python -m unittest discover -s tests -t . -v`

Covers: geometry (point in polygon, foot point, IoU); stillness classification (stationary, walking, unknown, pacing when window endpoints match); tracker (distinct IDs, ID kept across frames and box movement when supplied by the detector, reconnecting an external ID after local expiry, visible tracks exclude missed detections, survival through a 1 s gap, expiry before reassociation after a long gap); engine (threshold alert, single alert per episode, walking and pacing do not alert, exit resets dwell, brief tracking loss keeps timer, long tracking loss starts a new episode, multiple people, cooldown suppression, cooldown expiry, stationary requirement, per-zone thresholds, timestamp ordering, irregular frame intervals); configuration validation; evaluation metrics and positive shift-count validation.

Note: these tests use synthetic timelines with explicit timestamps. They verify logic, not detection quality.

## 2. Detector integration — PASS (smoke test)

Model: ultralytics `yolov8n.pt` (public COCO weights, used only as a stand-in, not the company's YOLO26s).
Input: `bus.jpg` from the ultralytics repository.
Result: 3 persons detected, confidences 0.83–0.87, person class only.

## 3. Video pipeline end to end — PASS (smoke test)

Clip: the same frame repeated for 90 s at 1 fps. Zone: x 100–200, y 850–1000, max dwell 60 s, min stationary 5 s.
Result: exactly one `LOITERING_DETECTED` alert, `dwell_time_seconds` 60, `person_id` 1, zone and camera fields correct.

This also exercises the video position clock (`CAP_PROP_POS_MSEC`) for timestamps.

Optional integration test: `tests/test_video_integration.py` runs the same check when `LOITER_WEIGHTS` and `LOITER_IMAGE` are set. It is skipped otherwise.

## 4. Provided sample video — PASS (functional run after regression fixes)

Input: `Sample video.mp4`, 2560×1440, 534 frames at 24 fps (about 22.2 s). Model: public `yolov8n.pt`. Zone: full frame; configured dwell threshold: 300 s.
Result: `run_video.py` processed all 534 frames without error and emitted no alert, as expected because the clip is shorter than the dwell threshold. The annotated output shows person detections and the full-frame zone: `alerts/sample_video_fixed_annotated.mp4`.

## 5. Not tested

| Item | Status |
|---|---|
| YOLO26s weights | Not available. Class mapping and output format unverified. |
| Real CCTV footage | None available. |
| Precision ≥ 90% | Not measured. |
| False alerts ≤ 2 per camera per shift | Not measured. |
| Aplix-CV integration | Repository not available. |
| Long-run stability, memory use | Not measured. |
| Frame-rate and resolution sensitivity | Not measured. |
| Occlusion and crowd scenes | Not measured. |

## 6. Result

The module runs end to end on a generated clip with public weights, and the logic is covered by synthetic tests. It has not been shown to meet the acceptance criteria. That requires labelled footage from the target sites and the evaluator in `loiter/evaluate.py`.
