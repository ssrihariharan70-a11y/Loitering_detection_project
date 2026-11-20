# Implementation Guide

## 1. Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install numpy opencv-python ultralytics

Python 3.10+ is assumed. The core engine needs only numpy, but the tests and video runner need OpenCV, and the detector needs ultralytics.

## 2. Configure zones

Edit a JSON file (start from `configs/example.json`). Each zone:

| Field | Meaning |
|---|---|
| `name` | Zone label, used in alerts and evaluation. Must be unique. |
| `polygon` | Image-pixel vertices `[[x, y], ...]`, at least 3. Tested against each person's foot point. |
| `max_dwell_s` | Seconds inside before an alert may fire. |
| `exit_grace_s` | Seconds outside or unseen before the dwell timer resets. Default 3. |
| `cooldown_s` | Seconds before the same person can alert again in this zone. Default 300. |
| `require_stationary` | If true, the person must not be walking for `min_stationary_s` when the alert fires. |
| `min_stationary_s` | Length of the stationary requirement. Default 5. |

Top-level tuning keys (all optional): `iou_threshold` (0.3), `max_lost_s` (2.0), `stillness_window_s` (3.0), `still_tolerance` (0.10), `slow_speed` (0.25).

Calibration: draw each polygon on a frame from the camera and check the foot point (bottom centre of the person box) lands where intended. Zones that cover the head or torso will misfire.

## 3. Run on a video

    python run_video.py --video cam1.mp4 --config my_config.json \
        --weights /path/to/weights.pt --out alerts/cam1.jsonl \
        --conf 0.3 --imgsz 960 --start-epoch 1763000000

`--start-epoch` is the wall-clock time of the first frame, in epoch seconds. Omit it to use the current time. Pass the real recording start for correct entry and detection times. Alerts are written as JSON lines.
`--conf` and `--imgsz` tune detector sensitivity and image scale; the defaults are 0.3 and 960.

## 4. Use the engine directly

The engine has no OpenCV dependency, so it can be driven from any detection source:

    from loiter import LoiterConfig, LoiteringEngine, Detection
    engine = LoiteringEngine(LoiterConfig.load("my_config.json"))
    events = engine.process(ts, [Detection(bbox=(x1, y1, x2, y2), confidence=0.9), ...])

`ts` must be non-decreasing epoch seconds. Each call returns the alerts raised on that frame.

## 5. Evaluate against labels

Create a labels file of episodes that should alert:

    [{"camera_id": "CAM_001", "zone": "Restricted Area", "start_ts": 1763000000.0, "end_ts": 1763000900.0}]

Then run:

    python -m loiter.evaluate --events alerts/cam1.jsonl --ground-truth labels.json --shifts 10 --tolerance 30

`--shifts` is the number of shifts the labelled footage covers, for each camera. False alerts per camera per shift are computed from it.

## 6. Tuning guide

- **Pacing is mistaken for loitering:** lower `slow_speed` so faster movement is classified as walking; check that
  `stillness_window_s` covers the person's movement cycle.
- **Misses people who pace inside a zone:** this is intended. Pacing counts as walking and does not alert while `require_stationary` is true. Set it to false to alert on dwell alone.
- **Duplicate alerts for one person:** raise `cooldown_s`. Within a single episode, alerts are already limited to one.
- **Dwell resets when a person is briefly lost:** ByteTrack IDs help reconnect detections across short gaps. If a gap exceeds the tracker's buffer, raise `max_lost_s` and `exit_grace_s` only after checking for identity swaps in the camera footage.

## 7. Integrating with another alert system

`LoiteringEngine.process` returns plain dicts. Pass them to any sink. `JsonlAlertSink` and `ListAlertSink` are included as examples.

## 8. Known gaps before production

- Real-footage accuracy has not been measured (see TESTING_REPORT.md).
- ByteTrack can still switch identities during long occlusions or crowded crossings; confirm identities and zone dwell timing against camera footage.
- Image-space zones only; no floor-plane mapping.
- Not yet connected to the Aplix-CV Zones Engine or alert system (no repository access).
