# Loitering Detection (Aplix-CV use case)

Standalone module for detecting people who remain in non-assigned zones beyond a configured time.
It is built from the spec in the CV use-case sheet (Phase 1, no model training). The repository was not
available for inspection, so this module does not integrate with existing Aplix-CV code yet. See "Integration" below.

## Pipeline

    frame → YOLO person detector + ByteTrack IDs → foot point in zone → stillness state
          → per-(track, zone) dwell episode → threshold + stationary check → cooldown → alert (JSON)

## Files

| File | Purpose |
|---|---|
| `loiter/config.py` | `ZoneRule`, `LoiterConfig`; loads JSON, validates values |
| `loiter/geometry.py` | Foot point of a box, point-in-polygon |
| `loiter/tracker.py` | Keeps detector-supplied track IDs; uses greedy IoU association when IDs are unavailable |
| `loiter/stillness.py` | `stationary` / `slow` / `walking` from centre history, in box-height units |
| `loiter/engine.py` | Dwell episodes, exit grace, one alert per episode, cooldown across re-entry |
| `loiter/detector.py` | Ultralytics YOLO wrapper, person class only. Weights path is supplied by you |
| `loiter/alerts.py` | JSON-lines sink and in-memory sink |
| `loiter/evaluate.py` | TP/FP/FN, precision, recall, false alerts per camera per shift vs. labels |
| `run_video.py` | Runs the pipeline on a video file, uses the video's position clock for timestamps |
| `configs/example.json` | Full-frame zone for the 2560x1440 sample video; thresholds are examples, not company values |
| `tests/test_loiter.py` | 32 synthetic-timeline tests, no video or weights needed |
| `tests/test_video_integration.py` | Optional end-to-end test, runs when `LOITER_WEIGHTS` and `LOITER_IMAGE` are set |
| `IMPLEMENTATION_GUIDE.md` | Setup, zone calibration, running, tuning, integration |
| `TESTING_REPORT.md` | What was tested, results, and what was not |

## Rules

- **Dwell:** starts when the person's foot point enters a zone. Continues while inside.
- **Exit reset:** the episode closes when the person is outside (or unseen) for longer than `exit_grace_s`.
  Brief tracking gaps inside the grace period do not reset the timer.
- **Alert:** fires once per episode, when `dwell ≥ max_dwell_s`. If `require_stationary` is true, the person must
  also have not been walking for `min_stationary_s`. Walking across a zone never alerts.
- **Stillness:** `stationary` if the centre stays within `still_tolerance` × box height of its window mean.
  `slow` if average path speed is under `slow_speed` box heights per second. Otherwise `walking`.
  Slow and stationary both count toward loitering. Only walking resets the stationary clock.
- **Cooldown:** after an alert for a (person, zone) pair, a new alert for the same pair is suppressed for `cooldown_s`.
- **Timestamps:** all timing uses epoch seconds passed in, never frame counts.

## Alert format

    {"event_type": "LOITERING_DETECTED", "person_id": 12, "zone": "Restricted Area", "camera_id": "CAM_001",
     "episode_id": 4, "entry_time": "2026-11-12T10:00:00+00:00", "detection_time": "2026-11-12T10:08:00+00:00",
     "entry_ts": 1789...,  "detection_ts": 1789..., "dwell_time_seconds": 480, "status": "ALERT"}

## Running

    pip install numpy opencv-python ultralytics
    python run_video.py --video cam1.mp4 --config configs/example.json \
        --weights /path/to/your_person_weights.pt --out alerts/cam1.jsonl \
        --conf 0.3 --imgsz 960 --start-epoch 1763000000

Weights are not included. The module does not assume a file name, so point `--weights` at the model you have.
The defaults use a 0.3 confidence threshold and 960-pixel inference size to improve detection of small or distant people.
Increase `--imgsz` or lower `--conf` if a camera still misses people, checking for false positives after tuning.

## Testing

    python -m unittest discover -s tests -t . -v

Evaluation on labelled data:

    python -m loiter.evaluate --events alerts/cam1.jsonl --ground-truth labels.json --shifts 10 --tolerance 30

Ground-truth format is documented in `loiter/evaluate.py`. The precision (≥ 90%) and false-alert (≤ 2 per camera
per shift) targets can only be judged on real labelled footage. Nothing in this repository has been measured on
real footage.

## Known limitations

- **Tracker:** video inference uses Ultralytics ByteTrack IDs to handle box movement and short detection gaps; the
  local IoU association remains as a fallback for detections without IDs. Long occlusions and crowded crossings can
  still cause identity switches, so validate tracking on each camera before relying on dwell alerts.
- **Zones:** point-in-polygon on the foot point, in image coordinates. Zones need calibrating per camera.
  No homography to floor coordinates.
- **Stillness thresholds** are in box-height units. They assume a roughly upright person and a fixed camera.
- **Detector:** the module assumes a COCO-style person class (id 0). Not verified against YOLO26s; check the class ids
  of the actual model.
- **Tested only with a public stand-in model:** the video pipeline ran end to end with ultralytics `yolov8n.pt` on a
  generated clip. Company weights (YOLO26s) and real footage have not been tested. See TESTING_REPORT.md.
- **Shift/camera aggregation:** false-alert rate uses a caller-supplied shift count, not calendar-derived shifts.

## Integration with Aplix-CV

Not done. Once the repository is available, the engine can be called with the existing detector and tracker output
(`LoiteringEngine.process(ts, detections)`), and events can be sent to the existing alert path instead of
`JsonlAlertSink`. The Zones Engine polygons should replace `configs/example.json`.
