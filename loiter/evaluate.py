"""Evaluate predicted loitering alerts against labelled ground-truth episodes.

Ground truth format (JSON list), one entry per episode that SHOULD alert:
  {"camera_id": "CAM_001", "zone": "Restricted Area", "start_ts": 1700000000.0, "end_ts": 1700000600.0}
start_ts/end_ts are the interval the person was in the zone (same epoch clock as the events).

Matching: an alert is a true positive if it has the same camera and zone and its detection_ts falls
within [start_ts - tolerance, end_ts + tolerance]. Each ground-truth episode can match at most one alert.
Unmatched alerts are false positives. Unmatched episodes are false negatives.
"""
import argparse
import json
from collections import Counter
from typing import Dict, List, Optional

PRECISION_TARGET = 0.90
FALSE_ALERTS_TARGET = 2.0  # per camera per shift


def evaluate(
    events: List[dict],
    ground_truth: List[dict],
    tolerance_s: float = 30.0,
    shifts_per_camera: int = 1,
) -> Dict:
    if shifts_per_camera <= 0:
        raise ValueError("shifts_per_camera must be > 0")

    gt = [dict(g, _matched=False) for g in ground_truth]
    tp = 0
    fp_by_camera: Counter = Counter()

    for ev in sorted(events, key=lambda e: e["detection_ts"]):
        match = None
        for g in gt:
            if g["_matched"] or g["camera_id"] != ev["camera_id"] or g["zone"] != ev["zone"]:
                continue
            if g["start_ts"] - tolerance_s <= ev["detection_ts"] <= g["end_ts"] + tolerance_s:
                match = g
                break
        if match is not None:
            match["_matched"] = True
            tp += 1
        else:
            fp_by_camera[ev["camera_id"]] += 1

    fp = sum(fp_by_camera.values())
    fn = sum(1 for g in gt if not g["_matched"])
    precision: Optional[float] = tp / (tp + fp) if (tp + fp) else None
    recall: Optional[float] = tp / (tp + fn) if (tp + fn) else None

    cameras = {g["camera_id"] for g in ground_truth} | {e["camera_id"] for e in events}
    fa_per_cam_shift = {c: fp_by_camera[c] / shifts_per_camera for c in sorted(cameras)}
    worst_fa = max(fa_per_cam_shift.values()) if fa_per_cam_shift else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "false_alerts_per_camera_per_shift": fa_per_cam_shift,
        "worst_false_alerts_per_camera_per_shift": worst_fa,
        "shifts_per_camera": shifts_per_camera,
        "tolerance_s": tolerance_s,
        "meets_precision_target": precision is not None and precision >= PRECISION_TARGET,
        "meets_false_alert_target": worst_fa <= FALSE_ALERTS_TARGET,
        "targets": {"precision_min": PRECISION_TARGET, "false_alerts_max": FALSE_ALERTS_TARGET},
        "note": "Results are only valid for the labelled data supplied. Do not extrapolate to other sites.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score loitering alerts against labelled episodes.")
    parser.add_argument("--events", required=True, help="JSONL file of alerts from run_video.py")
    parser.add_argument("--ground-truth", required=True, help="JSON list of labelled episodes")
    parser.add_argument("--tolerance", type=float, default=30.0, help="seconds of slack at episode edges")
    parser.add_argument("--shifts", type=int, default=1, help="number of shifts covered by the labelled data")
    args = parser.parse_args()

    with open(args.events, encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]
    with open(args.ground_truth, encoding="utf-8") as fh:
        gt = json.load(fh)
    print(json.dumps(evaluate(events, gt, args.tolerance, args.shifts), indent=2))


if __name__ == "__main__":
    main()
