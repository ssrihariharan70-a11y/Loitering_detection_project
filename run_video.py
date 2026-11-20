"""Run loitering detection over a video file and write alerts as JSON lines.

Example:
  python run_video.py --video cam1.mp4 --config configs/example.json \
      --weights /path/to/yolo_weights.pt --out alerts/cam1.jsonl --start-epoch 1700000000
"""
import argparse
import logging
import sys
import time

import cv2
import numpy as np

from loiter.alerts import JsonlAlertSink
from loiter.config import LoiterConfig
from loiter.detector import YoloPersonDetector
from loiter.engine import LoiteringEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=960,
                        help="YOLO inference size; larger sizes improve small/distant person detection")
    parser.add_argument("--device", default=None)
    parser.add_argument("--start-epoch", type=float, default=None,
                        help="wall-clock time of the first frame (epoch seconds). Defaults to now.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True,
                        help="show live annotated video (use --no-show for headless)")
    parser.add_argument("--save", default=None,
                        help="write the annotated video to this MP4 path")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    config = LoiterConfig.load(args.config)
    detector = YoloPersonDetector(args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device)
    engine = LoiteringEngine(config)
    sink = JsonlAlertSink(args.out)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logging.error("cannot open video: %s", args.video)
        return 1

    start = args.start_epoch if args.start_epoch is not None else time.time()
    frames = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.save:
        writer = cv2.VideoWriter(
            args.save,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            logging.error("cannot open output video: %s", args.save)
            cap.release()
            return 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Use the video's own position clock, so variable frame rates are handled correctly.
        ts = start + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        detections = detector.detect(frame)
        for event in engine.process(ts, detections):
            sink.emit(event)
        frames += 1

        annotated = frame.copy()
        for zone in config.zones:
            polygon = np.array(zone.polygon, dtype=np.int32)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [polygon], (0, 140, 255))
            cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)
            cv2.polylines(annotated, [polygon], True, (0, 180, 255), 2)
            if len(polygon):
                cv2.putText(annotated, zone.name, tuple(polygon[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

        active = engine.active_episodes
        for track in engine.tracker.visible_tracks:
            x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)
            episodes = [(key, ep) for key, ep in active.items() if key[0] == track.track_id]
            if episodes:
                (track_id, zone_name), episode = episodes[0]
                dwell = ts - episode.entry_ts
                rule = next(zone for zone in config.zones if zone.name == zone_name)
                color = (0, 0, 255) if dwell >= rule.max_dwell_s else (0, 165, 255)
                label = f"ID {track.track_id} | {track.state} | {dwell:.1f}s"
            else:
                color = (0, 255, 0)
                label = f"ID {track.track_id} | {track.state}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.putText(annotated, f"Frame {frames}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("Loitering Detection", annotated)
            delay = max(1, int(1000 / fps))
            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    logging.info("processed %d frames; suppressed by cooldown: %d", frames, engine.suppressed_by_cooldown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
