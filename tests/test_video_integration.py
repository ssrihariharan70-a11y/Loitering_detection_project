"""Optional end-to-end test: real detector + run_video pipeline on a generated static clip.

Skipped unless both are provided:
  LOITER_WEIGHTS  path to YOLO weights (e.g. yolov8n.pt for a smoke test)
  LOITER_IMAGE    path to an image containing a person whose feet fall in the test zone (see below)
Zone used: x 100-200, y 850-1000 (matches the sample bus.jpg frame from the ultralytics repo, person foot ~ (147, 903)).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.environ.get("LOITER_WEIGHTS")
IMAGE = os.environ.get("LOITER_IMAGE")


@unittest.skipUnless(WEIGHTS and IMAGE, "set LOITER_WEIGHTS and LOITER_IMAGE to run")
class VideoPipelineTest(unittest.TestCase):
    def test_stationary_person_yields_single_alert_at_threshold(self):
        import cv2
        frame = cv2.imread(IMAGE)
        self.assertIsNotNone(frame)
        h, w = frame.shape[:2]
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "static.mp4")
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (w, h))
            for _ in range(90):
                writer.write(frame)
            writer.release()

            cfg = os.path.join(tmp, "cfg.json")
            with open(cfg, "w") as fh:
                json.dump({"camera_id": "CAM_TEST", "zones": [{
                    "name": "Restricted Area", "polygon": [[100, 850], [200, 850], [200, 1000], [100, 1000]],
                    "max_dwell_s": 60, "exit_grace_s": 3, "cooldown_s": 600,
                    "require_stationary": True, "min_stationary_s": 5}]}, fh)

            out = os.path.join(tmp, "alerts.jsonl")
            subprocess.run([sys.executable, os.path.join(HERE, "run_video.py"), "--video", video,
                            "--config", cfg, "--weights", WEIGHTS, "--out", out,
                            "--start-epoch", "1763000000", "--log-level", "WARNING"], check=True)
            with open(out) as fh:
                events = [json.loads(line) for line in fh if line.strip()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["dwell_time_seconds"], 60)
        self.assertEqual(events[0]["zone"], "Restricted Area")


if __name__ == "__main__":
    unittest.main()
