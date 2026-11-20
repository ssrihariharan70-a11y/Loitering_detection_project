"""Person detector wrapper around Ultralytics YOLO.

Weights are not bundled and nothing is trained here. The weights path must point to the
model the company actually uses. The module does not assume a specific file name.
"""
import os
from typing import List

from loiter.tracker import Detection

PERSON_CLASS_ID = 0  # COCO class index for "person"


class YoloPersonDetector:
    def __init__(self, weights: str, conf: float = 0.3, imgsz: int = 960, device: str = None) -> None:
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required: pip install ultralytics") from exc
        self._model = YOLO(weights)
        self._conf = conf
        self._imgsz = imgsz
        self._device = device

    def detect(self, frame) -> List[Detection]:
        kwargs = dict(classes=[PERSON_CLASS_ID], conf=self._conf, imgsz=self._imgsz, verbose=False)
        if self._device:
            kwargs["device"] = self._device
        result = self._model.track(frame, persist=True, tracker="bytetrack.yaml", **kwargs)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        track_ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        return [
            Detection(
                bbox=tuple(float(v) for v in xyxy),
                confidence=float(confidence),
                track_id=None if track_id is None else int(track_id),
            )
            for xyxy, confidence, track_id in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), track_ids
            )
        ]
