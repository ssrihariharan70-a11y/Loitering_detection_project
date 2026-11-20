"""Greedy IoU tracker with time-based track expiry.

Each detection matches at most one track, so two tracks cannot share one box.
Tracks are expired by elapsed time, not by frame count, so variable frame rates are handled.
"""
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from loiter.geometry import BBox, bottom_center

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float = 1.0
    track_id: Optional[int] = None


@dataclass
class Track:
    track_id: int
    bbox: BBox
    first_seen: float
    last_seen: float
    last_walking_ts: float
    state: str = "unknown"
    # (timestamp, centre_x, centre_y, box_height) samples used by stillness analysis
    history: Deque[Tuple[float, float, float, float]] = field(default_factory=deque)

    @property
    def foot_point(self):
        return bottom_center(self.bbox)


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost_s: float = 2.0, history_s: float = 10.0) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost_s = max_lost_s
        self.history_s = history_s
        self._tracks: List[Track] = []
        self._next_id = 1
        self._external_ids: Dict[int, int] = {}
        self._visible_tracks: List[Track] = []

    @property
    def visible_tracks(self) -> List[Track]:
        """Tracks that received a detection on the most recent update."""
        return list(self._visible_tracks)

    def update(self, detections: List[Detection], ts: float) -> List[Track]:
        """Associate detections with tracks. Returns tracks visible in this frame."""
        self._tracks = [t for t in self._tracks if ts - t.last_seen <= self.max_lost_s]
        matched_tracks, matched_dets, visible = set(), set(), []

        for di, det in enumerate(detections):
            if det.track_id is None:
                continue
            track_id = self._external_ids.get(det.track_id)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._external_ids[det.track_id] = track_id
            tr = next((track for track in self._tracks if track.track_id == track_id), None)
            if tr is None:
                tr = Track(track_id=track_id, bbox=det.bbox, first_seen=ts, last_seen=ts,
                           last_walking_ts=ts)
                self._tracks.append(tr)
            self._observe(tr, det.bbox, ts)
            matched_tracks.add(track_id)
            matched_dets.add(di)
            visible.append(tr)

        pairs = []
        for ti, tr in enumerate(self._tracks):
            if tr.track_id in matched_tracks:
                continue
            for di, det in enumerate(detections):
                if di in matched_dets or det.track_id is not None:
                    continue
                score = iou(tr.bbox, det.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(key=lambda p: -p[0])

        for _, ti, di in pairs:
            track_id = self._tracks[ti].track_id
            if track_id in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(track_id)
            matched_dets.add(di)
            tr = self._tracks[ti]
            self._observe(tr, detections[di].bbox, ts)
            visible.append(tr)

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            tr = Track(track_id=self._next_id, bbox=det.bbox, first_seen=ts, last_seen=ts, last_walking_ts=ts)
            self._next_id += 1
            self._tracks.append(tr)
            self._observe(tr, det.bbox, ts)
            visible.append(tr)

        self._visible_tracks = visible
        return visible

    def _observe(self, tr: Track, bbox: BBox, ts: float) -> None:
        tr.bbox = bbox
        tr.last_seen = ts
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        tr.history.append((ts, cx, cy, bbox[3] - bbox[1]))
        while tr.history and ts - tr.history[0][0] > self.history_s:
            tr.history.popleft()
