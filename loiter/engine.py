"""Dwell timers and loitering decisions. Pure logic: no OpenCV, no model, no I/O.

Per (track, zone) pair:
  - An episode starts when the track's foot point enters the zone.
  - The episode ends when the track has been outside (or unseen) for longer than the zone's exit_grace_s.
  - An alert fires once per episode, when dwell >= max_dwell_s and, if required, the person has
    not been walking for min_stationary_s.
  - Cooldown suppresses a new alert for the same (track, zone) pair that re-enters too soon.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from loiter.config import LoiterConfig, ZoneRule
from loiter.geometry import point_in_polygon
from loiter.stillness import WALKING, classify_motion
from loiter.tracker import Detection, IoUTracker, Track

logger = logging.getLogger(__name__)

Key = Tuple[int, str]  # (track_id, zone_name)


@dataclass
class _Episode:
    episode_id: int
    entry_ts: float
    last_inside_ts: float
    alerted: bool = False


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


class LoiteringEngine:
    def __init__(self, config: LoiterConfig, tracker: Optional[IoUTracker] = None) -> None:
        self.config = config
        self.tracker = tracker or IoUTracker(
            iou_threshold=config.iou_threshold,
            max_lost_s=config.max_lost_s,
            history_s=max(10.0, 2 * config.stillness_window_s),
        )
        self._zones: Dict[str, ZoneRule] = {z.name: z for z in config.zones}
        self._episodes: Dict[Key, _Episode] = {}
        self._last_alert_ts: Dict[Key, float] = {}
        self._episode_seq = 0
        self._last_ts: Optional[float] = None
        self.suppressed_by_cooldown = 0

    @property
    def active_episodes(self) -> Dict[Key, _Episode]:
        return dict(self._episodes)

    def process(self, ts: float, detections: Sequence[Detection]) -> List[dict]:
        """Feed one frame's detections at timestamp ts (epoch seconds). Returns new alert events."""
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError(f"timestamps must be non-decreasing ({ts} < {self._last_ts})")
        self._last_ts = ts

        cfg = self.config
        tracks = self.tracker.update(list(detections), ts)

        # 1. Motion state for each visible track.
        for tr in tracks:
            tr.state = classify_motion(tr.history, ts, cfg.stillness_window_s, cfg.still_tolerance, cfg.slow_speed)
            if tr.state == WALKING:
                tr.last_walking_ts = ts

        # 2. Which (track, zone) pairs contain a foot point this frame.
        inside: Dict[Key, Track] = {}
        for tr in tracks:
            for zone in self._zones.values():
                if point_in_polygon(tr.foot_point, zone.polygon):
                    inside[(tr.track_id, zone.name)] = tr

        # 3. Start or extend episodes for pairs currently inside.
        for key in inside:
            ep = self._episodes.get(key)
            if ep is None:
                self._episode_seq += 1
                ep = _Episode(episode_id=self._episode_seq, entry_ts=ts, last_inside_ts=ts)
                self._episodes[key] = ep
            ep.last_inside_ts = ts

        # 4. Close episodes that have been out of the zone longer than the zone's grace period.
        for key in list(self._episodes):
            ep = self._episodes[key]
            grace = self._zones[key[1]].exit_grace_s
            if ep.last_inside_ts < ts and ts - ep.last_inside_ts > grace:
                del self._episodes[key]

        # 5. Alert decisions for pairs currently inside.
        events: List[dict] = []
        for key, tr in inside.items():
            ep = self._episodes[key]
            rule = self._zones[key[1]]
            if ep.alerted:
                continue  # one alert per episode
            if ts - ep.entry_ts < rule.max_dwell_s:
                continue
            if rule.require_stationary and ts - tr.last_walking_ts < rule.min_stationary_s:
                continue
            ep.alerted = True
            last = self._last_alert_ts.get(key)
            if last is not None and ts - last < rule.cooldown_s:
                self.suppressed_by_cooldown += 1
                logger.info("suppressed alert track=%s zone=%s (cooldown)", key[0], key[1])
                continue
            self._last_alert_ts[key] = ts
            events.append(self._make_event(key, ep, ts))

        self._prune_alert_history(ts)
        return events

    def _make_event(self, key: Key, ep: _Episode, ts: float) -> dict:
        track_id, zone_name = key
        return {
            "event_type": "LOITERING_DETECTED",
            "person_id": track_id,
            "zone": zone_name,
            "camera_id": self.config.camera_id,
            "episode_id": ep.episode_id,
            "entry_time": _iso(ep.entry_ts),
            "detection_time": _iso(ts),
            "entry_ts": ep.entry_ts,
            "detection_ts": ts,
            "dwell_time_seconds": int(round(ts - ep.entry_ts)),
            "status": "ALERT",
        }

    def _prune_alert_history(self, ts: float) -> None:
        for key in list(self._last_alert_ts):
            if key in self._episodes:
                continue
            if ts - self._last_alert_ts[key] > self._zones[key[1]].cooldown_s:
                del self._last_alert_ts[key]
