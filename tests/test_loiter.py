"""Synthetic-timeline tests for the loitering engine. No video or YOLO weights needed."""
import dataclasses
import math
import unittest
from typing import Callable, List, Optional, Sequence, Tuple

from loiter.config import LoiterConfig, ZoneRule
from loiter.detector import PERSON_CLASS_ID  # noqa: F401  (import check)
from loiter.engine import LoiteringEngine
from loiter.evaluate import evaluate
from loiter.geometry import bottom_center, point_in_polygon
from loiter.stillness import STATIONARY, UNKNOWN, WALKING, classify_motion
from loiter.tracker import Detection, IoUTracker, iou

BOX_H = 100.0
BOX_W = 50.0
SQUARE = ((0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0))
INSIDE = (200.0, 300.0)   # foot point inside SQUARE
OUTSIDE = (600.0, 300.0)  # foot point outside SQUARE


def box_at(foot: Tuple[float, float]) -> Tuple[float, float, float, float]:
    fx, fy = foot
    return (fx - BOX_W / 2, fy - BOX_H, fx + BOX_W / 2, fy)


def make_config(max_dwell=300.0, cooldown=600.0, grace=3.0, require_stationary=True, min_stationary=5.0,
                extra_zones: Sequence[ZoneRule] = ()) -> LoiterConfig:
    zone = ZoneRule(name="Restricted", polygon=SQUARE, max_dwell_s=max_dwell, exit_grace_s=grace,
                    cooldown_s=cooldown, require_stationary=require_stationary, min_stationary_s=min_stationary)
    return LoiterConfig(camera_id="CAM_T", zones=(zone,) + tuple(extra_zones))


def simulate(engine: LoiteringEngine, t0: float, t1: float, step: float,
             persons: Sequence[Callable[[float], Optional[Tuple[float, float]]]]) -> List[dict]:
    """Each person fn maps time -> foot point, or None when not visible in that frame."""
    events: List[dict] = []
    t = t0
    while t <= t1 + 1e-9:
        dets = []
        for fn in persons:
            foot = fn(t)
            if foot is not None:
                dets.append(Detection(bbox=box_at(foot)))
        events.extend(engine.process(t, dets))
        t += step
    return events


def path(waypoints: Sequence[Tuple[float, Tuple[float, float]]]) -> Callable[[float], Tuple[float, float]]:
    """Linear interpolation between (time, foot) waypoints. Keeps per-frame motion small so the IoU tracker keeps the ID."""
    wp = list(waypoints)
    def fn(t: float):
        for (t0, p0), (t1, p1) in zip(wp, wp[1:]):
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0)
                return (p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]))
        return wp[-1][1] if t >= wp[-1][0] else wp[0][1]
    return fn


def stationary(foot):
    return lambda t: foot


class GeometryTests(unittest.TestCase):
    def test_point_in_polygon(self):
        self.assertTrue(point_in_polygon((200, 200), SQUARE))
        self.assertFalse(point_in_polygon((500, 200), SQUARE))
        self.assertFalse(point_in_polygon((-1, 200), SQUARE))

    def test_foot_point_is_bottom_centre(self):
        self.assertEqual(bottom_center((10, 20, 30, 120)), (20.0, 120.0))

    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)


class StillnessTests(unittest.TestCase):
    def _history(self, pts):
        return [(t, x, y, BOX_H) for t, x, y in pts]

    def test_standing_with_jitter_is_stationary(self):
        hist = self._history([(t, 200 + (1.5 if t % 2 else -1.5), 300) for t in range(0, 4)])
        self.assertEqual(classify_motion(hist, 3.0, 3.0, 0.10, 0.25), STATIONARY)

    def test_fast_walk_is_walking(self):
        hist = self._history([(t, 200 + 80 * t, 300) for t in range(0, 4)])  # 0.8 box-heights/s
        self.assertEqual(classify_motion(hist, 3.0, 3.0, 0.10, 0.25), WALKING)

    def test_pacing_is_walking_even_when_window_endpoints_match(self):
        hist = self._history([
            (t / 10.0, 200 + 40 * math.sin(2 * math.pi * (t / 10.0) / 3.0), 300)
            for t in range(31)
        ])
        self.assertEqual(classify_motion(hist, 3.0, 3.0, 0.10, 0.25), WALKING)

    def test_too_little_history_is_unknown(self):
        hist = self._history([(0.0, 200, 300)])
        self.assertEqual(classify_motion(hist, 0.0, 3.0, 0.10, 0.25), UNKNOWN)


class TrackerTests(unittest.TestCase):
    def test_two_people_get_distinct_ids(self):
        tr = IoUTracker()
        tracks = tr.update([Detection(box_at((100, 300))), Detection(box_at((300, 300)))], 0.0)
        self.assertEqual(len({t.track_id for t in tracks}), 2)

    def test_id_kept_across_frames(self):
        tr = IoUTracker()
        first = tr.update([Detection(box_at((200, 300)))], 0.0)[0].track_id
        second = tr.update([Detection(box_at((202, 300)))], 0.5)[0].track_id
        self.assertEqual(first, second)

    def test_track_survives_short_gap_but_not_long_gap(self):
        tr = IoUTracker(max_lost_s=2.0)
        tid = tr.update([Detection(box_at((200, 300)))], 0.0)[0].track_id
        tr.update([], 1.0)  # 1 s gap: kept
        self.assertEqual(tr.update([Detection(box_at((200, 300)))], 1.5)[0].track_id, tid)
        tr.update([], 10.0)  # 8.5 s gap: dropped
        self.assertNotEqual(tr.update([Detection(box_at((200, 300)))], 10.5)[0].track_id, tid)

    def test_expired_track_is_removed_before_reassociation(self):
        tr = IoUTracker(max_lost_s=2.0)
        old_id = tr.update([Detection(box_at((200, 300)))], 0.0)[0].track_id
        new_id = tr.update([Detection(box_at((200, 300)))], 10.0)[0].track_id
        self.assertNotEqual(new_id, old_id)

    def test_external_track_id_keeps_identity_when_box_moves(self):
        tr = IoUTracker()
        first = tr.update([Detection(box_at((100, 300)), track_id=17)], 0.0)[0].track_id
        moved = tr.update([Detection(box_at((300, 300)), track_id=17)], 0.5)[0].track_id
        self.assertEqual(moved, first)

    def test_external_track_id_reconnects_after_local_expiry(self):
        tr = IoUTracker(max_lost_s=2.0)
        first = tr.update([Detection(box_at((200, 300)), track_id=17)], 0.0)[0].track_id
        tr.update([], 3.0)
        reappeared = tr.update([Detection(box_at((200, 300)), track_id=17)], 3.5)[0].track_id
        self.assertEqual(reappeared, first)

    def test_visible_tracks_excludes_temporarily_lost_tracks(self):
        tr = IoUTracker()
        tr.update([Detection(box_at((200, 300)))], 0.0)
        self.assertEqual(len(tr.visible_tracks), 1)
        tr.update([], 0.5)
        self.assertEqual(tr.visible_tracks, [])


class LoiteringEngineTests(unittest.TestCase):
    def test_stationary_person_alerts_once_at_threshold(self):
        engine = LoiteringEngine(make_config(max_dwell=300))
        events = simulate(engine, 0, 1000, 1.0, [stationary(INSIDE)])
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["event_type"], "LOITERING_DETECTED")
        self.assertEqual(ev["zone"], "Restricted")
        self.assertEqual(ev["camera_id"], "CAM_T")
        self.assertEqual(ev["status"], "ALERT")
        self.assertEqual(ev["dwell_time_seconds"], 300)
        self.assertEqual(ev["detection_ts"], 300.0)
        self.assertEqual(ev["entry_ts"], 0.0)

    def test_no_duplicate_alerts_for_long_episode(self):
        engine = LoiteringEngine(make_config(max_dwell=60, cooldown=0))
        events = simulate(engine, 0, 2000, 1.0, [stationary(INSIDE)])
        self.assertEqual(len(events), 1)

    def test_walker_crossing_zone_does_not_alert(self):
        engine = LoiteringEngine(make_config(max_dwell=20))
        # Walks 0.8 box-heights/s (80 px/s) across the 400 px zone in ~5 s.
        walk = lambda t: (t * 80.0, 300.0) if t * 80.0 <= 400 else None
        events = simulate(engine, 0, 30, 0.25, [walk])
        self.assertEqual(events, [])

    def test_person_walking_inside_zone_does_not_alert(self):
        engine = LoiteringEngine(make_config(max_dwell=60, min_stationary=5))
        # Oscillates left-right inside the zone at walking speed for 200 s.
        pace = lambda t: (200 + 150 * math.sin(2 * math.pi * t / 10.0), 300.0)
        events = simulate(engine, 0, 200, 0.5, [pace])
        self.assertEqual(events, [])

    def test_pacing_period_matching_window_does_not_alert(self):
        engine = LoiteringEngine(make_config(max_dwell=30, min_stationary=5))
        pace = lambda t: (200 + 40 * math.sin(2 * math.pi * t / 3.0), 300.0)
        events = simulate(engine, 0, 40, 0.1, [pace])
        self.assertEqual(events, [])

    def test_exit_resets_dwell_timer(self):
        engine = LoiteringEngine(make_config(max_dwell=300, grace=3.0))
        # Inside until 200 s, walks out (20 px/s), stands outside, walks back in at 240 s, stays.
        person = path([(0, INSIDE), (200, INSIDE), (220, OUTSIDE), (240, OUTSIDE), (260, INSIDE), (600, INSIDE)])
        # First visit ends ~210 s (0 -> ~210 s = 210 s, below threshold). Second visit: 250 s -> 500 s = 250 s.
        events = simulate(engine, 0, 500, 1.0, [person])
        self.assertEqual(events, [])

    def test_exit_then_full_dwell_alerts_on_second_visit(self):
        engine = LoiteringEngine(make_config(max_dwell=300, grace=3.0, cooldown=0))
        person = path([(0, INSIDE), (200, INSIDE), (220, OUTSIDE), (240, OUTSIDE), (260, INSIDE), (800, INSIDE)])
        events = simulate(engine, 0, 800, 1.0, [person])
        self.assertEqual(len(events), 1)
        # Second visit: back inside at x<400 around t=250, dwell 300 s -> alert at ~550 s.
        self.assertEqual(events[0]["episode_id"], 2)
        self.assertAlmostEqual(events[0]["entry_ts"], 250.0, delta=1.5)
        self.assertAlmostEqual(events[0]["detection_ts"], 550.0, delta=1.5)

    def test_brief_tracking_loss_does_not_reset_timer(self):
        engine = LoiteringEngine(make_config(max_dwell=300, grace=3.0))
        # Detection drops for 1 s (inside the 2 s tracker tolerance and the 3 s zone grace).
        person = lambda t: None if 100 <= t < 101 else INSIDE
        events = simulate(engine, 0, 400, 0.5, [person])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["entry_ts"], 0.0)
        self.assertEqual(events[0]["detection_ts"], 300.0)

    def test_long_tracking_loss_starts_new_episode(self):
        engine = LoiteringEngine(make_config(max_dwell=300, grace=3.0, cooldown=0))
        # Unseen for 10 s (longer than the 2 s tracker tolerance): the track expires and returns with a new ID.
        # Second visit starts at 110 s, so the alert fires at 410 s.
        person = lambda t: None if 100 <= t < 110 else INSIDE
        events = simulate(engine, 0, 420, 1.0, [person])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["entry_ts"], 110.0)
        self.assertEqual(events[0]["detection_ts"], 410.0)
        self.assertEqual(events[0]["episode_id"], 2)

    def test_multiple_people_each_get_own_event(self):
        engine = LoiteringEngine(make_config(max_dwell=120, cooldown=0))
        a = stationary((150.0, 300.0))
        b = stationary((300.0, 320.0))
        events = simulate(engine, 0, 300, 1.0, [a, b])
        self.assertEqual(len(events), 2)
        self.assertEqual(len({e["person_id"] for e in events}), 2)

    def test_cooldown_suppresses_repeat_alert_for_same_person(self):
        engine = LoiteringEngine(make_config(max_dwell=100, cooldown=600, grace=3.0))
        # Alert at 100 s while standing. Walks out 110-120 s, stands outside, walks back in 125-145 s.
        # Dwell reaches 100 s again at ~235 s, but cooldown (600 s) suppresses it.
        person = path([(0, INSIDE), (100, INSIDE), (120, OUTSIDE), (125, OUTSIDE), (145, INSIDE), (400, INSIDE)])
        events = simulate(engine, 0, 300, 1.0, [person])
        self.assertEqual(len(events), 1)
        self.assertEqual(engine.suppressed_by_cooldown, 1)

    def test_repeat_alert_allowed_after_cooldown(self):
        engine = LoiteringEngine(make_config(max_dwell=100, cooldown=30, grace=3.0))
        person = path([(0, INSIDE), (100, INSIDE), (120, OUTSIDE), (125, OUTSIDE), (145, INSIDE), (400, INSIDE)])
        events = simulate(engine, 0, 300, 1.0, [person])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1]["person_id"], events[0]["person_id"])

    def test_stationary_requirement_blocks_alert_while_walking(self):
        engine = LoiteringEngine(make_config(max_dwell=60, min_stationary=5))
        # Stands 0-50 s, then pacing inside the zone 50-200 s: dwell passes 60 s while walking.
        def person(t):
            if t < 50:
                return INSIDE
            return (200 + 150 * math.sin(2 * math.pi * t / 10.0), 300.0)
        events = simulate(engine, 0, 200, 0.5, [person])
        self.assertEqual(events, [])

    def test_zone_thresholds_are_independent(self):
        dock = ZoneRule(name="Dock", polygon=((600, 0), (900, 0), (900, 400), (600, 400)),
                        max_dwell_s=50, cooldown_s=0)
        cfg = make_config(max_dwell=300, extra_zones=(dock,))
        engine = LoiteringEngine(cfg)
        dock_foot = (700.0, 300.0)
        events = simulate(engine, 0, 60, 1.0, [stationary(dock_foot)])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["zone"], "Dock")
        self.assertEqual(events[0]["dwell_time_seconds"], 50)

    def test_timestamps_must_not_go_backwards(self):
        engine = LoiteringEngine(make_config())
        engine.process(10.0, [])
        with self.assertRaises(ValueError):
            engine.process(9.0, [])

    def test_irregular_frame_intervals_use_timestamps(self):
        config = dataclasses.replace(
            make_config(max_dwell=300, grace=3.0, require_stationary=False),
            max_lost_s=200.0,
        )
        engine = LoiteringEngine(config)
        stamps = [0.0, 0.04, 2.0, 7.5, 60.0, 150.0, 299.0, 300.0, 301.0]
        events = []
        for ts in stamps:
            events += engine.process(ts, [Detection(box_at(INSIDE))])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["detection_ts"], 300.0)


class ConfigTests(unittest.TestCase):
    def test_from_dict_and_validation(self):
        cfg = LoiterConfig.from_dict({
            "camera_id": "C1",
            "zones": [{"name": "Z", "polygon": [[0, 0], [1, 0], [1, 1]], "max_dwell_s": 10}],
        })
        self.assertEqual(cfg.zones[0].cooldown_s, 300.0)
        with self.assertRaises(ValueError):
            ZoneRule(name="bad", polygon=((0, 0), (1, 1)), max_dwell_s=10)
        with self.assertRaises(ValueError):
            LoiterConfig(camera_id="C", zones=(ZoneRule("a", SQUARE, 1), ZoneRule("a", SQUARE, 1)))


class EvaluationTests(unittest.TestCase):
    def test_shifts_per_camera_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "shifts_per_camera must be > 0"):
            evaluate([], [], shifts_per_camera=0)

    def test_precision_fp_fn_and_per_shift_rate(self):
        gt = [
            {"camera_id": "C1", "zone": "Z", "start_ts": 0.0, "end_ts": 400.0},
            {"camera_id": "C1", "zone": "Z", "start_ts": 1000.0, "end_ts": 1400.0},
        ]
        events = [
            {"camera_id": "C1", "zone": "Z", "detection_ts": 300.0},   # TP
            {"camera_id": "C1", "zone": "Z", "detection_ts": 5000.0},  # FP
            {"camera_id": "C1", "zone": "Z", "detection_ts": 5100.0},  # FP
        ]
        r = evaluate(events, gt, tolerance_s=0.0, shifts_per_camera=2)
        self.assertEqual((r["true_positives"], r["false_positives"], r["false_negatives"]), (1, 2, 1))
        self.assertAlmostEqual(r["precision"], 1 / 3)
        self.assertEqual(r["false_alerts_per_camera_per_shift"]["C1"], 1.0)
        self.assertFalse(r["meets_precision_target"])
        self.assertTrue(r["meets_false_alert_target"])

    def test_each_episode_matches_once(self):
        gt = [{"camera_id": "C1", "zone": "Z", "start_ts": 0.0, "end_ts": 400.0}]
        events = [{"camera_id": "C1", "zone": "Z", "detection_ts": 100.0},
                  {"camera_id": "C1", "zone": "Z", "detection_ts": 200.0}]
        r = evaluate(events, gt, tolerance_s=0.0)
        self.assertEqual((r["true_positives"], r["false_positives"]), (1, 1))

    def test_wrong_zone_is_false_positive(self):
        gt = [{"camera_id": "C1", "zone": "Z", "start_ts": 0.0, "end_ts": 400.0}]
        events = [{"camera_id": "C1", "zone": "Other", "detection_ts": 100.0}]
        r = evaluate(events, gt)
        self.assertEqual((r["true_positives"], r["false_positives"], r["false_negatives"]), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
