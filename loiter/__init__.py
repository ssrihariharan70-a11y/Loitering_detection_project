"""Loitering detection module: person tracks, zone dwell timers, stillness-gated alerts."""
from loiter.config import LoiterConfig, ZoneRule
from loiter.engine import LoiteringEngine
from loiter.tracker import Detection, IoUTracker, Track

__all__ = ["LoiterConfig", "ZoneRule", "LoiteringEngine", "Detection", "IoUTracker", "Track"]
