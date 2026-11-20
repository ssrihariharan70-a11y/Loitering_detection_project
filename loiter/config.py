"""Configuration for zones and loitering rules. Loaded from JSON or built in code."""
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class ZoneRule:
    """One monitored zone and its loitering rule.

    max_dwell_s:        time inside the zone before an alert may fire.
    exit_grace_s:       how long a person may be outside (or untracked) before the dwell timer resets.
    cooldown_s:         after an alert for (person, zone), no new alert for that pair until this elapses.
    require_stationary: if True, the person must be non-walking for min_stationary_s when the alert fires.
    """
    name: str
    polygon: Tuple[Point, ...]
    max_dwell_s: float
    exit_grace_s: float = 3.0
    cooldown_s: float = 300.0
    require_stationary: bool = True
    min_stationary_s: float = 5.0

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"zone '{self.name}': polygon needs at least 3 points")
        if self.max_dwell_s <= 0:
            raise ValueError(f"zone '{self.name}': max_dwell_s must be > 0")
        for field_name in ("exit_grace_s", "cooldown_s", "min_stationary_s"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"zone '{self.name}': {field_name} must be >= 0")


@dataclass(frozen=True)
class LoiterConfig:
    """Camera-level configuration.

    iou_threshold:      minimum IoU to associate a detection with an existing track.
    max_lost_s:         a track unseen for longer than this is dropped (new ID on return).
    stillness_window_s: window over which motion is measured.
    still_tolerance:    max deviation from window mean centre, as a fraction of person box height, to count as stationary.
    slow_speed:         average path speed (box heights per second) below which motion counts as slow, not walking.
    """
    camera_id: str
    zones: Tuple[ZoneRule, ...]
    iou_threshold: float = 0.3
    max_lost_s: float = 2.0
    stillness_window_s: float = 3.0
    still_tolerance: float = 0.10
    slow_speed: float = 0.25

    def __post_init__(self) -> None:
        if not self.zones:
            raise ValueError("at least one zone is required")
        names = [z.name for z in self.zones]
        if len(set(names)) != len(names):
            raise ValueError("zone names must be unique")
        if not 0 < self.iou_threshold <= 1:
            raise ValueError("iou_threshold must be in (0, 1]")
        if self.max_lost_s < 0 or self.stillness_window_s <= 0:
            raise ValueError("max_lost_s must be >= 0 and stillness_window_s > 0")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoiterConfig":
        zone_optional = ("exit_grace_s", "cooldown_s", "require_stationary", "min_stationary_s")
        zones = tuple(
            ZoneRule(
                name=z["name"],
                polygon=tuple((float(x), float(y)) for x, y in z["polygon"]),
                max_dwell_s=float(z["max_dwell_s"]),
                **{k: z[k] for k in zone_optional if k in z},
            )
            for z in data["zones"]
        )
        top_optional = ("iou_threshold", "max_lost_s", "stillness_window_s", "still_tolerance", "slow_speed")
        return cls(
            camera_id=str(data["camera_id"]),
            zones=zones,
            **{k: float(data[k]) for k in top_optional if k in data},
        )

    @classmethod
    def load(cls, path: str) -> "LoiterConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
