"""Motion classification from a track's recent centre history.

Thresholds are fractions of person box height, so they hold across camera distances.
  stationary: centre stays within still_tol of its window mean (standing, shifting weight).
  slow:       average path speed is less than slow_speed box-heights per second (shuffling, slow browsing).
  walking:    anything faster.
  unknown:    not enough history yet.
"""
import math
from typing import Deque, Tuple

STATIONARY = "stationary"
SLOW = "slow"
WALKING = "walking"
UNKNOWN = "unknown"


def classify_motion(
    history: Deque[Tuple[float, float, float, float]],
    now: float,
    window_s: float,
    still_tol: float,
    slow_speed: float,
) -> str:
    samples = [s for s in history if now - s[0] <= window_s + 1e-9]
    if len(samples) < 2:
        return UNKNOWN
    span = samples[-1][0] - samples[0][0]
    if span < 0.5 * window_s:
        return UNKNOWN
    height = samples[-1][3]
    if height <= 0:
        return UNKNOWN

    mean_x = sum(s[1] for s in samples) / len(samples)
    mean_y = sum(s[2] for s in samples) / len(samples)
    max_dev = max(math.hypot(s[1] - mean_x, s[2] - mean_y) for s in samples) / height
    travel = sum(
        math.hypot(current[1] - previous[1], current[2] - previous[2])
        for previous, current in zip(samples, samples[1:])
    ) / height
    speed = travel / span

    if max_dev < still_tol:
        return STATIONARY
    if speed < slow_speed:
        return SLOW
    return WALKING
