"""Pure geometry helpers: foot point of a bounding box and point-in-polygon test."""
from typing import Sequence, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


def bottom_center(bbox: BBox) -> Point:
    """Foot point of a person box. Zones are tested against feet, not the box centre."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def point_in_polygon(pt: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting test. Points exactly on an edge may fall either way."""
    x, y = pt
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside
