"""The site's course in its own frame: where the simulated car is, and what is there.

The site file (`SITE`: sim/foothill-oct27.json, or the file `DRIVINGBENCH_SIM_SITE` names) lists
what the simulator loads (course, ground, cones, drivable region, splats) and what the build
(sim/pipeline) reads and when. The course, ground, cones and region are committed; the splats are
built.

The simulator's pose is the rear axle's (the point that moves along the heading), starting at the
origin with heading 0. `Course.place` puts the comma, the point GNSS measured for v1.1 scoring, on
the course: at the start facing along the course line, the axle advancing at true speed. The
drivable region is the course's lane widened a little beyond its cones; the car's whole body
stays inside it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SITE_FILE = Path(os.environ.get("DRIVINGBENCH_SIM_SITE", REPO / "sim/foothill-oct27.json"))
SITE = json.loads(SITE_FILE.read_text())
COURSE, GROUND, CONES, BOUNDS = (REPO / SITE[k] for k in ("course", "ground", "cones", "bounds"))
SPLATS = {role: REPO / path for role, path in SITE["splats"].items()}
TIMEZONE = ZoneInfo(SITE["timezone"])  # the site's clock, for the site file's times of day
TRUE_SPEED = 0.976  # GNSS Doppler distance per metre of the car's own speed reading (vEgo)
# A Model 3 around its comma: the rear axle 2.28 m behind it (rear overhang 0.98 m), the bumpers
# 1.46 m ahead and 3.26 m behind, the sides 0.925 m left and right.
COMMA_AHEAD_M = 2.28
FRONT_M, REAR_M, HALF_WIDTH_M = 1.46, 3.26, 0.925


@dataclass(frozen=True)
class Placed:
    east: float
    north: float
    heading: float  # radians, counterclockwise from east


def comma(start, start_heading: float, x, y, heading):
    """Where the comma is once the rear axle has moved (x, y) and turned by `heading` in the frame
    of a start with the comma at `start` facing `start_heading`. Arrays broadcast."""
    c, s = math.cos(start_heading), math.sin(start_heading)
    dx, dy = TRUE_SPEED * np.asarray(x), TRUE_SPEED * np.asarray(y)
    h = start_heading + np.asarray(heading)
    east = start[0] + c * dx - s * dy + COMMA_AHEAD_M * (np.cos(h) - c)
    north = start[1] + s * dx + c * dy + COMMA_AHEAD_M * (np.sin(h) - s)
    return east, north


class Ground:
    """The reconstructed asphalt height: one value per cell, bilinear between cell centres. A stepped
    surface catches grazing rays at cell edges (phantom cones) and jolts a camera crossing them."""

    def __init__(self, z, origin, cell: float):
        self.z, self.origin, self.cell = np.asarray(z), np.asarray(origin, float), cell

    @classmethod
    def load(cls, path: Path = GROUND) -> Ground:
        g = json.loads(path.read_text())
        return cls(np.array(g["z"]).reshape(g["shape"]), g["origin"], g["cell_m"])

    def __call__(self, east, north):
        (n, m), z = self.z.shape, self.z
        u = np.clip((np.asarray(east) - self.origin[0]) / self.cell - 0.5, 0, n - 1)
        v = np.clip((np.asarray(north) - self.origin[1]) / self.cell - 0.5, 0, m - 1)
        i, j = np.minimum(u.astype(int), n - 2), np.minimum(v.astype(int), m - 2)
        a, b = u - i, v - j
        return (1 - a) * ((1 - b) * z[i, j] + b * z[i, j + 1]) + a * (
            (1 - b) * z[i + 1, j] + b * z[i + 1, j + 1]
        )


def inside(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Whether each point (..., 2) lies inside the closed polygon (n, 2), by the even-odd rule."""
    x, y = points[..., 0, None], points[..., 1, None]
    (x0, y0), (x1, y1) = polygon.T, np.roll(polygon, -1, axis=0).T
    crosses = (y0 > y) != (y1 > y)
    meet = x0 + (y - y0) * (x1 - x0) / np.where(y1 == y0, 1.0, y1 - y0)
    return (crosses & (x < meet)).sum(-1) % 2 == 1


class Course:
    def __init__(self, course: Path = COURSE, cones: Path = CONES, bounds: Path = BOUNDS):
        c = json.loads(course.read_text())
        self.origin, self.earth_m = c["origin"], c["earth_radius_m"]
        self.line = np.array(c["centerline_enu_m"])
        self.finish = np.array(c["finish_zone_enu_m"])
        self.cones = json.loads(cones.read_text())["cones"]
        self.drivable = np.array(json.loads(bounds.read_text())["polygon_enu_m"])
        (e0, n0), (e1, n1) = self.line[0], self.line[6]
        self.start, self.start_heading = np.array([e0, n0]), math.atan2(n1 - n0, e1 - e0)

    def enu(self, lat, lon):
        """The course frame's east and north (m) of GNSS positions. Arrays broadcast."""
        scale = math.cos(math.radians(self.origin["latitude"])) * self.earth_m
        east = np.radians(np.asarray(lon) - self.origin["longitude"]) * scale
        return east, np.radians(np.asarray(lat) - self.origin["latitude"]) * self.earth_m

    def place(self, pose) -> Placed:
        """The comma's place on the course for a simulator pose."""
        east, north = comma(self.start, self.start_heading, pose.x, pose.y, pose.heading)
        return Placed(float(east), float(north), self.start_heading + pose.heading)

    def contains(self, placed: Placed) -> bool:
        """Whether the whole body is inside the drivable region."""
        c, s = math.cos(placed.heading), math.sin(placed.heading)
        corners = [(FRONT_M, HALF_WIDTH_M), (FRONT_M, -HALF_WIDTH_M),
                   (-REAR_M, -HALF_WIDTH_M), (-REAR_M, HALF_WIDTH_M)]  # fmt: skip
        body = np.array([[placed.east + c * ahead - s * left, placed.north + s * ahead + c * left]
                         for ahead, left in corners])  # fmt: skip
        return bool(inside(self.drivable, body).all())
