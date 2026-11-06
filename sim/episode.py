"""A simulated drive's benchmark measures: v1.1 course progress, cone contacts and the finish.

Operator-only, like sessions on the car: nothing here reaches the model. Progress follows v1.1's
rule, sampled at 5 Hz on the comma's position: the course line is resampled every metre up to its
first sample inside the finish zone, and progress is the furthest sample within 4 m of the comma,
searched from 10 m behind to 20 m ahead of the furthest reached so far. A cone is touched when its
base overlaps the Model 3's footprint around the comma. The finish needs full progress, the comma
inside the finish zone and the car stopped.
"""

from __future__ import annotations

import math

import numpy as np
from drivingbench.sim.course import FRONT_M, HALF_WIDTH_M, REAR_M, inside

STEP_M, ON_COURSE_M, LOOKBEHIND_M, LOOKAHEAD_M = 1.0, 4.0, 10.0, 20.0
STOPPED_MPS = 0.05
BASE_RADIUS_M = {"cone": 0.09, "dome": 0.095, "tall": 0.13}


def resample(line: np.ndarray, step: float) -> np.ndarray:
    """The polyline's points every `step` metres along it, from its start."""
    along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    s = np.arange(0.0, along[-1] + 1e-9, step)
    return np.stack([np.interp(s, along, line[:, i]) for i in range(2)], axis=1)


class Episode:
    def __init__(self, course):
        self.course = course
        reference = resample(course.line, STEP_M)
        end = np.flatnonzero(inside(course.finish, reference))
        self.reference = reference[: end[0] + 1] if len(end) else reference
        self.cones = np.array([[c["x"], c["y"], BASE_RADIUS_M[c["kind"]]] for c in course.cones])
        self.reset()

    def reset(self):
        self.reached, self.time, self.distance = 0.0, None, 0.0
        self.contacts: dict[int, float] = {}  # cone index -> first contact time
        self.off_course_s, self.finished_at, self.last = 0.0, None, None

    @property
    def progress(self) -> float:
        return self.reached / (STEP_M * (len(self.reference) - 1))

    def update(self, t: float, placed, speed: float):
        """One 5 Hz sample: time (s), the comma's course pose, the car's speed."""
        here = np.array([placed.east, placed.north])
        if self.last is not None:
            self.distance += float(np.linalg.norm(here - self.last[1]))
            if self.nearest_distance(here) > ON_COURSE_M:
                self.off_course_s += t - self.last[0]
        lo = max(0, int((self.reached - LOOKBEHIND_M) / STEP_M))
        hi = min(len(self.reference) - 1, int((self.reached + LOOKAHEAD_M) / STEP_M))
        window = np.linalg.norm(self.reference[lo : hi + 1] - here, axis=1)
        if window.min() <= ON_COURSE_M:
            self.reached = max(self.reached, (lo + int(window.argmin())) * STEP_M)
        for k in self.touching(placed):
            self.contacts.setdefault(int(k), t)
        if (
            self.finished_at is None
            and self.progress >= 1.0
            and inside(self.course.finish, here)
            and speed < STOPPED_MPS
        ):
            self.finished_at = t
        self.time, self.last = t, (t, here)

    def nearest_distance(self, here) -> float:
        return float(np.min(np.linalg.norm(self.reference - here, axis=1)))

    def touching(self, placed) -> np.ndarray:
        """Cones whose base circle overlaps the body rectangle."""
        c, s = math.cos(placed.heading), math.sin(placed.heading)
        d = self.cones[:, :2] - [placed.east, placed.north]
        forward, left = d @ [c, s], d @ [-s, c]
        dx = np.maximum(np.maximum(-REAR_M - forward, forward - FRONT_M), 0)
        dy = np.maximum(np.abs(left) - HALF_WIDTH_M, 0)
        return np.flatnonzero(np.hypot(dx, dy) <= self.cones[:, 2])

    def state(self) -> dict:
        return {
            "progress": round(self.progress, 4),
            "distance_m": round(self.distance, 2),
            "cone_contacts": [{"cone": k, "at": t} for k, t in sorted(self.contacts.items())],
            "off_course_s": round(self.off_course_s, 2),
            "finished_at": self.finished_at,
            "time": self.time,
        }
