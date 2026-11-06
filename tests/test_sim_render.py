"""The simulated road camera: its placement on the course, and Chrome rendering it."""

import math
from pathlib import Path

import numpy as np
import pytest
from drivingbench.sim import render
from drivingbench.sim.runtime import Pose


class View:
    def __init__(self):
        self.calls = []

    def render(self, *args):
        self.calls.append(args)
        return np.full((4, 6, 3), 40, np.uint8)


def test_the_camera_rides_the_comma_at_the_measured_mount():
    view = View()
    camera = render.CourseCamera(view, ground=lambda east, north: 0.5)
    pose = Pose(0, 3.0, 1.0, 0.2, 1.0, 0.0)
    frame = camera(pose, "narrow")
    east, north, up, heading, pitch, fov = view.calls[-1]
    # The splat's grey 40 comes out at the road camera's auto-exposure target.
    assert frame.mean() == pytest.approx(render.EXPOSURE_LUMA, abs=0.5)
    placed = camera.course.place(pose)
    assert (east, north) == pytest.approx((placed.east, placed.north))
    assert heading == pytest.approx(placed.heading + math.radians(render.YAW_DEG))  # 1.8° right
    assert up == pytest.approx(0.5 + render.CAMERA_HEIGHT_M)
    assert pitch == pytest.approx(math.radians(render.PITCH_DEG))
    assert fov == pytest.approx(36.3, abs=0.1)  # the road camera's vertical field of view
    camera(pose, "wide")
    assert view.calls[-1][-1] == render.WIDE_FOV_DEG


@pytest.mark.skipif(
    not (Path(render.CHROME).exists() and render.SPLATS["clean"].exists()),
    reason="needs Google Chrome and the built splat",
)
def test_a_lost_webgl_context_reloads_the_page_instead_of_drawing_nothing():
    view = render.ChromeView("clean")
    try:
        before = view.render(20.0, 10.0, 1.24, 1.2, 0.0, render.ROAD_FOV_DEG)
        view.evaluate("document.querySelector('canvas').getContext('webgl2')"
                      ".getExtension('WEBGL_lose_context').loseContext()")  # fmt: skip
        after = view.render(20.0, 10.0, 1.24, 1.2, 0.0, render.ROAD_FOV_DEG)
        assert after[: render.HEIGHT // 2].mean() > 60  # the splat, not a black frame
        assert np.abs(after.astype(int) - before).mean() < 4  # the same view as before
    finally:
        view.close()
