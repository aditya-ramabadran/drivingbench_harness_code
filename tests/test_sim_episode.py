"""The course frame and episode measures on a synthetic straight course."""

import json
import math
import time

import numpy as np
import pytest
from drivingbench.sim.course import COMMA_AHEAD_M, FRONT_M, TRUE_SPEED, Course, Ground, Placed
from drivingbench.sim.episode import Episode
from drivingbench.sim.runtime import Pose, simulator_app
from fastapi.testclient import TestClient

EDGE_M = 14.0  # the synthetic drivable region ends 14 m along the line


@pytest.fixture
def course(tmp_path):
    line = [[float(x), 0.0] for x in range(0, 41)]
    (tmp_path / "course.json").write_text(json.dumps({
        "origin": {"latitude": 37.36, "longitude": -122.13}, "earth_radius_m": 6378137.0,
        "centerline_enu_m": line, "finish_zone_enu_m": [[36, -2], [44, -2], [44, 2], [36, 2]],
    }))  # fmt: skip
    (tmp_path / "cones.json").write_text(json.dumps({"cones": [
        {"x": 10.0, "y": 1.5, "z": 0.0, "kind": "cone"},  # 1.5 m left of the line
        {"x": 20.0, "y": -0.8, "z": 0.0, "kind": "dome"},  # inside the car's path
    ]}))  # fmt: skip
    region = [[-6.0, -4.0], [EDGE_M, -4.0], [EDGE_M, 4.0], [-6.0, 4.0]]
    (tmp_path / "bounds.json").write_text(json.dumps({"polygon_enu_m": region}))
    return Course(tmp_path / "course.json", tmp_path / "cones.json", tmp_path / "bounds.json")


def test_the_comma_rides_ahead_of_the_rear_axle_at_true_speed(course):
    start = course.place(Pose(0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert (start.east, start.north, start.heading) == pytest.approx((0.0, 0.0, 0.0))
    ahead = course.place(Pose(0, 10.0, 0.0, 0.0, 1.0, 0.0))  # vEgo reads 2.4% fast
    assert (ahead.east, ahead.north) == pytest.approx((10 * TRUE_SPEED, 0.0))
    turned = course.place(Pose(0, 0.0, 0.0, math.pi / 2, 0.0, 0.0))  # a quarter turn about the axle
    assert (turned.east, turned.north) == pytest.approx((-COMMA_AHEAD_M, COMMA_AHEAD_M))
    assert turned.heading == pytest.approx(math.pi / 2)


def test_gnss_positions_land_in_the_course_frame(course):
    assert course.enu(37.36, -122.13) == pytest.approx((0.0, 0.0))
    east, north = course.enu(np.array([37.36, 37.36001]), np.array([-122.12999, -122.13]))
    metre = math.radians(1e-5) * 6378137.0  # 1e-5 degree of latitude
    assert east == pytest.approx([metre * math.cos(math.radians(37.36)), 0.0])
    assert north == pytest.approx([0.0, metre])


def test_the_ground_has_no_steps_at_cell_edges():
    ground = Ground(np.array([[0.0, 0.0], [1.0, 1.0]]), origin=(0.0, 0.0), cell=2.0)
    assert ground(1.0, 1.0) == pytest.approx(0.0)  # cell centres keep their heights
    assert ground(3.0, 1.0) == pytest.approx(1.0)
    assert ground(np.array([1.99, 2.0, 2.01]), 1.0) == pytest.approx([0.495, 0.5, 0.505])
    assert ground(-5.0, 9.0) == pytest.approx(0.0)  # held at the nearest edge outside the grid


def drive(episode, points, speed=1.0, start=0.0):
    t = start
    for east, north in points:
        episode.update(t, Placed(east, north, 0.0), speed)
        t += 0.2
    return t


def test_progress_follows_the_v1_rule_and_only_on_course(course):
    episode = Episode(course)
    assert len(episode.reference) == 37  # stops at the first 1 m sample inside the finish
    drive(episode, [(x / 5, 0.0) for x in range(0, 101)])  # 0 to 20 m along the line
    assert episode.progress == pytest.approx(20 / 36)
    drive(episode, [(25.0, 6.0)], start=30)  # 6 m off the line: no credit, counted off course
    assert episode.progress == pytest.approx(20 / 36) and episode.off_course_s > 0


def test_cone_contacts_use_the_body_footprint_and_are_recorded_once(course):
    episode = Episode(course)
    drive(episode, [(x / 5, 0.0) for x in range(0, 151)])  # to 30 m, straight down the line
    # The dome 0.8 m right of the line is under the 1.85 m-wide body; the cone 1.5 m left is not.
    assert [c["cone"] for c in episode.state()["cone_contacts"]] == [1]
    # At 1 m/s time is distance: contact when the front bumper (1.46 m ahead) meets the dome's edge.
    assert episode.state()["cone_contacts"][0]["at"] == pytest.approx(20 - 1.46 - 0.095, abs=0.2)


def test_the_finish_needs_full_progress_inside_the_zone_and_a_stop(course):
    episode = Episode(course)
    t = drive(episode, [(x / 5, 0.0) for x in range(0, 191)])  # to 38 m, still moving
    assert episode.progress == 1.0 and episode.state()["finished_at"] is None
    drive(episode, [(38.0, 0.0)], speed=0.0, start=t)
    assert episode.state()["finished_at"] == pytest.approx(t)


def test_the_whole_body_must_be_inside_the_drivable_region(course):
    assert course.contains(Placed(0.0, 0.0, 0.0))  # the start, its rear 3.26 m behind
    assert course.contains(Placed(EDGE_M - FRONT_M - 0.01, 0.0, 0.0))
    assert not course.contains(Placed(EDGE_M - FRONT_M + 0.01, 0.0, 0.0))  # the bumper is out
    assert not course.contains(Placed(5.0, 3.5, 0.0))  # a side is out, the comma is not
    assert not course.contains(Placed(5.0, -1.5, math.pi / 2))  # turned across: the tail is out
    assert not course.contains(Placed(40.0, 0.0, 0.0))


def test_the_region_edge_stops_the_car_until_the_reset(course, tmp_path):
    headers = {"X-Drivingbench-Protocol": "2", "X-Client": "driver"}
    with TestClient(simulator_app(tmp_path / "state", course=course)) as client:
        native = client.app.state.native

        def until(predicate, timeout=15.0):
            deadline = time.monotonic() + timeout
            while not predicate():
                assert time.monotonic() < deadline, "condition not reached"
                time.sleep(0.05)

        until(lambda: client.get("/status").json().get("boot_id"))
        move = {"direction": "straight", "steering_percent": 0, "speed_mps": 3.0, "duration_s": 20}
        assert client.post("/motion", json=move, headers=headers).json()["status"] == "accepted"
        until(lambda: client.get("/sim/episode").json()["edge_at"] is not None)
        fronts = []
        for _ in range(20):  # the request still asks for 3 m/s for 2 s: the car presses on the edge
            fronts.append(course.place(native.pose(time.monotonic())).east + FRONT_M)
            time.sleep(0.1)
        assert EDGE_M - 0.1 < min(fronts) and max(fronts) <= EDGE_M
        assert client.post("/sim/reset").json()["edge_at"] is None
