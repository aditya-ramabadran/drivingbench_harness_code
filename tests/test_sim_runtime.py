"""The simulator behind the unchanged producer: plans and frames as clients use them."""

import sys
import time

import pytest
from drivingbench.device.cameras import camera_clock
from drivingbench.device.service import create_app
from drivingbench.sim import render, runtime
from drivingbench.sim.runtime import SimCameras, build, simulator_app
from fastapi.testclient import TestClient

HEADERS = {"X-Drivingbench-Protocol": "2", "X-Client": "driver"}


@pytest.fixture
def producer(tmp_path):
    native = build()
    with TestClient(create_app(tmp_path, native=native, cameras=SimCameras(native))) as client:
        until(lambda: client.get("/status").json().get("boot_id"), timeout=3)
        yield client, native


def until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.05)


def test_a_capture_anchored_plan_drives_the_simulated_car_in_real_time(producer):
    client, native = producer
    status = client.get("/status").json()
    assert status["protocol"] == 2 and status["settings"]["speed_limit_mps"] > 1
    plan = {
        "curvature_inv_m": [0.0] * 40,
        "speed_mps": [1.0] * 40,
        "anchor_s": camera_clock() - 0.4,  # a frame captured 0.2 s ago, less the 0.2 s lead
        "reason": "test plan",
    }
    assert client.post("/chunk", json=plan, headers=HEADERS).json() == {"status": "accepted"}
    until(lambda: native.sim.car.speed > 0.3)
    assert native.sim.car.x > 0 and abs(native.sim.car.y) < 0.05
    stopped = client.post("/stop", json={}, headers=HEADERS).json()
    assert stopped == {"status": "stopping"}
    until(lambda: native.sim.car.standstill)


def test_frames_are_fresh_and_show_the_pose_at_their_capture(producer):
    client, native = producer
    frame = client.get("/camera/narrow")
    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
    assert float(frame.headers["X-Image-Age-S"]) < 0.5
    captured = float(frame.headers["X-Captured-At"])
    assert abs(captured - camera_clock()) < 0.5
    later = client.get("/camera/wide?raw=true")
    assert float(later.headers["X-Captured-At"]) >= captured
    assert later.headers["X-Frame-ID"].startswith("sim")


def test_chat_models_observe_and_move_through_the_same_producer(producer):
    client, native = producer
    seen = client.get("/observe").json()
    assert seen["images"] and seen["images"][0]["age_s"] < 0.5
    move = {"direction": "left", "steering_percent": 20, "speed_mps": 1.0, "duration_s": 5}
    assert client.post("/motion", json=move, headers=HEADERS).json()["status"] == "accepted"
    until(lambda: native.sim.car.steer_deg > 30)  # the Tesla steers even before it rolls


def test_episodes_score_v1_progress_and_reset_to_the_start(tmp_path):
    with TestClient(simulator_app(tmp_path)) as client:
        native = client.app.state.native
        until(lambda: client.get("/status").json().get("boot_id"))
        move = {"direction": "straight", "steering_percent": 0, "speed_mps": 2.0, "duration_s": 8}
        assert client.post("/motion", json=move, headers=HEADERS).json()["status"] == "accepted"
        until(lambda: client.get("/sim/episode").json()["distance_m"] > 6, timeout=10)
        state = client.get("/sim/episode").json()
        assert 0 < state["progress"] < 0.2 and state["off_course_s"] == 0
        assert state["finished_at"] is None and state["cone_contacts"] == []
        fresh = client.post("/sim/reset").json()
        assert fresh["progress"] == 0 and fresh["distance_m"] == 0
        assert native.sim.car.x == 0 and native.sim.car.speed == 0
        until(lambda: client.get("/sim/episode").json()["time"] is not None)
        assert client.get("/sim/episode").json()["distance_m"] < 0.05  # stopped at the start


def test_without_the_splat_the_runtime_names_it_and_the_placeholder(tmp_path, monkeypatch):
    monkeypatch.setitem(render.SPLATS, "full", tmp_path / "absent.splat")
    monkeypatch.setattr(sys, "argv", ["runtime"])
    with pytest.raises(SystemExit, match=r"oct27-nocones-full\.splat.*docs/sim\.md.*--placeholder"):
        runtime.main()
