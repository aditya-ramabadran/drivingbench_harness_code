"""The fitted Tesla, openpilot as the comma runs it, and the native stack driving the car."""

import importlib.util
import inspect
import math
import sys
import uuid
from dataclasses import replace

import numpy as np
import pytest
from drivingbench.shared.contracts import PROTOCOL, Settings
from drivingbench.sim import logs
from drivingbench.sim.fit import search
from drivingbench.sim.logs import DRIVE_HELPERS, PATCHED, use_openpilot
from drivingbench.sim.plant import Params, Tesla

P = Params(accel_delay_s=0.1, accel_tau_s=0.1, accel_gain=1.0, launch_s=0.2, launch_accel_mps2=0.2,
           steer_delay_s=0.04, steer_tau_s=0.1, steer_rate_dps=400)  # fmt: skip


def straight(angle, speed):
    return 0.0


def test_a_standstill_car_departs_only_after_the_launch_wait_and_then_follows_the_request():
    car = Tesla(P, straight)
    for _ in range(25):  # 0.1 s delay plus 0.15 s of the 0.2 s wait
        car.step(1.0, 0.0, True, True)
    assert car.standstill
    for _ in range(10):
        car.step(1.0, 0.0, True, True)
    assert not car.standstill
    for _ in range(300):
        car.step(1.0, 0.0, True, True)
    assert car.accel == pytest.approx(1.0, abs=0.01) and car.speed > 2.5
    for _ in range(600):
        car.step(-2.0, 0.0, True, True)
    assert car.standstill and car.speed == 0.0  # braking stops; it never reverses


def test_the_wheel_follows_the_applied_angle_at_the_epas_rate_and_holds_when_released():
    car = Tesla(P, straight)
    for _ in range(14):  # 0.04 s delay, then 0.1 s at the 400 deg/s cap
        car.step(0.0, 180.0, True, True)
    assert car.steer_deg == pytest.approx(40.0, abs=4.1)
    for _ in range(300):
        car.step(0.0, 180.0, True, True)
    assert car.steer_deg == pytest.approx(180.0, abs=0.01)
    for _ in range(100):
        car.step(0.0, 0.0, False, True)
    assert car.steer_deg == pytest.approx(180.0, abs=0.01)  # no lateral control: nothing moves


def test_the_pose_turns_by_the_yaw_scale_times_the_vehicle_model_yaw():
    curvature = lambda angle, speed: 0.1 if angle > 0 else 0.0  # noqa: E731
    car = Tesla(replace(P, yaw_scale=1.1), curvature, speed=1.0, steer_deg=90.0)
    for _ in range(100):
        car.step(0.0, 90.0, True, True)
    assert car.heading == pytest.approx(1.1 * 0.1 * 1.0 * 1.0, rel=0.02)


def test_search_moves_each_field_to_its_best_candidate():
    target = {"accel_delay_s": 0.2, "steer_tau_s": 0.06}
    error = lambda p: sum((getattr(p, k) - v) ** 2 for k, v in target.items())  # noqa: E731
    found = search(Params(), error, {k: [0.02, 0.06, 0.1, 0.2, 0.3] for k in target})
    assert (found.accel_delay_s, found.steer_tau_s) == (0.2, 0.06)


def tesla_params():
    use_openpilot()
    from opendbc.car.tesla.interface import CarInterface

    return CarInterface.get_non_essential_params("TESLA_MODEL_3")


def motion(sim, direction, percent, speed, duration):
    controller = sim.bridge.controller
    return {
        "protocol": PROTOCOL, "request_id": uuid.uuid4().hex, "operation": "motion",
        "boot_id": controller.boot_id, "command_epoch": controller.command_epoch,
        "motion": {"direction": direction, "steering_percent": percent, "speed_mps": speed,
                   "duration_s": duration, "reason": ""},
        "settings": Settings().model_dump(),
    }  # fmt: skip


def test_the_native_stack_drives_the_simulated_tesla_like_the_car():
    from drivingbench.sim.car import SimNative

    sim = SimNative(tesla_params(), params=P, steer_ratio=12.4, stiffness=1.0, angle_offset=-1.9)
    assert sim.call(motion(sim, "straight", 0, 1.0, 10))["status"] == "accepted"
    sim.advance(6.0)
    assert sim.car.speed == pytest.approx(1.0, abs=0.05)  # the requested speed, no overshoot
    x, y, heading = sim.pose
    assert x > 3.5 and abs(y) < 0.1 and abs(heading) < 1.0
    sim.call(motion(sim, "left", 50, 1.0, 10))
    sim.advance(9.0)
    scale = Settings().max_steering_angle_deg
    # Half the shared scale, measured without the angle offset, as the Bridge reports it.
    assert sim.car.steer_deg - (-1.9) == pytest.approx(scale / 2, abs=1.0)
    assert sim.pose[2] > 5  # turning left
    sim.call({"protocol": PROTOCOL, "request_id": uuid.uuid4().hex, "operation": "stop",
              "boot_id": sim.bridge.controller.boot_id,
              "command_epoch": sim.bridge.controller.command_epoch})  # fmt: skip
    sim.advance(13.0)
    assert sim.car.standstill and math.isclose(sim.car.speed, 0.0)
    assert np.isfinite(sim.status()["speed_mps"])


def test_the_simulator_clips_curvature_with_the_patched_drive_helpers(monkeypatch):
    from drivingbench.sim.car import SimNative

    helpers = sys.modules[DRIVE_HELPERS]
    assert helpers.__file__ == str(PATCHED) and helpers.MAX_LATERAL_ACCEL_NO_ROLL == 12.0
    calls, clip = [], helpers.clip_curvature
    monkeypatch.setattr(helpers, "clip_curvature", lambda *a, **k: calls.append(k) or clip(*a, **k))
    sim = SimNative(tesla_params(), params=P, steer_ratio=12.4, stiffness=1.0)
    assert sim.call(motion(sim, "left", 50, 1.0, 10))["status"] == "accepted"
    sim.advance(1.0)
    assert len(calls) > 50 and all(k["max_curvature"] > 0 for k in calls)


def test_a_stock_drive_helpers_imported_first_is_replaced(monkeypatch):
    use_openpilot()
    stock = logs.VENDOR / "openpilot/selfdrive/controls/lib/drive_helpers.py"
    spec = importlib.util.spec_from_file_location(DRIVE_HELPERS, stock)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "max_curvature" not in inspect.signature(module.clip_curvature).parameters
    monkeypatch.setitem(sys.modules, DRIVE_HELPERS, module)
    use_openpilot()
    from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature

    assert "max_curvature" in inspect.signature(clip_curvature).parameters


def test_an_uninitialized_submodule_says_how_to_check_it_out(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, DRIVE_HELPERS, raising=False)
    monkeypatch.setattr(logs, "VENDOR", tmp_path)
    (tmp_path / "openpilot").mkdir()  # an empty submodule folder, as a plain clone leaves it
    with pytest.raises(ImportError, match="git submodule update --init --depth 1 vendor/openpilot"):
        use_openpilot()
