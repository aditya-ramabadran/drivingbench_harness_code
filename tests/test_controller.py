import math

import pytest
from drivingbench.controller.core import Controller, Sample
from drivingbench.controller.speed import SpeedControl
from drivingbench.shared.contracts import (
    DURATION_RANGE_S,
    EMERGENCY_SPEED_MPS,
    SPEED_RANGE_MPS,
    STEERING_SCALE_DEG,
    Motion,
    Request,
    Settings,
    SpeedSettings,
)


def send(c, op, now=0, *, identifier=None, **fields):
    return c.handle(
        Request(
            request_id=identifier or f"{op}-{now}",
            operation=op,
            boot_id=c.boot_id,
            command_epoch=c.command_epoch,
            **fields,
        ).model_dump(),
        now,
    )


FULL, HALF = STEERING_SCALE_DEG, STEERING_SCALE_DEG / 2


def ready():
    c = Controller()
    c.handle({"protocol": 2, "operation": "heartbeat"}, 0)
    c.refresh(
        Sample(
            gear="drive",
            enabled=True,
            sensors_valid=True,
            car_state_valid=True,
            lateral_active=True,
            longitudinal_active=True,
            speed_mps=0.6,
        ),
        0,
    )
    return c


def motion(c, direction="left", percent=50, speed=0.6, duration=5, now=0, **kwargs):
    return send(
        c,
        "motion",
        now,
        motion=Motion(
            direction=direction,
            steering_percent=percent,
            speed_mps=speed,
            duration_s=duration,
        ),
        settings=kwargs.pop("settings", Settings()),
        **kwargs,
    )


@pytest.mark.parametrize(
    "direction,percent,target",
    [
        ("left", 0, 0),
        ("right", 0, 0),
        ("straight", 100, 0),
        ("left", 50, HALF),
        ("right", 50, -HALF),
        ("left", 100, FULL),
        ("right", 100, -FULL),
    ],
)
def test_mapping(direction, percent, target):
    c = ready()
    motion(c, direction, percent)
    assert c.command["target_angle_deg"] == target


def test_target_is_immediate_and_native_feedback_does_not_ramp_it():
    c = ready()
    motion(c)
    assert c.step(0).angle_deg == HALF
    c.applied_angle(0.1)
    assert c.step(0.01).angle_deg == HALF
    motion(c, "right", 100, now=0.01)
    assert c.step(0.02).angle_deg == -FULL


def test_retained_wheels_and_inactive_lateral_do_not_accumulate():
    c = ready()
    c.sample.steering_deg, c.sample.lateral_active = 110, False
    motion(c, "right")
    for t in (0, 0.5, 1):
        assert c.step(t).angle_deg == 110
    c.sample.steering_deg, c.sample.lateral_active = 100, True
    assert c.step(1.01).angle_deg == -HALF


def test_target_does_not_depend_on_elapsed_ticks():
    c = ready()
    motion(c)
    first = c.step(0).angle_deg
    assert c.step(0.5).angle_deg == first == HALF


def test_snapshot_settings_speed_rejection_and_stop_continuation():
    c = ready()
    assert motion(c, speed=3)["status"] == "accepted"
    prior = c.command.copy()
    ceiling = Settings(speed_limit_mps=2)
    assert motion(c, speed=2.5, now=0.01, settings=ceiling)["reason"] == "speed_limit_exceeded"
    assert c.command == prior
    assert c.command["target_angle_deg"] == HALF
    changed = Settings(max_steering_angle_deg=400)
    send(c, "stop", 0.1)
    assert c.step(0.1).should_stop
    motion(c, now=0.2, settings=changed)
    assert c.command["target_angle_deg"] == 200


def test_duplicate_does_not_extend_and_reused_id_rejects():
    c = ready()
    motion(c, identifier="abc")
    original = c.command.copy()
    motion(c, identifier="abc", now=1)
    assert c.command == original
    assert motion(c, percent=20, identifier="abc", now=1)["reason"] == "request_id_reused"
    assert c.command == original


def test_stop_fences_delayed_command():
    c = ready()
    epoch = c.command_epoch
    send(c, "stop", 0.1)
    result = c.handle(
        dict(
            protocol=2,
            operation="motion",
            boot_id=c.boot_id,
            command_epoch=epoch,
            request_id="delayed",
            motion=dict(direction="left", steering_percent=50, speed_mps=1, duration_s=5),
            settings=Settings().model_dump(),
        ),
        0.2,
    )
    assert result["reason"] == "command_epoch_changed"
    assert c.command is None


def test_expiry_and_explicit_stop_retain_wheel():
    c = ready()
    c.sample.steering_deg = 120
    assert motion(c, duration=5)["status"] == "accepted"
    c.step(0)
    assert c.step(5).angle_deg == 120
    assert c.command is None and c.step(5).should_stop


def test_first_motion_snapshots_settings_without_arming():
    c = ready()
    settings = Settings(speed_limit_mps=2, speed_control=SpeedSettings(gain=0.8))
    assert motion(c, settings=settings)["status"] == "accepted"
    assert c.settings.speed_limit_mps == 2 and c.speed.settings.gain == 0.8
    settings.speed_limit_mps = 3
    assert c.settings.speed_limit_mps == 2
    c.sample.speed_mps, c.sample.standstill = EMERGENCY_SPEED_MPS + 0.6, False
    c.refresh(c.sample, 0.1)
    assert c.reason == "emergency_speed_exceeded" and c.command is None


def test_hard_limits_ceiling_and_fixed_emergency_are_three_different_things():
    c = ready()
    low = Settings(speed_limit_mps=1)
    assert motion(c, speed=1, settings=low)["status"] == "accepted"
    # Overshoot far above the ceiling (2.27 m/s on 1.6 was recorded) keeps the command.
    c.sample.speed_mps = EMERGENCY_SPEED_MPS - 0.1
    c.refresh(c.sample, 0.5)
    c.refresh(c.sample, 1.5)
    assert c.command is not None and c.reason is None
    c.sample.speed_mps = EMERGENCY_SPEED_MPS + 0.51
    c.refresh(c.sample, 1.6)
    assert c.command is None and c.reason == "emergency_speed_exceeded"
    assert motion(c, now=1.7)["reason"] == "emergency_speed_exceeded"  # refused until slower
    low_s, high_s = SPEED_RANGE_MPS
    low_t, high_t = DURATION_RANGE_S
    valid = dict(direction="left", steering_percent=0, speed_mps=1, duration_s=10)
    assert Motion(**valid)
    for field, bad in (
        ("speed_mps", low_s - 0.01),
        ("speed_mps", high_s + 0.01),
        ("duration_s", low_t - 0.01),
        ("duration_s", high_t + 0.01),
    ):
        with pytest.raises(ValueError):
            Motion(**{**valid, field: bad})
    for bad in (low_s - 0.01, high_s + 0.01, EMERGENCY_SPEED_MPS):
        with pytest.raises(ValueError):
            Settings(speed_limit_mps=bad)
    assert low_s <= Settings().speed_limit_mps == 3 <= high_s < EMERGENCY_SPEED_MPS


def test_stale_zero_speed_does_not_claim_hold_or_preserve_command():
    c = ready()
    motion(c)
    stale = Sample(
        gear="drive",
        enabled=True,
        standstill=True,
        speed_mps=0,
        car_state_valid=False,
        sensors_valid=False,
    )
    c.refresh(stale, 0.1)
    assert not stale.stationary_verified and not stale.held
    assert c.command is None
    assert c.reason == "motion_inputs_unavailable"


def test_stationary_sensor_gap_drops_command_without_replay():
    c = ready()
    motion(c)
    s = Sample(
        gear="drive", enabled=True, standstill=True, sensors_valid=False, car_state_valid=True
    )
    c.refresh(s, 0.1)
    assert c.command is None
    s.sensors_valid = True
    c.refresh(s, 0.2)
    assert c.step(0.2).should_stop
    assert motion(c, now=0.3)["status"] == "accepted"


@pytest.mark.parametrize(
    "changed,reason",
    [
        ({"speed_mps": math.nan, "sensors_valid": False}, "motion_inputs_unavailable"),
        ({"gear": "park"}, "gear_not_drive"),
        ({"brake_pressed": True}, "operator_brake"),
        ({"gas_pressed": True}, "operator_intervention"),
        ({"enabled": False}, "native_disengaged"),
    ],
)
def test_native_intervention_cancels(changed, reason):
    c = ready()
    motion(c)
    for k, v in changed.items():
        setattr(c.sample, k, v)
    c.refresh(c.sample, 0.1)
    assert c.command is None and c.reason == reason


def test_watchdog_is_local_not_session_lifetime():
    c = ready()
    for t in (1, 2, 3, 100):
        c.handle({"protocol": 2, "operation": "heartbeat"}, t)
        c.refresh(c.sample, t)
        assert c.reason is None
    c.refresh(c.sample, 103)
    assert c.reason == "producer_unavailable"


def test_malformed_input_preserves_previous_command():
    c = ready()
    motion(c)
    old = c.command.copy()
    bad = Request(
        request_id="bad", operation="motion", boot_id=c.boot_id, command_epoch=c.command_epoch
    ).model_dump()
    bad["motion"] = dict(direction="left", steering_percent=math.nan, speed_mps=1, duration_s=5)
    assert c.handle(bad, 1)["reason"] == "malformed_request"
    assert c.command == old


def test_status_names_the_missing_native_authority():
    c = ready()
    motion(c)
    healthy = c.status(0)
    assert healthy["reason"] is None
    c.sample.longitudinal_active = False
    assert c.status(0)["reason"] == "longitudinal_unavailable"
    c.sample.longitudinal_active, c.sample.lateral_active = True, False
    assert c.status(0)["reason"] == "steering_unavailable"
    c.sample.standstill = True
    assert c.status(0)["reason"] == "waiting_for_res"
    assert set(c.status(0)) == set(healthy)  # response shape unchanged


def test_emergency_speed_allows_noise_and_transient_but_stops_real_excursion():
    c = ready()
    c.sample.speed_mps = EMERGENCY_SPEED_MPS + 0.04
    c.refresh(c.sample, 0.1)
    assert c.reason is None
    c.sample.speed_mps = EMERGENCY_SPEED_MPS + 0.1
    c.refresh(c.sample, 0.2)
    c.refresh(c.sample, 0.6)
    assert c.reason is None
    c.refresh(c.sample, 0.71)
    assert c.reason == "emergency_speed_exceeded"
    c = ready()
    c.sample.speed_mps = EMERGENCY_SPEED_MPS + 0.51
    c.refresh(c.sample, 0.1)
    assert c.reason == "emergency_speed_exceeded"


def test_speed_feedback_targets_exact_speed_and_acceleration_cap():
    speed = SpeedControl(SpeedSettings())
    for i in range(200):
        acceleration = speed.step(5, 0, 0.01)
        assert 0 < acceleration <= 1
    assert acceleration == 1
    speed.reset()
    for i in range(200):
        acceleration = speed.step(1, 1, 0.01)
    assert acceleration == 0  # no hidden reserve causing deceleration at target
    assert speed.step(1, 1.1, 0.01) < 0


def test_native_engagement_recovery_accepts_fresh_command():
    c = ready()
    c.sample.enabled = False
    c.sample.standstill = True
    c.sample.speed_mps = 0
    c.refresh(c.sample, 0.1)
    assert c.reason == "native_disengaged"
    assert motion(c, now=0.2)["reason"] == "native_disengaged"
    c.sample.enabled = True
    c.refresh(c.sample, 0.3)
    assert motion(c, now=0.4)["status"] == "accepted"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"gear": "park"}, "gear_not_drive"),
        ({"brake_pressed": True}, "operator_brake"),
        ({"gas_pressed": True}, "operator_intervention"),
        ({"sensors_valid": False}, "motion_inputs_unavailable"),
        ({"enabled": False}, "native_disengaged"),
    ],
)
def test_intervention_rejects_motion_then_recovers_without_arming(change, reason):
    c = ready()
    motion(c)
    old = {key: getattr(c.sample, key) for key in change}
    for key, value in change.items():
        setattr(c.sample, key, value)
    c.refresh(c.sample, 0.1)
    assert c.command is None and c.step(0.1).should_stop
    assert motion(c, now=0.2)["reason"] == reason
    for key, value in old.items():
        setattr(c.sample, key, value)
    c.refresh(c.sample, 0.3)
    assert c.command is None and c.step(0.3).should_stop
    assert motion(c, now=0.4)["status"] == "accepted"
    assert not c.step(0.4).should_stop


def test_a_jolted_wheel_keeps_the_command_and_a_held_wheel_takes_over():
    c = ready()
    motion(c)
    c.sample.steering_pressed = True  # a full-lock stop jolts the column for about 50 ms
    c.refresh(c.sample, 0.10)
    c.refresh(c.sample, 0.15)
    c.sample.steering_pressed = False
    c.refresh(c.sample, 0.16)
    assert c.command is not None and not c.step(0.16).should_stop
    c.sample.steering_pressed = True
    for t in (0.2, 0.3, 0.44):
        c.refresh(c.sample, t)
    assert c.command is not None  # 0.24 s held: not yet the driver's
    c.refresh(c.sample, 0.45)
    assert c.command is None and c.reason == "operator_intervention" and c.step(0.45).should_stop
    assert motion(c, now=0.5)["reason"] == "operator_intervention"
    c.sample.steering_pressed = False
    c.refresh(c.sample, 0.6)
    assert motion(c, now=0.7)["status"] == "accepted"


def test_heartbeat_recovery_never_replays_cancelled_command():
    c = ready()
    motion(c)
    c.refresh(c.sample, 3)
    assert motion(c, now=3)["reason"] == "producer_unavailable"
    c.handle({"protocol": 2, "operation": "heartbeat"}, 3.1)
    c.refresh(c.sample, 3.1)
    assert c.command is None
    assert motion(c, now=3.2)["status"] == "accepted"


def test_stop_ignores_stale_command_epoch():
    c = ready()
    old_epoch = c.command_epoch
    send(c, "stop")
    motion(c, now=0.1)
    reply = c.handle(
        Request(
            request_id="late-stop", operation="stop", boot_id=c.boot_id, command_epoch=old_epoch
        ).model_dump(),
        0.2,
    )
    assert reply["status"] == "accepted" and c.command is None


def chunk(c, curvature, speed, now=0, **kwargs):
    from drivingbench.shared.contracts import Chunk

    settings = kwargs.pop("settings", Settings())
    return send(
        c, "chunk", now, chunk=Chunk(curvature_inv_m=curvature, speed_mps=speed), settings=settings
    )


def plan(c, rows, anchor, now=0, settings=None):
    from drivingbench.shared.contracts import Chunk

    body = Chunk(
        curvature_inv_m=[k for k, _ in rows], speed_mps=[v for _, v in rows], anchor_s=anchor
    )
    return send(c, "chunk", now, chunk=body, settings=settings or Settings())


def test_an_anchored_plan_runs_the_row_due_each_step_on_the_comma_clock_and_holds_the_last():
    c = ready()
    rows = [(0.001 * k, 1.0) for k in range(48)]
    # Row 0 was due 0.9 s before acceptance (capture age less the actuator lead); each 0.1 s
    # step runs row min(round((t - anchor) / 0.1), 47).
    assert plan(c, rows, anchor=0.1, now=1.0)["status"] == "accepted"
    seen = [c.step(now).curvature_inv_m for now in (1.0, 1.05, 1.1, 1.42, 1.65)]
    assert seen == pytest.approx([0.009, 0.009, 0.010, 0.013, 0.015])  # held within a step
    # Late transit or a slow acknowledgement cannot shift rows: the anchor times them.
    late = ready()
    plan(late, rows, anchor=0.1, now=1.3)
    assert late.step(1.3).curvature_inv_m == pytest.approx(0.012)
    held = ready()
    plan(held, rows, anchor=-4.0, now=1.0)  # past the end: hold the final row
    assert held.step(1.0).curvature_inv_m == pytest.approx(0.047)
    early = ready()
    plan(early, rows, anchor=1.5, now=1.0)  # not due yet: the first row, never a negative index
    assert early.step(1.0).curvature_inv_m == 0.0


def test_an_anchored_plan_runs_to_its_end_and_at_least_its_lease_and_keeps_every_limit():
    from drivingbench.shared.contracts import ANCHORED_LEASE_S

    c = ready()
    rows = [(0.01, 1.0)] * 16  # 1.6 s of plan whose capture was 1 s ago: 0.4 s of it remain
    plan(c, rows, anchor=-0.5, now=0.5)
    assert c.command["expires_at_s"] == pytest.approx(0.5 + ANCHORED_LEASE_S)
    assert not c.step(1.29).should_stop
    s = c.step(1.31)  # the client stopped renewing: brake exactly like any expiry
    assert c.command is None and s.should_stop
    long = ready()  # a 5 s plan runs to its end without renewal
    plan(long, [(0.0, 1.0)] * 50, anchor=0.3, now=0.5)
    assert long.command["expires_at_s"] == pytest.approx(5.3)
    assert not long.step(5.25).should_stop and long.step(5.35).should_stop
    future = ready()  # an anchor in the future never outlasts the longest chunk (6 s)
    plan(future, [(0.0, 1.0)] * 60, anchor=100.0, now=0.5)
    assert future.command["expires_at_s"] == pytest.approx(6.5)
    assert not future.step(6.45).should_stop and future.step(6.55).should_stop
    capped = Settings(speed_limit_mps=1.5)
    fast = [(0.0, 1.0)] * 47 + [(0.0, 2.0)]  # one late row over the ceiling rejects the plan
    assert plan(c, fast, anchor=1.4, now=1.4, settings=capped)["reason"] == "speed_limit_exceeded"


def test_a_chunk_steps_through_its_setpoints_on_the_comma_clock_then_brakes():
    c = ready()
    assert chunk(c, [0.05, -0.1, 0.0], [1.0, 2.0, 1.5])["status"] == "accepted"
    targets = []
    for now in (0.01, 0.09, 0.11, 0.25, 0.29):
        s = c.step(now)
        targets.append((s.curvature_inv_m, c.command["speed_mps"]))
        assert not s.should_stop and s.resume
    assert targets == [(0.05, 1.0), (0.05, 1.0), (-0.1, 2.0), (0.0, 1.5), (0.0, 1.5)]
    assert c.status(0.25)["remaining_s"] == pytest.approx(0.05)
    s = c.step(0.31)  # past the last setpoint: the same braking as a motion's expiry
    assert c.command is None and s.should_stop and s.curvature_inv_m is None


def test_a_zero_speed_setpoint_brakes_and_holds_without_ending_the_chunk():
    c = ready()
    chunk(c, [0.0, 0.1, 0.1], [1.0, 0.0, 1.0])
    assert c.step(0.05).acceleration_mps2 != 0
    held = c.step(0.15)
    assert held.should_stop and held.acceleration_mps2 == 0 and held.curvature_inv_m == 0.1
    assert c.command is not None
    assert not c.step(0.25).should_stop


def test_chunks_obey_the_same_ceiling_epoch_and_readiness_rules_as_motion():
    c = ready()
    capped = Settings(speed_limit_mps=1.5)
    assert chunk(c, [0, 0], [1.0, 2.0], settings=capped)["reason"] == "speed_limit_exceeded"
    stale = Request(
        request_id="stale",
        operation="chunk",
        boot_id=c.boot_id,
        command_epoch=c.command_epoch - 1,
        chunk={"curvature_inv_m": [0], "speed_mps": [1]},
        settings=Settings(),
    )
    assert c.handle(stale.model_dump(), 0)["reason"] == "command_epoch_changed"
    c.sample.brake_pressed = True
    assert chunk(c, [0], [1.0], now=0.5)["reason"] == "operator_brake"
    c.sample.brake_pressed = False
    assert chunk(c, [0.1] * 5, [1.0] * 5, now=1)["status"] == "accepted"
    assert motion(c, now=1.1)["status"] == "accepted"  # the latest command replaces a chunk
    assert "setpoints" not in c.command and c.step(1.2).curvature_inv_m is None
    assert chunk(c, [0.1] * 5, [1.0] * 5, now=1.3)["status"] == "accepted"
    assert send(c, "stop", 1.31)["status"] == "accepted" and c.command is None


@pytest.mark.parametrize(
    "fields",
    [
        {"curvature_inv_m": [0, 0], "speed_mps": [1]},  # one value per step
        {"curvature_inv_m": [0], "speed_mps": [0.3]},  # between brake and the motion range
        {"curvature_inv_m": [1.5], "speed_mps": [1]},
        {"curvature_inv_m": [0], "speed_mps": [9]},
        {"curvature_inv_m": [0] * 61, "speed_mps": [1] * 61},
        {"curvature_inv_m": [], "speed_mps": []},
        {"curvature_inv_m": [float("nan")], "speed_mps": [1]},
        {"curvature_inv_m": [0], "speed_mps": [1], "surprise": 1},
    ],
)
def test_malformed_chunks_are_rejected_before_anything_moves(fields):
    c = ready()
    message = {
        "protocol": 2,
        "request_id": "bad",
        "operation": "chunk",
        "boot_id": c.boot_id,
        "command_epoch": c.command_epoch,
        "chunk": fields,
        "settings": Settings().model_dump(),
    }
    assert c.handle(message, 0)["reason"] == "malformed_request"
    assert c.command is None
