"""DrivingBench's native stack driving the fitted Tesla, tick for tick.

DrivingBench's `Bridge` (its curvature clipped by the patched drive_helpers, sim/logs.py) and
`Controller`, openpilot's `LongControl` and `LatControlAngle`, and the Tesla car controller's angle
and acceleration limits run as `controlsd.state_control` and `card` run them on the comma; only the
car below them is a model (sim/plant.py). `SimNative` has the producer's native interface (`call`,
`status`).
"""

from __future__ import annotations

import math
from types import SimpleNamespace as NS

from drivingbench.device.native import Bridge
from drivingbench.shared.contracts import PROTOCOL
from drivingbench.sim.logs import use_openpilot
from drivingbench.sim.plant import Params, Tesla

use_openpilot()
from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN  # noqa: E402
from opendbc.car.lateral import apply_steer_angle_limits_vm  # noqa: E402
from opendbc.car.tesla.carcontroller import get_safety_CP  # noqa: E402
from opendbc.car.tesla.values import CarControllerParams  # noqa: E402
from opendbc.car.vehicle_model import VehicleModel  # noqa: E402
from openpilot.selfdrive.controls.lib.latcontrol_angle import (  # noqa: E402
    STEER_ANGLE_SATURATION_THRESHOLD,
    LatControlAngle,
)
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState  # noqa: E402

DT = 0.01
HEARTBEAT_S = 0.05  # NativeClient's poll interval
STEER_STEP, ACCEL_STEP = 2, 4  # Tesla sends the angle at 50 Hz and acceleration at 25 Hz


class Inbox:
    """Bridge's server: requests queued by `call`, answered on the next native tick."""

    def __init__(self):
        self.pending = []

    def poll(self, handler):
        for message, replies in self.pending:
            replies.append(handler(message))
        self.pending.clear()


class Bus(dict):
    """The SubMaster subset the Bridge reads; every simulated input is valid."""

    def all_checks(self, _names=None):
        return True


class SimNative:
    def __init__(self, cp, *, params=None, steer_ratio, stiffness, angle_offset=0.0, roll=0.0,
                 lat_delay=0.3, time=0.0, speed=0.0, steer_deg=0.0):  # fmt: skip
        self.time = time
        self.inbox = Inbox()
        self.bridge = Bridge(self.inbox, clock=lambda: self.time, vehicle=str(cp.carFingerprint),
                             wheelbase=float(cp.wheelbase))  # fmt: skip
        self.vm = VehicleModel(cp)
        self.vm.update_params(stiffness, steer_ratio)
        self.safety_vm = VehicleModel(get_safety_CP())
        self.lp = NS(angleOffsetDeg=angle_offset, roll=roll, steerRatio=steer_ratio,
                     stiffnessFactor=stiffness)  # fmt: skip
        self.lat_delay = lat_delay
        self.long = LongControl(cp)
        self.lat = LatControlAngle(cp, None, DT)
        self.car = Tesla(params or Params.load(), lambda sa, u: self.vm.calc_curvature(sa, u, roll),
                         DT, speed=speed, steer_deg=steer_deg, offset_deg=angle_offset)  # fmt: skip
        self.enabled, self.brake = True, False
        self.desired_curvature = 0.0
        self.applied_angle, self.applied_accel = steer_deg, 0.0
        self.steer_limited_by_safety, self.frame = False, 0
        self.snapshot, self.next_heartbeat = {}, time
        self.tick()  # the native loop is always running before the producer calls it

    def call(self, body, **_):
        replies = []
        self.inbox.pending.append((body, replies))
        self.tick()
        return replies[0]

    def status(self):
        return dict(self.snapshot)

    def advance(self, until):
        while self.time + DT / 2 < until:
            self.tick()

    def tick(self):
        replies = None
        if self.time >= self.next_heartbeat:
            self.next_heartbeat += HEARTBEAT_S
            replies = []
            self.inbox.pending.insert(
                0, ({"protocol": PROTOCOL, "operation": "heartbeat"}, replies)
            )
        car = self.car
        cs = NS(
            vEgo=car.speed, vEgoRaw=car.speed, aEgo=car.accel, steeringAngleDeg=car.steer_deg,
            steeringRateDeg=car.steer_rate, standstill=car.standstill, gearShifter="drive",
            brakePressed=self.brake, gasPressed=False, steeringPressed=False, canValid=True,
            cruiseState=NS(standstill=False),
        )  # fmt: skip
        sm = Bus(
            carState=cs,
            vehicleParameters=self.lp,
            selfdriveState=NS(enabled=self.enabled, state="enabled"),
        )
        # controlsd.state_control with the planner replaced by the Bridge. The Tesla
        # steers at standstill (CarParams.steerAtStandstill), so only enablement gates it.
        lat_active = long_active = self.enabled
        selection = self.bridge.tick(sm, lateral_active=lat_active, longitudinal_active=long_active)
        lat_active = lat_active and self.bridge.controller.sample.sensors_valid
        if not lat_active:
            self.lat.reset()
        if not long_active:
            self.long.reset()
        if selection.resume and long_active and not cs.brakePressed:
            self.long.long_control_state = LongCtrlState.pid
        if not selection.should_stop and cs.standstill:
            self.long.pid.reset()
        accel = float(
            self.long.update(long_active, cs, selection.acceleration_mps2, selection.should_stop,
                             (ACCEL_MIN, ACCEL_MAX))
        )  # fmt: skip
        self.desired_curvature, limited = self.bridge.apply_curvature(
            self.vm, cs.vEgo, self.lp.roll, self.desired_curvature
        )
        _, angle, _ = self.lat.update(lat_active, cs, self.vm, self.lp, self.steer_limited_by_safety,
                                      self.desired_curvature, limited, self.lat_delay)  # fmt: skip
        # Tesla CarController: the angle limits at 50 Hz, the clipped request at 25 Hz.
        if self.frame % STEER_STEP == 0:
            self.applied_angle = apply_steer_angle_limits_vm(
                angle, self.applied_angle, cs.vEgoRaw, cs.steeringAngleDeg, lat_active,
                CarControllerParams, self.safety_vm,
            )  # fmt: skip
        if self.frame % ACCEL_STEP == 0:
            limits = CarControllerParams
            self.applied_accel = min(limits.ACCEL_MAX, max(limits.ACCEL_MIN, accel))
        self.steer_limited_by_safety = (
            abs(angle - self.applied_angle) > STEER_ANGLE_SATURATION_THRESHOLD
        )
        car.step(self.applied_accel, self.applied_angle, lat_active, long_active)
        self.frame += 1
        self.time += DT
        if replies:
            self.snapshot = dict(replies[0], received_at=self.time)

    @property
    def pose(self):
        return self.car.x, self.car.y, math.degrees(self.car.heading)
