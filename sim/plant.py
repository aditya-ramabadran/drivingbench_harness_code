"""The Tesla below openpilot's car controller, fitted from its logs (sim/fit.py).

Inputs are what the car receives: the applied steering angle after Tesla's angle limits
(`carOutput.actuatorsOutput.steeringAngleDeg`, raw, angle offset included) and the clipped
acceleration request. The step functions take arrays so the fit and the simulator share them.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PARAMS = Path(__file__).with_name("params.json")


@dataclass(frozen=True)
class Params:
    accel_delay_s: float = 0.2
    accel_tau_s: float = 0.15
    accel_gain: float = 0.7  # steady acceleration per unit request
    accel_bias_mps2: float = 0.0
    launch_s: float = 0.5  # positive request at standstill until the car moves
    launch_accel_mps2: float = 0.5
    steer_delay_s: float = 0.04
    steer_tau_s: float = 0.12
    steer_rate_dps: float = 250.0  # the EPAS's fastest wheel motion
    yaw_scale: float = 1.0  # measured yaw rate per VehicleModel yaw rate

    @classmethod
    def load(cls, path: Path = PARAMS) -> Params:
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path = PARAMS):
        values = {name: round(value, 4) for name, value in asdict(self).items()}
        path.write_text(json.dumps(values, indent=2) + "\n")


def accel_step(p: Params, accel, command, dt):
    steady = p.accel_gain * command + p.accel_bias_mps2
    return accel + (steady - accel) * min(1.0, dt / p.accel_tau_s)


def steer_step(p: Params, angle, target, dt):
    move = (target - angle) * min(1.0, dt / p.steer_tau_s)
    return angle + np.clip(move, -p.steer_rate_dps * dt, p.steer_rate_dps * dt)


class Tesla:
    """Speed, raw steering-wheel angle and planar pose, stepped at controlsd's 100 Hz."""

    def __init__(self, p: Params, curvature, dt=0.01, speed=0.0, steer_deg=0.0, offset_deg=0.0):
        self.p, self.curvature, self.dt, self.offset = p, curvature, dt, offset_deg
        self.speed, self.accel, self.steer_deg, self.steer_rate = speed, 0.0, steer_deg, 0.0
        self.x = self.y = self.heading = 0.0
        self.waiting_s = 0.0
        self.accel_queue = deque([0.0] * round(p.accel_delay_s / dt))
        self.steer_queue = deque([steer_deg] * round(p.steer_delay_s / dt))

    @property
    def standstill(self):
        return self.speed == 0.0

    def step(self, accel_cmd, angle_cmd, lat_active, long_active):
        p, dt = self.p, self.dt
        self.accel_queue.append(accel_cmd if long_active else 0.0)
        delayed = self.accel_queue.popleft()
        if self.standstill:
            self.accel = 0.0
            self.waiting_s = self.waiting_s + dt if long_active and delayed > 0 else 0.0
            if self.waiting_s >= p.launch_s - dt / 2:
                self.speed, self.accel = 1e-3, p.launch_accel_mps2
        else:
            self.accel = float(accel_step(p, self.accel, delayed, dt))
            self.speed = max(0.0, self.speed + self.accel * dt)
            self.waiting_s = 0.0
        # The EPAS follows the applied angle only while openpilot steers; otherwise it holds.
        self.steer_queue.append(angle_cmd if lat_active else self.steer_deg)
        target = self.steer_queue.popleft()
        previous = self.steer_deg
        if lat_active:
            self.steer_deg = float(steer_step(p, self.steer_deg, target, dt))
        self.steer_rate = (self.steer_deg - previous) / dt
        angle = math.radians(self.steer_deg - self.offset)
        yaw_rate = p.yaw_scale * self.speed * self.curvature(angle, self.speed)
        self.heading += yaw_rate * dt
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
