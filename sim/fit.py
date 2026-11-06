"""Fit the Tesla plant (sim/params.json) on one day's drives.

    uv run python -m drivingbench.sim.fit [--series runs/sim/series] [--day YYYY-MM-DD]

The day defaults to the site's `fit_day`, which is not the simulated day: that one is held out
(sim/validate.py). Speed and steering are fitted separately by multi-step simulation error over
3 s windows: speed from the applied acceleration request, the wheel from the applied angle.
Launch timing, the departure lurch and the yaw scale are measured directly. Coordinate search,
no SciPy.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from drivingbench.sim.course import SITE
from drivingbench.sim.logs import car_params, use_openpilot
from drivingbench.sim.plant import Params, accel_step, steer_step
from drivingbench.sim.series import SERIES, day, load

DRIVE = 2  # car.capnp GearShifter
DT, HORIZON, STRIDE, HISTORY = 0.01, 300, 50, 100


def vehicle_model(path: Path, c: dict[str, np.ndarray]):
    """openpilot's VehicleModel for a saved route: its CarParams, the learned stiffness and steer
    ratio (median)."""
    use_openpilot()
    from opendbc.car.vehicle_model import VehicleModel

    vm = VehicleModel(car_params(np.load(path)["car_params"].tobytes()))
    vm.update_params(float(np.nanmedian(c["stiffness"])), float(np.nanmedian(c["steer_ratio"])))
    return vm


def route(path: Path) -> dict[str, np.ndarray]:
    """Columns plus `vm_yaw`, the VehicleModel yaw rate of the measured wheel angle."""
    _, _, c = load(path)
    vm = vehicle_model(path, c)
    angle = np.radians(c["steer_deg"] - np.nan_to_num(c["angle_offset"]))
    roll = np.nan_to_num(c["roll"])
    c["vm_yaw"] = np.array([vm.calc_curvature(a, v, r) for a, v, r in zip(angle, c["v"], roll)])
    c["vm_yaw"] *= c["v"]
    return c


def starts(mask: np.ndarray) -> np.ndarray:
    """Window starts whose history and horizon lie wholly inside `mask`."""
    run = np.convolve(mask.astype(int), np.ones(HISTORY + HORIZON, int), "valid")
    return np.where(run == HISTORY + HORIZON)[0][::STRIDE] + HISTORY


def driving(c):
    return (c["brake"] == 0) & (c["gas"] == 0) & (c["gear"] == DRIVE) & (c["steer_pressed"] == 0)


def speed_error(p: Params, sets) -> float:
    errors, delay = [], round(p.accel_delay_s / DT)
    for c, s in sets:
        idx = s[:, None] + np.arange(HORIZON)[None]
        command, truth = c["accel_out"][idx - delay], c["v"][idx]
        v, a = truth[:, 0].copy(), c["a"][s].copy()
        for k in range(1, HORIZON):
            a = accel_step(p, a, command[:, k - 1], DT)
            v = np.maximum(0.0, v + a * DT)
            errors.append(np.square(v - truth[:, k]))
    return float(np.sqrt(np.mean(np.concatenate(errors))))


def steer_error(p: Params, sets) -> float:
    errors, delay = [], round(p.steer_delay_s / DT)
    for c, s in sets:
        idx = s[:, None] + np.arange(HORIZON)[None]
        target, truth = c["steer_out"][idx - delay], c["steer_deg"][idx]
        angle = truth[:, 0].copy()
        for k in range(1, HORIZON):
            angle = steer_step(p, angle, target[:, k - 1], DT)
            errors.append(np.square(angle - truth[:, k]))
    return float(np.sqrt(np.mean(np.concatenate(errors))))


def search(p: Params, error, grid: dict[str, list[float]], rounds=3) -> Params:
    """Coordinate search over each field's candidates, repeated until nothing improves."""
    best = error(p)
    for _ in range(rounds):
        improved = False
        for name, values in grid.items():
            for value in values:
                trial = replace(p, **{name: value})
                if (e := error(trial)) < best - 1e-6:
                    p, best, improved = trial, e, True
        if not improved:
            break
    return p


def launches(sets_all) -> tuple[float, float]:
    """Median wait from a positive request at standstill to motion, and the first 0.3 s accel."""
    waits, lurches = [], []
    for c in sets_all:
        t, v = c["t"], c["v"]
        asked = (c["long_active"] == 1) & (c["accel_out"] > 0.05) & (v < 0.01) & driving(c)
        for i in np.flatnonzero(asked[1:] & ~asked[:-1]) + 1:
            j = i
            while j < len(t) - 30 and v[j] < 0.01 and asked[j]:
                j += 1
            if v[j] >= 0.01 and 0 < t[j] - t[i] < 5:
                waits.append(t[j] - t[i])
                lurches.append(float(np.mean(c["a"][j : j + 30])))
    return float(np.median(waits)), float(np.median(lurches))


def yaw_scale(sets_all) -> float:
    """Gyro yaw rate (rest bias removed) per VehicleModel yaw rate while moving."""
    xs, ys = [], []
    for c in sets_all:
        rest = (c["v"] < 0.01) & np.isfinite(c["gyro_x"])
        moving = (c["v"] > 0.5) & np.isfinite(c["gyro_x"]) & np.isfinite(c["vm_yaw"]) & driving(c)
        xs.append(c["vm_yaw"][moving])
        ys.append(c["gyro_x"][moving] - np.median(c["gyro_x"][rest]))
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float(np.dot(x, y) / np.dot(x, x))


def fit(paths: list[Path]) -> tuple[Params, dict]:
    columns = [route(p) for p in paths]
    longitudinal = [
        (c, starts((c["long_active"] == 1) & driving(c) & (c["v"] > 0.05))) for c in columns
    ]
    lateral = [
        (c, starts((c["lat_active"] == 1) & driving(c) & np.isfinite(c["steer_out"])))
        for c in columns
    ]
    p = Params()
    p = search(p, lambda q: speed_error(q, longitudinal), {
        "accel_delay_s": [0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3],
        "accel_tau_s": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6],
        "accel_gain": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        "accel_bias_mps2": [-0.08, -0.05, -0.02, 0.0, 0.02, 0.05, 0.08],
    })  # fmt: skip
    p = search(p, lambda q: steer_error(q, lateral), {
        "steer_delay_s": [0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
        "steer_tau_s": [0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2],
        "steer_rate_dps": [150, 200, 250, 300, 400, 600],
    })  # fmt: skip
    wait, lurch = launches(columns)
    p = replace(p, launch_s=wait, launch_accel_mps2=lurch, yaw_scale=yaw_scale(columns))
    ideal = replace(p, accel_delay_s=0.0, accel_tau_s=DT, accel_gain=1.0, accel_bias_mps2=0.0,
                    steer_delay_s=0.0, steer_tau_s=DT, steer_rate_dps=1e6)  # fmt: skip
    report = {
        "speed_rmse_mps": {"fit": speed_error(p, longitudinal),
                           "ideal": speed_error(ideal, longitudinal)},
        "steer_rmse_deg": {"fit": steer_error(p, lateral), "ideal": steer_error(ideal, lateral)},
        "windows": {"speed": sum(len(s) for _, s in longitudinal),
                    "steer": sum(len(s) for _, s in lateral)},
    }  # fmt: skip
    return p, report


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--series", type=Path, default=SERIES)
    a.add_argument("--day", default=SITE["fit_day"])
    args = a.parse_args()
    paths = [p for p in sorted(args.series.glob("*.npz")) if day(p) == args.day]
    if not paths:
        raise SystemExit(f"no series in {args.series} from {args.day}")
    p, report = fit(paths)
    p.save()
    print(p)
    print(report)


if __name__ == "__main__":
    main()
