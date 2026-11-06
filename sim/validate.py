"""Score the fitted Tesla on held-out drives: the car's own applied requests, replayed.

    uv run python -m drivingbench.sim.validate [--series runs/sim/series] [--routes ID ...]
                                               [--out runs/sim/validate]

By default every series route not driven on the site's `fit_day` is held out. Every accepted
command (`set_motion` or `chunk`) opens a window. The plant starts from the logged speed and wheel
angle and receives the logged applied acceleration and angle at their logged times; the window
ends after 30 s or at the first operator input, disengagement or gear change. Replaying the
requests the car received, not the commands, isolates the car model from the controller that
issued them. Truth is the logged speed and wheel angle, gyro yaw (rest bias removed), and the
logged speed dead-reckoned along it. The baseline is an ideal car that follows each request at
once.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from drivingbench.sim.course import SITE
from drivingbench.sim.fit import driving, route, vehicle_model
from drivingbench.sim.logs import REPO
from drivingbench.sim.plant import Params, Tesla
from drivingbench.sim.series import SERIES, day, load

DT, HORIZON_S, CHECKPOINTS = 0.01, 30.0, (5.0, 10.0, 30.0)


def ideal(p: Params) -> Params:
    return replace(p, accel_delay_s=0.0, accel_tau_s=DT, accel_gain=1.0, accel_bias_mps2=0.0,
                   steer_delay_s=0.0, steer_tau_s=DT, steer_rate_dps=1e9, yaw_scale=1.0,
                   launch_s=0.0)  # fmt: skip


def windows(c, commands):
    """(start row, end row) per accepted command."""
    t = c["t"]
    stop = ~driving(c) | (c["enabled"] != 1)  # driving: Drive, no pedal or steering input
    for cmd in commands:
        if cmd["status"] != "accepted" or cmd["tool"] == "stop_now":
            continue
        k = int(np.searchsorted(t, cmd["t"]))
        if k < 100 or k >= len(t) - 100 or stop[k]:
            continue
        end = min(int(np.searchsorted(t, cmd["t"] + HORIZON_S)), len(t) - 1)
        if hits := np.flatnonzero(stop[k:end]).tolist():
            end = k + hits[0]
        if end - k > 100:
            yield k, end


def simulate(p: Params, c, k, end, vm, roll, offset):
    car = Tesla(p, lambda a, u: vm.calc_curvature(a, u, roll), DT, speed=float(c["v"][k]),
                steer_deg=float(c["steer_deg"][k]), offset_deg=offset)  # fmt: skip
    car.accel = float(c["a"][k])
    history = np.nan_to_num(c["accel_out"][k - len(car.accel_queue) : k])
    car.accel_queue.clear()
    car.accel_queue.extend(history.tolist())
    out = np.zeros((end - k, 5))
    for n, j in enumerate(range(k, end)):
        ticks = 1 if n == 0 else max(1, round((c["t"][j] - c["t"][j - 1]) / DT))
        for _ in range(ticks):
            car.step(float(np.nan_to_num(c["accel_out"][j])), float(c["steer_out"][j]),
                     bool(c["lat_active"][j]), bool(c["long_active"][j]))  # fmt: skip
        out[n] = car.speed, car.steer_deg, car.heading, car.x, car.y
    return out


def observed(c, k, end, bias):
    dt = np.diff(c["t"][k:end], prepend=c["t"][k])
    speed = c["v"][k:end]
    heading = np.cumsum(np.nan_to_num(c["gyro_x"][k:end] - bias) * dt)
    x, y = np.cumsum(speed * np.cos(heading) * dt), np.cumsum(speed * np.sin(heading) * dt)
    return np.stack([speed, c["steer_deg"][k:end], heading, x, y], axis=1)


def score(model, real, t):
    e = {
        "speed_mps": np.abs(model[:, 0] - real[:, 0]),
        "angle_deg": np.abs(model[:, 1] - real[:, 1]),
        "heading_deg": np.degrees(np.abs(model[:, 2] - real[:, 2])),
        "position_m": np.hypot(model[:, 3] - real[:, 3], model[:, 4] - real[:, 4]),
    }
    row = {"length_s": float(t[-1])}
    for h in CHECKPOINTS:
        n = int(np.searchsorted(t, h - DT))
        if n < len(t) and t[-1] >= h - 0.05:
            row[f"{h:g}s"] = {name: float(v[n]) for name, v in e.items()}
    row["rms"] = {name: float(np.sqrt(np.mean(v**2))) for name, v in e.items()}
    return row


def replayable(path: Path):
    """A route's columns and meta, its VehicleModel, median roll and angle offset, and gyro bias."""
    c = route(path)
    _, meta, _ = load(path)
    vm = vehicle_model(path, c)
    roll, offset = float(np.nanmedian(c["roll"])), float(np.nanmedian(c["angle_offset"]))
    rest = (c["v"] < 0.01) & np.isfinite(c["gyro_x"])
    return c, meta, vm, roll, offset, float(np.median(c["gyro_x"][rest]))


def evaluate(job):
    path, params = job
    c, meta, vm, roll, offset, bias = replayable(path)
    rows = []
    for k, end in windows(c, meta["commands"]):
        real, t = observed(c, k, end, bias), c["t"][k:end] - c["t"][k]
        row = {"route": path.stem, "t": float(c["t"][k])}
        for name, p in (("sim", params), ("ideal", ideal(params))):
            row[name] = score(simulate(p, c, k, end, vm, roll, offset), real, t)
        rows.append(row)
    return rows


def summarize(rows):
    out = {}
    for model in ("sim", "ideal"):
        out[model] = {}
        for key in [f"{h:g}s" for h in CHECKPOINTS] + ["rms"]:
            values = [r[model][key] for r in rows if key in r[model]]
            if values:
                out[model][key] = {
                    name: round(float(np.median([v[name] for v in values])), 3)
                    for name in values[0]
                } | {"windows": len(values)}
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--series", type=Path, default=SERIES)
    a.add_argument("--routes", nargs="*", help="default: every route not driven on the fit day")
    a.add_argument("--out", type=Path, default=REPO / "runs/sim/validate")
    args = a.parse_args()
    paths = (
        [args.series / f"{r}.npz" for r in args.routes]
        if args.routes
        else [p for p in sorted(args.series.glob("*.npz")) if day(p) != SITE["fit_day"]]
    )
    params = Params.load()
    with ProcessPoolExecutor() as pool:
        rows = [r for rows in pool.map(evaluate, [(p, params) for p in paths]) for r in rows]
    summary = summarize(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "windows.json").write_text(json.dumps(rows))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(rows)} windows from {len(paths)} routes (median absolute error)")
    for key in [f"{h:g}s" for h in CHECKPOINTS] + ["rms"]:
        for model in ("sim", "ideal"):
            if s := summary[model].get(key):
                print(f"{key:>4} {model:6s} " + "  ".join(f"{n} {v}" for n, v in s.items()))


if __name__ == "__main__":
    main()
