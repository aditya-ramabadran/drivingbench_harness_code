"""Check the levelled reconstruction against sensors it never used: per-step heading against the
gyro and per-step travel against GNSS Doppler speed (the car's own speed reads ~2.4% high).

    python -m drivingbench.sim.pipeline.check WORK [--series runs/sim/series]

Needs each route's series (sim/series.py). Writes WORK/check.json and WORK/check.png (points,
reconstructed cameras and their GPS priors).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pycolmap
from drivingbench.sim.course import SITE
from drivingbench.sim.series import SERIES, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def series(path: Path) -> dict:
    _, _, c = load(path)
    rest = (c["v"] < 0.01) & np.isfinite(c["gyro_x"])
    dt = np.diff(c["t"], prepend=c["t"][0])
    c["yaw"] = np.cumsum(np.nan_to_num(c["gyro_x"] - np.median(c["gyro_x"][rest])) * dt)
    c["travel"] = np.cumsum(np.nan_to_num(c["gps_speed"]) * dt)
    return c


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("work", type=Path)
    a.add_argument("--series", type=Path, default=SERIES)
    args = a.parse_args()
    m = pycolmap.Reconstruction(args.work / "sparse/level")
    priors = json.loads((args.work / "priors.json").read_text())
    by_minute = defaultdict(list)
    for image in m.images.values():
        if image.name.startswith("narrow/"):
            name = image.name.split("/", 1)[1]
            by_minute[priors[name]["minute"]].append((priors[name]["t_mono"], image))
    travel_err, yaw_err, sfm_len, gnss_len = [], [], 0.0, 0.0
    loaded = {}
    for minute, frames in by_minute.items():
        route = minute.rsplit("--", 1)[0]
        c = loaded.setdefault(route, series(args.series / f"{route}.npz"))
        frames.sort(key=lambda f: f[0])
        t = np.array([f[0] for f in frames])
        centres = np.array([f[1].projection_center() for f in frames])
        forward = np.array([f[1].cam_from_world().rotation.matrix().T @ [0, 0, 1] for f in frames])
        yaw = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
        close = np.diff(t) < 1.0  # consecutive selected frames, not jumps across a gap
        step = np.linalg.norm(np.diff(centres[:, :2], axis=0), axis=1)[close]
        truth = np.diff(np.interp(t, c["t"], c["travel"]))[close]
        moving = truth > 0.05
        travel_err += list(np.abs(step - truth)[moving])
        sfm_len += step[moving].sum()
        gnss_len += truth[moving].sum()
        turn = np.diff(yaw)[close] - np.diff(np.interp(t, c["t"], c["yaw"]))[close]
        yaw_err += list(np.degrees(np.abs(turn)))
    report = {
        "minutes": len(by_minute),
        "steps": len(yaw_err),
        "travel_ratio_sfm_over_gnss": round(sfm_len / gnss_len, 4),
        "travel_error_m": {"median": round(float(np.median(travel_err)), 3),
                           "p95": round(float(np.percentile(travel_err, 95)), 3)},
        "yaw_error_deg": {"median": round(float(np.median(yaw_err)), 3),
                          "p95": round(float(np.percentile(yaw_err, 95)), 3)},
    }  # fmt: skip
    (args.work / "check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    points = np.array([p.xyz for p in m.points3D.values()])
    keep = np.random.default_rng(0).choice(len(points), min(150_000, len(points)), replace=False)
    centres = np.array([i.projection_center() for i in m.images.values()])
    gps = np.array([priors[i.name.split("/", 1)[1]]["enu"] for i in m.images.values()])
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(gps[:, 0], gps[:, 1], s=0.4, c="tab:orange", label="GPS")
    ax.scatter(centres[:, 0], centres[:, 1], s=0.4, c="tab:blue", label="reconstructed cameras")
    lo, hi = centres[:, :2].min(0) - 20, centres[:, :2].max(0) + 20
    ax.scatter(points[keep, 0], points[keep, 1], s=0.05, c="0.55", zorder=0)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal", title=f"{SITE['name']} (m)")
    ax.legend(markerscale=10)
    fig.savefig(args.work / "check.png", dpi=90, bbox_inches="tight")


if __name__ == "__main__":
    main()
