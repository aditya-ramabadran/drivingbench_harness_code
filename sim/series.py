"""One 100 Hz series per route, plus DrivingBench's commands on its log clock.

    uv run python -m drivingbench.sim.series ROOT [ROOT ...] [--recordings runs/recordings]
                                             [--routes ID ...] [--out runs/sim/series]

ROOT holds openpilot route segments as the comma stores them (`<route>--<n>/rlog.zst`). Rows are
carState ticks; every other signal is its latest value at that tick. Commands (`set_motion`,
`chunk`, `stop_now`) come from the producer's recorded segment events (the mirror `drivingbench
sync` keeps), in wall time, and are placed on each route's monotonic clock through its rlog
`clocks` messages. Each route also records its local day in the site's timezone.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
from drivingbench.sim.course import TIMEZONE
from drivingbench.sim.logs import REPO, events

KINDS = {
    "carState", "carControl", "carOutput", "selfdriveState", "vehicleParameters",
    "gyroscope", "gpsLocationExternal", "clocks", "carParams", "lateralDelay",
}  # fmt: skip
COLUMNS = (
    "t", "v", "v_raw", "a", "gear", "steer_deg", "steer_rate", "standstill", "brake", "gas",
    "steer_pressed", "steer_fault", "enabled", "lat_active", "long_active", "long_state",
    "accel_cmd", "accel_out", "curvature_cmd", "steer_cmd", "steer_out", "angle_offset",
    "steer_ratio", "stiffness", "roll", "lat_delay", "gyro_x", "gyro_y", "gyro_z", "lat", "lon",
    "gps_accuracy", "gps_speed", "gps_bearing",
)  # fmt: skip
TOOLS = ("set_motion", "chunk", "stop_now")
SERIES = REPO / "runs/sim/series"


def route_segments(roots: list[Path]) -> dict[str, list[Path]]:
    """Route id → its segment rlogs in order (folders are named <route>--<n>)."""
    routes = defaultdict(dict)
    for root in roots:
        for rlog in root.glob("*/rlog.zst"):
            route, _, n = rlog.parent.name.rpartition("--")
            routes[route].setdefault(int(n), rlog)
    return {r: [s[n] for n in sorted(s)] for r, s in routes.items()}


def extract(item) -> tuple[str, dict]:
    route, rlogs = item
    latest = dict.fromkeys(COLUMNS, np.nan)
    rows, offsets, car_params = [], [], None
    for rlog in rlogs:
        for mono, kind, m in events(rlog, KINDS):
            if kind == "carState":
                latest.update(
                    t=mono / 1e9, v=m.vEgo, v_raw=m.vEgoRaw, a=m.aEgo, gear=m.gearShifter.raw,
                    steer_deg=m.steeringAngleDeg, steer_rate=m.steeringRateDeg,
                    standstill=m.standstill, brake=m.brakePressed, gas=m.gasPressed,
                    steer_pressed=m.steeringPressed,
                    steer_fault=m.steerFaultTemporary or m.steerFaultPermanent,
                )  # fmt: skip
                rows.append([float(latest[c]) for c in COLUMNS])
            elif kind == "selfdriveState":
                latest["enabled"] = m.enabled
            elif kind == "carControl":
                act = m.actuators
                latest.update(
                    lat_active=m.latActive, long_active=m.longActive,
                    long_state=act.longControlState.raw, accel_cmd=act.accel,
                    curvature_cmd=act.curvature, steer_cmd=act.steeringAngleDeg,
                )  # fmt: skip
            elif kind == "carOutput":
                out = m.actuatorsOutput
                latest.update(accel_out=out.accel, steer_out=out.steeringAngleDeg)
            elif kind == "vehicleParameters":
                latest.update(
                    angle_offset=m.angleOffsetDeg, steer_ratio=m.steerRatio,
                    stiffness=m.stiffnessFactor, roll=m.roll,
                )  # fmt: skip
            elif kind == "lateralDelay":
                latest["lat_delay"] = m.lateralDelay
            elif kind == "gyroscope":
                x, y, z = m.gyroUncalibrated.v
                latest.update(gyro_x=x, gyro_y=y, gyro_z=z)
            elif kind == "gpsLocationExternal" and m.hasFix:
                latest.update(
                    lat=m.latitude, lon=m.longitude, gps_accuracy=m.horizontalAccuracy,
                    gps_speed=m.speed, gps_bearing=m.bearingDeg,
                )  # fmt: skip
            elif kind == "clocks":
                offsets.append(m.wallTimeNanos / 1e9 - mono / 1e9)
            elif kind == "carParams" and car_params is None:
                car_params = m.as_builder().to_bytes()
    data = np.array(rows, dtype=np.float64) if rows else np.zeros((0, len(COLUMNS)))
    meta = {
        "route": route,
        "segments": [p.parent.name for p in rlogs],
        "wall_minus_mono_s": median(offsets) if offsets else None,
    }
    return route, {"data": data, "meta": meta, "car_params": car_params}


def commands(recordings: Path) -> list[dict]:
    """Every command a client sent the comma, with its wall time (s since epoch)."""
    out = []
    for log in sorted(recordings.glob("segments/*/events.jsonl")):
        for line in log.read_text().splitlines():
            e = json.loads(line)
            if e.get("kind") == "tool" and e.get("tool") in TOOLS:
                out.append(
                    {
                        "wall": datetime.fromisoformat(e["timestamp"]).timestamp(),
                        "tool": e["tool"],
                        "arguments": {k: v for k, v in e["arguments"].items() if k != "reason"},
                        "status": (e.get("outcome") or {}).get("status"),
                        "client": e.get("client"),
                        "session": e.get("session"),
                        "segment": e.get("segment"),
                    }
                )
    return sorted(out, key=lambda c: c["wall"])


def load(path: Path) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """(data, meta, columns by name) for one saved route."""
    z = np.load(path, allow_pickle=False)
    data, meta = z["data"], json.loads(str(z["meta"]))
    return data, meta, {c: data[:, i] for i, c in enumerate(COLUMNS)}


def day(path: Path) -> str | None:
    """The local day a saved route was driven (YYYY-MM-DD), without loading its rows."""
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"])).get("day")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("roots", type=Path, nargs="+", help="folders of <route>--<n>/rlog.zst")
    p.add_argument("--routes", nargs="*", help="only these route ids")
    p.add_argument("--recordings", type=Path, default=REPO / "runs/recordings")
    p.add_argument("--out", type=Path, default=SERIES)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cmds = commands(a.recordings)
    routes = route_segments(a.roots)
    if a.routes:
        routes = {r: s for r, s in routes.items() if r in a.routes}
    with ProcessPoolExecutor(a.workers) as pool:
        for route, r in pool.map(extract, routes.items()):
            t, offset = r["data"][:, 0], r["meta"]["wall_minus_mono_s"]
            mine, driven = [], None
            if len(t) and offset is not None:
                lo, hi = t[0] + offset, t[-1] + offset
                mine = [c | {"t": c["wall"] - offset} for c in cmds if lo <= c["wall"] <= hi]
                driven = datetime.fromtimestamp(lo, TIMEZONE).date().isoformat()
            r["meta"] |= {"day": driven, "commands": mine}
            np.savez_compressed(
                a.out / f"{route}.npz",
                data=r["data"].astype(np.float64),
                meta=json.dumps(r["meta"]),
                car_params=np.frombuffer(r["car_params"] or b"", dtype=np.uint8),
            )
            print(f"{route}: {len(r['meta']['segments'])} min, {len(t)} rows, {len(mine)} commands")


if __name__ == "__main__":
    main()
