"""Coverage-selected frames from a day's drives at the site, both road cameras.

    python -m drivingbench.sim.pipeline.frames WORK [--cell 1.0 --per-bin 2]

Reads every route segment under the site's `routes` and keeps the frames captured on its `day`
within its `daylight` window (local time). Every moving 10 Hz candidate is binned by 1 m cell and
45° heading; each bin keeps up to `per-bin` frames from different passes (route minutes), so
repeated laps add viewpoints, not duplicates, plus one frame per stop. Each frame's GPS prior, in
the course frame, is interpolated at its exact capture time (wide-camera EOF; the road cameras are
hardware-synchronised). Writes WORK/images/{narrow,wide}/ and WORK/priors.json.
"""

import argparse
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from drivingbench.sim.course import REPO, SITE, TIMEZONE, Course
from drivingbench.sim.logs import events

ROUTES = REPO / SITE["routes"]
VIDEOS = (("narrow", "fcamera.hevc"), ("wide", "ecamera.hevc"))
FRAME = Course()


def daylight(wall: float) -> bool:
    """Whether a wall time falls on the site's day within its daylight window."""
    local = datetime.fromtimestamp(wall, TIMEZONE)
    start, end = SITE["daylight"]
    return local.date().isoformat() == SITE["day"] and start <= local.strftime("%H:%M") < end


def candidates(minute: str) -> list[dict]:
    """10 Hz frames with prior position, GNSS bearing and speed at capture."""
    frames, gps, speed, offsets = {}, [], [], []
    for mono, kind, m in events(ROUTES / minute / "rlog.zst", {"wideRoadEncodeIdx", "gpsLocationExternal", "carState", "clocks"}):  # fmt: skip
        if kind == "wideRoadEncodeIdx":
            frames[m.segmentId] = m.timestampEof / 1e9
        elif kind == "gpsLocationExternal" and m.hasFix and m.horizontalAccuracy < 5:
            gps.append((mono / 1e9, m.latitude, m.longitude, m.horizontalAccuracy, m.bearingDeg))
        elif kind == "carState":
            speed.append((mono / 1e9, m.vEgo))
        elif kind == "clocks":
            offsets.append(m.wallTimeNanos / 1e9 - mono / 1e9)
    if len(gps) < 10 or not offsets:
        return []
    gt, glat, glon, gacc, gbear = map(np.array, zip(*gps))
    st, sv = map(np.array, zip(*speed))
    offset = float(np.median(offsets))
    out = []
    for n, t in sorted(frames.items()):
        if n % 2 or not gt[0] <= t <= gt[-1] or not daylight(t + offset):
            continue
        e, north = FRAME.enu(np.interp(t, gt, glat), np.interp(t, gt, glon))
        bearing = gbear[min(np.searchsorted(gt, t), len(gt) - 1)]
        out.append({"minute": minute, "n": n, "t_mono": t, "wall": t + offset,
                    "enu": (float(e), float(north)), "accuracy_m": float(np.interp(t, gt, gacc)),
                    "bearing_deg": float(bearing), "speed_mps": float(np.interp(t, st, sv))})  # fmt: skip
    return out


def select(frames: list[dict], cell: float, per_bin: int) -> list[dict]:
    bins = defaultdict(list)
    stops = {}
    for f in frames:
        if f["speed_mps"] > 0.3:
            e, n = f["enu"]
            key = (round(e / cell), round(n / cell), int((f["bearing_deg"] % 360) // 45))
            bins[key].append(f)
        else:  # one frame per minute's standstill position (start line, finish)
            stops.setdefault((f["minute"], round(f["enu"][0] / 2), round(f["enu"][1] / 2)), f)
    chosen = list(stops.values())
    for members in bins.values():
        seen = set()
        for f in sorted(members, key=lambda f: f["accuracy_m"]):
            if f["minute"] not in seen and len(seen) < per_bin:
                seen.add(f["minute"])
                chosen.append(f)
    return sorted(chosen, key=lambda f: (f["minute"], f["n"]))


def extract(job):
    minute, numbers, out = job
    for cam, video in VIDEOS:
        folder = out / "images" / cam
        folder.mkdir(parents=True, exist_ok=True)
        select = "+".join(f"eq(n\\,{n})" for n in numbers)
        temp = folder / f"{minute}.%04d.tmp.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "hevc", "-i", str(ROUTES / minute / video),
                        "-vf", f"select='{select}'", "-fps_mode", "passthrough", "-q:v", "2", str(temp)],
                       check=True)  # fmt: skip
        for k, n in enumerate(numbers, start=1):
            (folder / f"{minute}.{k:04d}.tmp.jpg").rename(folder / f"{minute}-{n:04d}.jpg")
    return minute


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("work", type=Path)
    a.add_argument("--cell", type=float, default=1.0)
    a.add_argument("--per-bin", type=int, default=2)
    args = a.parse_args()
    minutes = sorted(p.name for p in ROUTES.iterdir()
                     if all((p / f).exists() for f in ("rlog.zst", *(v for _, v in VIDEOS))))  # fmt: skip
    with ProcessPoolExecutor(8) as pool:
        frames = [f for found in pool.map(candidates, minutes) for f in found]
        chosen = select(frames, args.cell, args.per_bin)
        by_minute = defaultdict(list)
        for f in chosen:
            by_minute[f["minute"]].append(f["n"])
        list(pool.map(extract, [(m, ns, args.work) for m, ns in by_minute.items()]))
    priors = {f"{f['minute']}-{f['n']:04d}.jpg": f for f in chosen}
    (args.work / "priors.json").write_text(json.dumps(priors))
    used = len({f["minute"] for f in frames})
    print(f"{used} minutes, {len(frames)} candidates, {len(chosen)} frames per camera")


if __name__ == "__main__":
    main()
