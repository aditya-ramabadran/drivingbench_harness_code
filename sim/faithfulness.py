"""How faithful the simulator is along a real attempt at the site.

    uv run python -m drivingbench.sim.faithfulness SESSION [--out runs/sim/faithfulness]
                                                   [--skip-visual]

SESSION is a recorded session (runs/recordings/sessions/SESSION.json) whose drive the build
reconstructed; it also needs the drive's route segments and series. Path: see `path`. Visual:
every 0.2 s of the attempt's control window near a reconstructed frame, the simulator renders the
splat and cones at the real car's reconstructed camera pose (auto-exposed as the simulated camera
is) and is scored against the real road-camera frame captured then: PSNR and SSIM, and PSNR after a
per-frame colour gain (what remaining exposure and colour differences cost). Writes OUT/SESSION.json
and a side-by-side video, OUT/SESSION.mp4 (real | simulated).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from drivingbench.sim.course import REPO, SITE, Course, Placed, comma
from drivingbench.sim.episode import Episode
from drivingbench.sim.logs import events
from drivingbench.sim.plant import Params
from drivingbench.sim.series import SERIES
from drivingbench.sim.validate import observed, replayable, simulate
from PIL import Image

POSES = REPO / SITE["work"] / "poses.json"
ROUTES = REPO / SITE["routes"]
RECORDINGS = REPO / "runs/recordings"
STEP_S, NEAR_S, HORIZON_S = 0.2, 0.6, 10.0


def control_window(session: str) -> tuple[float, float]:
    """Wall time of the first accepted command to the last event of the session's segments."""
    s = json.loads((RECORDINGS / "sessions" / f"{session}.json").read_text())
    ev = [json.loads(line) for g in s["segments"]
          for line in (RECORDINGS / "segments" / g / "events.jsonl").read_text().splitlines()]  # fmt: skip
    wall = lambda e: datetime.fromisoformat(e["timestamp"]).timestamp()  # noqa: E731
    accepted = [wall(e) for e in ev if e.get("kind") == "tool"
                and (e.get("outcome") or {}).get("status") == "accepted"]  # fmt: skip
    return min(accepted), wall(ev[-1])


def camera_path(t0: float, t1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstructed road-camera poses every STEP_S between wall times t0 and t1, only within
    NEAR_S of a reconstructed frame (frame selection skipped stretches other laps covered)."""
    poses = [p for p in json.loads(POSES.read_text()) if t0 - 2 <= p["wall"] <= t1 + 2]
    walls = np.array([p["wall"] for p in poses])
    times = np.arange(t0, t1, STEP_S)
    times = times[(times >= walls[0]) & (times <= walls[-1])]
    k = np.clip(np.searchsorted(walls, times), 1, len(walls) - 1)
    near = np.minimum(np.abs(walls[k] - times), np.abs(walls[k - 1] - times))
    times, k = times[near < NEAR_S], k[near < NEAR_S]
    w = ((times - walls[k - 1]) / (walls[k] - walls[k - 1]))[:, None]

    def mix(key):
        v = np.array([p[key] for p in poses])
        return (1 - w) * v[k - 1] + w * v[k]

    forward, down = mix("forward"), mix("down")
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    down -= (down * forward).sum(1, keepdims=True) * forward
    down /= np.linalg.norm(down, axis=1, keepdims=True)
    return times, mix("centre"), forward, down


def frame_index(minutes: list[str]) -> list[tuple[float, str, int]]:
    """(wall time, route minute, frame number) for every road-camera frame in those minutes."""
    out = []
    for minute in minutes:
        frames, offsets = {}, []
        for mono, kind, m in events(ROUTES / minute / "rlog.zst", {"wideRoadEncodeIdx", "clocks"}):
            if kind == "wideRoadEncodeIdx":
                frames[m.segmentId] = m.timestampEof / 1e9
            else:
                offsets.append(m.wallTimeNanos / 1e9 - mono / 1e9)
        offset = float(np.median(offsets))
        out += [(t + offset, minute, n) for n, t in frames.items()]
    return sorted(out)


def real_frames(times: np.ndarray, index) -> list[np.ndarray]:
    """The road-camera frame nearest each wall time (the cameras run at 20 Hz)."""
    walls = np.array([w for w, _, _ in index])
    picks = [index[int(np.argmin(np.abs(walls - t)))] for t in times]
    frames: dict[tuple[str, int], np.ndarray] = {}
    for minute in sorted({m for _, m, _ in picks}):
        numbers = sorted({n for _, m, n in picks if m == minute})
        select = "+".join(f"eq(n\\,{n})" for n in numbers)
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "hevc", "-i", str(ROUTES / minute / "fcamera.hevc"),
             "-vf", f"select='{select}'", "-fps_mode", "passthrough", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-"], capture_output=True, check=True,
        ).stdout  # fmt: skip
        images = np.frombuffer(raw, np.uint8).reshape(-1, 760, 1344, 3)
        frames |= {(minute, n): image for n, image in zip(numbers, images)}
    return [frames[(m, n)] for _, m, n in picks]


def box(x, k=7):
    s = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / (k * k)


def scores(render: np.ndarray, real: np.ndarray) -> dict:
    a, b = render.astype(np.float64), real.astype(np.float64)
    psnr = lambda x: 10 * math.log10(255**2 / np.mean((x - b) ** 2))  # noqa: E731
    gain = (a * b).sum((0, 1)) / np.maximum((a * a).sum((0, 1)), 1e-9)  # per-channel exposure
    ga, gb = a.mean(2), b.mean(2)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = box(ga), box(gb)
    va, vb, cov = box(ga * ga) - ma**2, box(gb * gb) - mb**2, box(ga * gb) - ma * mb
    ssim = np.mean(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2)))
    return {"psnr": psnr(a), "psnr_exposure_matched": psnr(np.clip(a * gain, 0, 255)),
            "ssim": float(ssim), "gain": gain.round(3).tolist()}  # fmt: skip


def visual(session: str, out: Path) -> dict:
    from drivingbench.sim.render import ROAD_FOV_DEG, ChromeView, auto_expose

    t0, t1 = control_window(session)
    times, centres, forward, down = camera_path(t0, t1)
    minutes = sorted(
        {p["minute"] for p in json.loads(POSES.read_text()) if t0 - 2 <= p["wall"] <= t1 + 2}
    )
    reals = real_frames(times, frame_index(minutes))
    view = ChromeView()
    rows, video = [], out / f"{session}.mp4"
    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1344x380",
         "-r", str(1 / STEP_S), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
         str(video)], stdin=subprocess.PIPE,
    )  # fmt: skip
    try:
        for t, c, f, d, real in zip(times, centres, forward, down, reals):
            sim = auto_expose(view.render(*c, 0.0, 0.0, ROAD_FOV_DEG, forward=f, top=-d))
            rows.append({"t": float(t - t0)} | scores(sim, real))
            half = lambda x: np.asarray(Image.fromarray(x).resize((672, 380)))  # noqa: E731
            encoder.stdin.write(np.concatenate([half(real), half(sim)], 1).tobytes())  # real | sim
    finally:
        view.close()
    encoder.stdin.close()
    encoder.wait()
    summary = {k: round(float(np.median([r[k] for r in rows])), 3)
               for k in ("psnr", "psnr_exposure_matched", "ssim")}  # fmt: skip
    return {"frames": len(rows), "median": summary, "video": str(video), "per_frame": rows}


def path(session: str) -> dict:
    """Replays re-anchored on the reconstruction: from each reconstructed pose of the attempt, the
    fitted car runs the logged requests and is compared with the later reconstructed poses within
    HORIZON_S. `measured` integrates the logged speed and gyro yaw instead, the floor set by the
    pose measurement itself. Course progress is v1.1's, from the GNSS track it scored."""
    from drivingbench.sim.render import YAW_DEG

    t0, t1 = control_window(session)
    poses = [p for p in json.loads(POSES.read_text()) if t0 <= p["wall"] <= t1]
    series = SERIES / f"{poses[0]['minute'].rsplit('--', 1)[0]}.npz"
    c, meta, vm, roll, offset, bias = replayable(series)
    wall = c["t"] + meta["wall_minus_mono_s"]
    course = Course()
    k0, k1 = np.searchsorted(wall, [t0, t1])
    fixes = np.flatnonzero(np.isfinite(c["lat"][k0:k1])) + k0
    gnss = np.stack(course.enu(c["lat"][fixes], c["lon"][fixes]), axis=1)
    episode = Episode(course)
    for t in np.arange(t0, t1, STEP_S):
        e, n = (float(np.interp(t, wall[fixes], gnss[:, i])) for i in range(2))
        episode.update(float(t), Placed(e, n, 0.0), 1.0)
    pw = np.array([p["wall"] for p in poses])
    xy = np.array([p["centre"][:2] for p in poses])
    heading = np.array([math.atan2(p["forward"][1], p["forward"][0]) for p in poses])
    heading -= math.radians(YAW_DEG)  # the camera looks 1.8° right of the car's travel
    params, errors = Params.load(), {"sim": [], "measured": []}
    for i in range(len(poses)):
        later = np.flatnonzero((pw > pw[i]) & (pw <= pw[i] + HORIZON_S))
        if not len(later):
            continue
        k, end = np.searchsorted(wall, [pw[i], pw[later[-1]]])
        rows = np.searchsorted(wall[k : end + 1], pw[later])
        for name, track in (("sim", simulate(params, c, k, end + 1, vm, roll, offset)),
                            ("measured", observed(c, k, end + 1, bias))):  # fmt: skip
            placed = comma(xy[i], heading[i], track[rows, 3], track[rows, 4], track[rows, 2])
            error = np.hypot(placed[0] - xy[later, 0], placed[1] - xy[later, 1])
            errors[name] += zip(pw[later] - pw[i], error)
    median = {name: {f"{a}-{b}s": round(float(np.median([x for dt, x in e if a < dt <= b])), 2)
                     for a, b in ((0, 2), (2, 5), (5, 10))} | {"pairs": len(e)}
              for name, e in errors.items()}  # fmt: skip
    return {"progress_from_gnss": round(episode.progress, 4), "poses": len(poses),
            "position_error_m": median}  # fmt: skip


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("session")
    a.add_argument("--out", type=Path, default=REPO / "runs/sim/faithfulness")
    a.add_argument("--skip-visual", action="store_true")
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"session": args.session, "path": path(args.session)}
    if not args.skip_visual:
        report["visual"] = visual(args.session, args.out)
    (args.out / f"{args.session}.json").write_text(json.dumps(report, indent=1))
    brief = {k: {n: v for n, v in r.items() if n != "per_frame"}
             for k, r in report.items() if isinstance(r, dict)}  # fmt: skip
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
