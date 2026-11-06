"""Build the simulator for a site from the car's drives there.

    uv run --group sim-build python -m drivingbench.sim.pipeline [--site FILE] [--run [STAGE ...]]

Without --run: every stage in order, whether its outputs exist, and the exact commands it runs.
--run: the stages whose outputs are missing, in order (named stages: those, even if done),
stopping at the first failure. The site file (sim/foothill-oct27.json) names the course, the days
and time windows, and where inputs and outputs are; every stage reads it, as does the simulator.

Inputs: openpilot route segments at the site under the site's `routes` (`<route>--<n>/` with
rlog.zst, fcamera.hevc and ecamera.hevc, as the comma stores them) from its `day` and `fit_day`;
the course; and the producer's recordings (runs/recordings, mirrored by `drivingbench sync`) for
the car's commands. Training needs Brush 0.3.0 (github.com/ArthurBrussee/brush, release v0.3.0;
check its SHA-256), as `brush_app` on PATH or BRUSH; a GPU Mac trains 30,000 steps in about 40
minutes. The slow stages: reconstruct (hours of CPU, ~34 GB of memory for a day), level (~15 min),
train.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from drivingbench.sim.course import REPO, SITE_FILE
from drivingbench.sim.plant import PARAMS
from drivingbench.sim.series import SERIES

STEPS, MAX_SPLATS, HOLDOUT = 30_000, 3_000_000, 20  # HOLDOUT: as training_set.py


@dataclass(frozen=True)
class Stage:
    name: str
    makes: str
    outputs: list[Path]
    commands: list[list]

    @property
    def done(self) -> bool:
        return all(p.exists() and (p.is_file() or any(p.iterdir())) for p in self.outputs)


def stages(site: dict) -> list[Stage]:
    work = REPO / site["work"]
    out = {k: REPO / site[k] for k in ("ground", "cones", "bounds")}
    splats = {role: REPO / path for role, path in site["splats"].items()}
    trained, clean, verified = (
        work / f"splat/out/splat_{STEPS}.ply",
        work / "splat/clean.ply",
        work / "verified.json",
    )
    windows = [(a, b, f"{a}-{b}".replace(":", "")) for a, b in site["cone_windows"]]
    red = [work / f"cones-{tag}.json" for *_, tag in windows]
    green = [work / f"cones-{tag}-green.json" for *_, tag in windows]

    def py(module, *args):
        return [sys.executable, "-m", f"drivingbench.sim.{module}", *args]

    brush = os.environ.get("BRUSH", "brush_app")
    return [
        Stage("series", "the car's logs and commands as 100 Hz series, per route", [SERIES],
              [py("series", REPO / site["routes"], "--out", SERIES)]),
        Stage("fit", "the car's dynamics, fitted to one day's series", [PARAMS],
              [py("fit", "--series", SERIES, "--day", site["fit_day"])]),
        Stage("frames", "frames from both road cameras, chosen for coverage, with GPS priors",
              [work / "priors.json"], [py("pipeline.frames", work)]),
        Stage("reconstruct", "the 3D reconstruction, aligned to GPS and cleaned",
              [work / "sparse/clean"], [py("pipeline.reconstruct", work)]),
        Stage("level", "the reconstruction levelled to one asphalt; the cameras' poses",
              [work / "sparse/level", work / "poses.json"], [py("pipeline.level", work)]),
        Stage("check", "the reconstruction against the gyro and GNSS it never used",
              [work / "check.json"], [py("pipeline.check", work, "--series", SERIES)]),
        Stage("ground", "the asphalt's height", [out["ground"]],
              [py("pipeline.ground", work, out["ground"])]),
        Stage("training-set", "the splat's training frames, undistorted, some held out",
              [work / "splat/data"], [py("pipeline.training_set", work, *site["splat_window"])]),
        Stage("train", "the Gaussian splat (Brush)", [trained],
              [[brush, work / "splat/data", "--total-steps", STEPS, "--max-splats", MAX_SPLATS,
                "--eval-split-every", HOLDOUT, "--eval-every", STEPS, "--eval-save-to-disk",
                "--export-every", STEPS, "--export-path", work / "splat/out",
                "--export-name", "splat_{iter}.ply"]]),  # fmt: skip
        Stage("score", "the splat on held-out frames", [work / "splat/heldout.json"],
              [py("pipeline.score_splat", work, STEPS)]),
        Stage("clean-splat", "the splat without its real cones and the haze the car drove through",
              [clean], [py("pipeline.clean_splat", work, trained, clean)]),
        Stage("explorer", "the explorer's and simulated camera's splats", list(splats.values()),
              [py("explorer.convert", trained, splats["raw"]),
               py("explorer.convert", clean, splats["clean"]),
               py("explorer.convert", clean, splats["full"], "--max-splats", MAX_SPLATS)]),  # fmt: skip
        Stage("cones", "cone and finish-dome candidates in each window", red + green,
              [py("pipeline.cones", work, a, b, *colour) for a, b, _ in windows for colour in ((), ("green",))]),
        Stage("verify", "the candidates the reference run's own frames do not contradict",
              [verified], [py("pipeline.verify_cones", work, verified, *site["reference_run"], *red, *green)]),
        Stage("layout", "the course: cones in two smooth rows, the finish bay, the drivable region",
              [out["cones"], out["bounds"]],
              [py("pipeline.layout", verified, *green, out["cones"], out["bounds"])]),
    ]  # fmt: skip


def shown(command: list) -> str:
    """A command as typed from the repository root."""

    def short(part):
        text = str(part)
        return "python" if text == sys.executable else text.replace(f"{REPO}/", "")

    return shlex.join(short(part) for part in command)


def main():
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("--site", type=Path, default=SITE_FILE)
    a.add_argument(
        "--run", nargs="*", metavar="STAGE", help="run missing stages, or the named ones"
    )
    args = a.parse_args()
    site = json.loads(args.site.read_text())
    plan = stages(site)
    names = [s.name for s in plan]
    if args.run is None:
        where = args.site.relative_to(REPO) if args.site.is_relative_to(REPO) else args.site
        print(f"{site['name']} ({where})")
        for s in plan:
            print(f"\n{'done' if s.done else '----'}  {s.name}: {s.makes}")
            for command in s.commands:
                print(f"      {shown(command)}")
        return
    unknown = set(args.run) - set(names)
    if unknown:
        a.error(f"unknown stages {sorted(unknown)}; stages are {', '.join(names)}")
    env = os.environ | {"DRIVINGBENCH_SIM_SITE": str(args.site.resolve())}
    for s in plan:
        if (s.name in args.run) if args.run else not s.done:
            print(f"== {s.name}: {s.makes}", flush=True)
            for command in s.commands:
                print(f"   {shown(command)}", flush=True)
                subprocess.run([str(part) for part in command], cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
