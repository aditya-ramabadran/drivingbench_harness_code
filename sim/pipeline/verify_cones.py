"""One run's cones: measured candidates, kept where that run's own frames confirm them.

    python -m drivingbench.sim.pipeline.verify_cones WORK OUT START END CANDIDATES.json...

Candidates are cones.py outputs (a wider window has more views; the run's own window has the
run's positions where cones moved). For every reconstructed road-camera frame between START and
END (the site's local HH:MM:SS), each candidate within RANGE_M and in view is projected through the
frame's pose and confirmed when cones.py's detector finds a blob base of its colour within
TOLERANCE_PX.
A candidate the run had in view in MIN_EXPECTED or more frames stays only if confirmed in at least
MIN_CONFIRMED of them; candidates within MERGE_M of a better confirmed one of the same colour are
duplicates. Writes OUT: the kept candidates with `run_expected` and `run_confirmed` (layout.py
turns them into the simulator's course).
"""

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pycolmap
from drivingbench.sim.course import TIMEZONE
from drivingbench.sim.pipeline.cones import MODEL, detect

RANGE_M, TOLERANCE_PX, MIN_EXPECTED, MIN_CONFIRMED, MERGE_M = 12.0, 18.0, 4, 1 / 3, 0.5


def main():
    work, out, start, end = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    cones = [c for path in sys.argv[5:] for c in json.loads(Path(path).read_text())]
    priors = json.loads((work / "priors.json").read_text())
    m = pycolmap.Reconstruction(work / "sparse" / MODEL)
    base = np.array([[c["x"], c["y"], c["z"]] for c in cones])
    colour = np.array([c.get("colour", "red") for c in cones])

    def local(image):
        wall = priors[image.name.split("/", 1)[1]]["wall"]
        return datetime.fromtimestamp(wall, TIMEZONE).strftime("%H:%M:%S")

    frames = [
        i for i in m.images.values() if i.name.startswith("narrow/") and start <= local(i) < end
    ]
    paths = [work / "images" / i.name for i in frames]
    with ProcessPoolExecutor(8) as pool:
        blobs = {c: list(pool.map(partial(detect, colour=c), paths, chunksize=8))
                 for c in set(colour)}  # fmt: skip
    expected, confirmed = np.zeros(len(cones), int), np.zeros(len(cones), int)
    for n, image in enumerate(frames):
        cam = image.cam_from_world() * base
        near = (cam[:, 2] > 0.5) & (np.linalg.norm(cam, axis=1) < RANGE_M)
        uv = np.full((len(cones), 2), -1e9)
        uv[near] = m.cameras[image.camera_id].img_from_cam(cam[near])
        view = near & (uv[:, 0] >= 0) & (uv[:, 0] < 1344) & (uv[:, 1] >= 360) & (uv[:, 1] < 758)
        expected += view
        for c, found in blobs.items():
            if found[n]:
                seen = np.array([b[:2] for b in found[n]])
                gap = np.linalg.norm(uv[:, None] - seen[None], axis=2).min(1)
                confirmed += view & (colour == c) & (gap < TOLERANCE_PX)
    judged = expected >= MIN_EXPECTED
    ratio = np.where(judged, confirmed / np.maximum(expected, 1), 0.0)
    keep = []
    for k in sorted(np.flatnonzero(~judged | (ratio >= MIN_CONFIRMED)),
                    key=lambda k: (not judged[k], -ratio[k], -cones[k]["views"])):  # fmt: skip
        if all(
            math.dist(base[k, :2], base[j, :2]) > MERGE_M or colour[k] != colour[j] for j in keep
        ):
            keep.append(k)
    kept = [cones[k] | {"colour": colour[k], "run_expected": int(expected[k]),
                        "run_confirmed": int(confirmed[k])} for k in keep]  # fmt: skip
    out.write_text(json.dumps(sorted(kept, key=lambda c: (c["x"], c["y"])), indent=1))
    print(
        f"{len(frames)} frames, {len(cones)} candidates ({judged.sum()} in view): {len(kept)} "
        f"kept, {sum(c['run_expected'] >= MIN_EXPECTED for c in kept)} confirmed by the run"
    )


if __name__ == "__main__":
    main()
