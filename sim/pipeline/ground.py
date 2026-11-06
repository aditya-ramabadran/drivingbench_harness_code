"""The lot's ground, from where the road camera drove: camera height below each reconstructed pose.

    python -m drivingbench.sim.pipeline.ground WORK OUT.json

The reconstruction's own low points are not the ground: the asphalt is nearly textureless, and
where passes were registered at different heights (before level.py) a second copy of it lay a
metre lower. The road camera is always CAMERA_HEIGHT_M above the asphalt (within a centimetre of
its height above the 15th percentile of dense nearby points, where they are clean), so each cell
with poses within RADIUS_M takes their median height less the camera height. Cells that disagree
with their measured neighbours by more than SPIKE_M (a few misplaced frames) are dropped, and cells
without poses are filled smoothly from the measured ones. The grid covers the cameras with MARGIN_M
to spare.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
from drivingbench.sim.render import CAMERA_HEIGHT_M
from scipy import ndimage
from scipy.spatial import cKDTree

CELL_M, RADIUS_M, MIN_POSES, SPIKE_M, MARGIN_M = 2.0, 3.0, 3, 0.4, 20.0


def main():
    work, out = Path(sys.argv[1]), Path(sys.argv[2])
    m = pycolmap.Reconstruction(work / "sparse/level")
    centres = np.array(
        [i.projection_center() for i in m.images.values() if i.name.startswith("narrow/")]
    )
    lo = np.floor((centres[:, :2].min(0) - MARGIN_M) / CELL_M) * CELL_M
    shape = tuple(int(n) for n in np.ceil((centres[:, :2].max(0) + MARGIN_M - lo) / CELL_M))
    tree = cKDTree(centres[:, :2])
    grid = np.full(shape, np.nan)
    for i in range(shape[0]):
        for j in range(shape[1]):
            near = tree.query_ball_point(lo + CELL_M * (np.array([i, j]) + 0.5), RADIUS_M)
            if len(near) >= MIN_POSES:
                grid[i, j] = np.median(centres[near, 2]) - CAMERA_HEIGHT_M
    while True:  # drop spikes against the median of measured neighbours within two cells
        local = ndimage.generic_filter(grid, np.nanmedian, size=5, mode="nearest")
        spikes = np.abs(grid - local) > SPIKE_M
        if not spikes.any():
            break
        grid[spikes] = np.nan
    measured = ~np.isnan(grid)
    filled = np.where(measured, grid, np.nanmedian(grid))
    for _ in range(4000):  # harmonic fill: unmeasured cells relax to their neighbours' mean
        mean = ndimage.uniform_filter(filled, size=3, mode="nearest")
        filled = np.where(measured, grid, mean)
    out.write_text(json.dumps({
        "origin": lo.tolist(), "cell_m": CELL_M, "shape": list(shape),
        "source": f"median road-camera height less {CAMERA_HEIGHT_M} m within {RADIUS_M} m "
                  f"({len(centres)} poses, {int(measured.sum())} measured cells); others filled",
        "z": np.round(filled, 3).ravel().tolist(),
    }))  # fmt: skip
    print(f"{len(centres)} poses, {measured.sum()} measured cells of {grid.size}")


if __name__ == "__main__":
    main()
