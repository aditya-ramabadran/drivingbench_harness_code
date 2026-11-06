"""Delete the real cones, and the haze the car drove through, from the trained splat; every other
Gaussian is kept byte for byte.

    python -m drivingbench.sim.pipeline.clean_splat WORK IN.ply OUT.ply

Cones: the cones moved during the training window, so removal does not use measured positions: it
takes cone-red Gaussians (hue above 340° or below 8°, saturation over 0.45; the faded curb paint is
orange and stays) from BELOW_M under to ABOVE_M over the driven ground (ground.py) and within BAND_M
of the v1.1 course line, plus small neutral Gaussians within HALO_M of them (highlights and edges).
BELOW_M reaches the reconstruction's second, lower copy of the asphalt east of its seam.

Haze: Gaussians from HAZE_M[0] to HAZE_M[1] over the ground within CLEAR_M of a road camera's path
(WORK/poses.json). The car occupied that space, so nothing solid stood there; training put
translucent splats there to explain windshield glare and the afternoon's changing light, and from
any other viewpoint they veil the asphalt. Held-out frames score the same without them (SSIM
0.753 → 0.746, PSNR 19.8 dB both); a replayed run whose low-sun glare they partly reproduced loses
0.55 dB.
"""

import json
import sys
from pathlib import Path

import numpy as np
from drivingbench.sim.course import COURSE, Ground
from drivingbench.sim.explorer.convert import SH_C0, read_ply
from scipy.spatial import cKDTree

BAND_M, BELOW_M, ABOVE_M, HALO_M, HALO_SIZE_M = 7.0, 1.9, 0.6, 0.1, 0.12
HAZE_M, CLEAR_M = (0.12, 2.5), 3.0


def main():
    work, source, target = (Path(a) for a in sys.argv[1:4])

    c = read_ply(source)
    xyz = np.stack([c["x"], c["y"], c["z"]], axis=1)
    rgb = np.clip(0.5 + SH_C0 * np.stack([c[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)
    top, low = rgb.max(1), rgb.min(1)
    saturation = (top - low) / np.maximum(top, 1e-6)
    r, g, b = rgb.T
    hue = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    size = np.exp(np.stack([c[f"scale_{i}"] for i in range(3)], 1)).max(1)
    above = xyz[:, 2] - Ground.load()(xyz[:, 0], xyz[:, 1])
    line = np.array(json.loads(COURSE.read_text())["centerline_enu_m"])
    near_course = cKDTree(line).query(xyz[:, :2])[0] <= BAND_M
    on_ground = (above >= -BELOW_M) & (above <= ABOVE_M)
    cone = ((hue > 340) | (hue < 8)) & (saturation > 0.45) & on_ground & near_course
    halo = np.zeros(len(cone), bool)
    if cone.any():
        close = cKDTree(xyz[cone]).query(xyz)[0] <= HALO_M
        halo = close & ~cone & (size <= HALO_SIZE_M) & on_ground & near_course & (saturation < 0.3)
    paths = np.array([p["centre"][:2] for p in json.loads((work / "poses.json").read_text())])
    driven = cKDTree(paths).query(xyz[:, :2])[0] <= CLEAR_M
    haze = driven & (above > HAZE_M[0]) & (above < HAZE_M[1]) & ~cone & ~halo
    drop = cone | halo | haze
    with source.open("rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        rows = np.frombuffer(f.read(), np.float32).reshape(len(drop), -1)
    count = str(int((~drop).sum())).encode()
    lines = [count.join(line.rsplit(line.split()[-1], 1)) if line.startswith(b"element vertex")
             else line for line in header.split(b"\n")]  # fmt: skip
    target.write_bytes(b"\n".join(lines) + rows[~drop].tobytes())
    print(f"{len(drop)} Gaussians: removed {cone.sum()} cone-red, {halo.sum()} halo and "
          f"{haze.sum()} haze; kept {(~drop).sum()}")  # fmt: skip


if __name__ == "__main__":
    main()
