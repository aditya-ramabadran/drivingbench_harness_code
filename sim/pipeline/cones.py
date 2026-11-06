"""Measure the course's cones from footage and the reconstruction.

    python -m drivingbench.sim.pipeline.cones WORK START END [green]

WORK holds sparse/MODEL (the levelled reconstruction, level.py), priors.json and images/. In each
road-camera frame from START to END (the site's local HH:MM), red blobs are cast from their base
pixel, through the frame's reconstructed pose, onto the ground the cameras drove on (ground.py).
Hits from many frames cluster per cone; a cone needs at least MIN_VIEWS frames seen from positions
at least MIN_SPREAD_M apart; its height is the median blob height within 10 m. `green` measures the
finish bay's green domes instead. Writes WORK/cones-STARTEND[-green].json and a matching .png.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pycolmap
from drivingbench.sim.course import COURSE, SITE, TIMEZONE, Ground
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

GROUND = Ground.load()

MODEL = "level"
MAX_RANGE_M, MERGE_M, MIN_VIEWS, MIN_SPREAD_M = 20.0, 0.35, 6, 2.0
# Hue window (degrees) and pixel / blob-median saturation floors. Real cones are pure red (hue
# ~355°, saturation ~0.68) and compact; the faded fire-lane curb paint is orange (hue 5–23°,
# saturation ~0.51) and long, foliage is neither. The finish domes are yellow-green (hue ~76°,
# saturation ~0.75), far more saturated than foliage.
COLOURS = {"red": ((340, 10), 0.42, 0.48), "green": ((65, 90), 0.55, 0.6)}


def detect(path, colour="red"):
    """Cone blobs of one colour: (base x, base y, height px, width px, area px)."""
    (lo, hi), floor, blob_floor = COLOURS[colour]
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    top, low = im.max(2), im.min(2)
    saturation = (top - low) / np.maximum(top, 1e-6)
    hue = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    in_hue = (hue > lo) | (hue < hi) if lo > hi else (hue > lo) & (hue < hi)
    mask = in_hue & (saturation > floor) & (top > 0.15)  # shadowed cones too
    mask[:360] = False  # above the horizon: never a cone on the ground
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    blobs = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        inside = labels[sl] == k
        area = int(inside.sum())
        if area < 4 or ys.stop >= im.shape[0] - 2 or h > 140 or w > 160:
            continue
        if (
            area / (h * w) < 0.35
            or not 0.25 <= h / w <= 3.5
            or np.median(saturation[sl][inside]) < blob_floor
        ):
            continue
        bottom = np.nonzero(labels[ys.stop - 1, xs] == k)[0]
        blobs.append((xs.start + bottom.mean() + 0.5, ys.stop, h, w, area))
    return blobs


def cast(job):
    name, centre, rotation, params, blobs = job
    camera = pycolmap.Camera(model="SIMPLE_RADIAL", width=1344, height=760, params=params)
    hits = []
    for x, y, h, w, area in blobs:
        xy = camera.cam_from_img(np.array([x, y]))
        ray = rotation.T @ np.array([xy[0], xy[1], 1.0])
        ray /= np.linalg.norm(ray)
        if ray[2] >= -0.01:
            continue
        t = np.linspace(0.5, MAX_RANGE_M, 400)
        p = centre + t[:, None] * ray
        above = p[:, 2] - GROUND(p[:, 0], p[:, 1])
        cross = np.nonzero((above[:-1] > 0) & (above[1:] <= 0))[0]
        if not len(cross):
            continue
        k = cross[0]
        f = above[k] / (above[k] - above[k + 1])
        hit = p[k] + f * (p[k + 1] - p[k])
        hits.append((*hit, float(t[k] + f * (t[1] - t[0])), h, w, area, name, *centre, *ray))
    return hits


def triangulate(centres, rays, ranges, members, rounds=4):
    """The point nearest every base ray (weighted 1/range), dropping rays that miss it."""
    keep = np.ones(len(rays), bool)
    point = None
    for _ in range(rounds):
        if keep.sum() < MIN_VIEWS:
            return None, members
        w = 1 / np.maximum(ranges[keep], 1.0)
        projector = np.eye(3)[None] - rays[keep, :, None] * rays[keep, None, :]
        a = (w[:, None, None] * projector).sum(0)
        b = (w[:, None, None] * projector @ centres[keep, :, None]).sum(0)[:, 0]
        point = np.linalg.solve(a, b)
        offset = point - centres
        miss = np.linalg.norm(offset - (offset * rays).sum(1, keepdims=True) * rays, axis=1)
        keep = miss < max(0.15, 2.5 * np.median(miss[keep]))
    return point, members[keep]


def main():
    work, window = Path(sys.argv[1]), (sys.argv[2], sys.argv[3])  # the site's local time
    colour = sys.argv[4] if len(sys.argv) > 4 else "red"
    priors = json.loads((work / "priors.json").read_text())
    m = pycolmap.Reconstruction(work / "sparse" / MODEL)
    params = m.cameras[1].params
    jobs = []

    def local(image):
        wall = priors[image.name.split("/", 1)[1]]["wall"]
        return datetime.fromtimestamp(wall, TIMEZONE).strftime("%H:%M")

    names = [i for i in m.images.values()
             if i.name.startswith("narrow/") and window[0] <= local(i) < window[1]]  # fmt: skip
    with ProcessPoolExecutor(8) as pool:
        paths = [work / "images" / i.name for i in names]
        found = list(pool.map(partial(detect, colour=colour), paths, chunksize=16))
        for image, blobs in zip(names, found):
            pose = image.cam_from_world()
            jobs.append((image.name, image.projection_center(), pose.rotation.matrix(), params,
                         blobs))  # fmt: skip
        hits = [h for found_hits in pool.map(cast, jobs, chunksize=16) for h in found_hits]
    xyz = np.array([h[:3] for h in hits])
    rng = np.array([h[3] for h in hits])
    aspect = np.array([h[4] / max(h[5], 1) for h in hits])
    height = np.array([h[4] * h[3] / params[0] for h in hits])  # blob height at its range
    image = np.array([h[7] for h in hits])
    cams = np.array([h[8:11] for h in hits])
    rays = np.array([h[11:14] for h in hits])
    pairs = cKDTree(xyz[:, :2]).query_pairs(MERGE_M, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xyz),) * 2)
    _, label = connected_components(graph, directed=False)
    cones = []
    for k in np.unique(label):
        members = np.flatnonzero(label == k)
        views = len(set(image[members]))
        spread = np.ptp(cams[members, :2], axis=0).max() if len(members) > 1 else 0.0
        if views < MIN_VIEWS or spread < MIN_SPREAD_M:
            continue
        weight = 1 / np.maximum(rng[members], 1.0) ** 2
        centre = (xyz[members] * weight[:, None]).sum(0) / weight.sum()
        point, _ = triangulate(cams[members], rays[members], rng[members], members)
        close = members[rng[members] < 10]
        tall = float(np.median(aspect[close])) > 1.0 if len(close) else None
        tall_m = round(float(np.median(height[close])), 3) if len(close) else None
        scatter = np.sqrt(np.average(np.sum((xyz[members, :2] - centre[:2]) ** 2, axis=1),
                                     weights=weight))  # fmt: skip
        cones.append({"x": round(float(centre[0]), 3), "y": round(float(centre[1]), 3),
                      "z": round(float(centre[2]), 3),
                      "kind": "cone" if tall else "dome" if tall is not None else "unknown",
                      "colour": colour, "height_m": tall_m,
                      "views": views, "spread_m": round(float(spread), 1),
                      "scatter_m": round(float(scatter), 3),
                      "triangulated": None if point is None else [round(float(v), 3) for v in point]})  # fmt: skip
    cones.sort(key=lambda c: (c["x"], c["y"]))
    tag = f"-{window[0]}-{window[1]}".replace(":", "") + ("" if colour == "red" else f"-{colour}")
    (work / f"cones{tag}.json").write_text(json.dumps(cones, indent=1))
    kinds = {k: sum(c["kind"] == k for c in cones) for k in ("cone", "dome", "unknown")}
    print(f"{len(names)} frames, {len(hits)} hits, {len(cones)} cones {kinds}")
    figure(cones, m, work / f"cones{tag}.png")


def figure(cones, m, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    course = json.loads(COURSE.read_text())
    centres = np.array([i.projection_center() for i in m.images.values()])
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(centres[:, 0], centres[:, 1], s=0.2, c="0.8", label="reconstructed cameras")
    line = np.array(course["centerline_enu_m"])
    ax.plot(line[:, 0], line[:, 1], c="0.4", lw=1, label="course line")
    for kind, colour in (("cone", "tab:red"), ("dome", "tab:orange"), ("unknown", "tab:purple")):
        pts = np.array([[c["x"], c["y"]] for c in cones if c["kind"] == kind]).reshape(-1, 2)
        ax.scatter(pts[:, 0], pts[:, 1], s=14, c=colour, label=f"{kind} ({len(pts)})")
    lo, hi = line.min(0) - 15, line.max(0) + 15
    ax.set(
        xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal", title=f"Cones: {SITE['name']} (m)"
    )
    ax.legend(loc="lower right", markerscale=2)
    fig.savefig(out, dpi=90, bbox_inches="tight")


if __name__ == "__main__":
    main()
