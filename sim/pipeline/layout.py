"""The simulator's course: the measured cones as two smooth rows, the tall pair and the finish bay,
and the drivable region they bound.

    python -m drivingbench.sim.pipeline.layout VERIFIED.json GREEN.json... CONES_OUT.json BOUNDS_OUT.json

VERIFIED is verify_cones.py's output for the reference run over every candidate set (its own window
and wider ones): cones the run confirmed count double, cones out of its frames' view fill stretches
it never saw, and cones it saw no sign of are already gone. GREEN are cones.py's green candidates
(the small far domes rarely pass a single run's check; the bay stood all afternoon). The course is two
rows of cones along the v1.1 course line. Each row's offset from the line is a smooth curve fitted to
its cones (windowed weighted medians, verified cones counting double, outliers beyond OUTLIER_M left
out), kept within INNER of the line's radius on the inside of a turn. A measured cone within
ROW_TOL_M of its row stands where it was measured; one further off moves onto the row; neither
stands within MIN_GAP_M of a cone already placed. Gaps longer than 1.5 SPACING_M along a row fill
with evenly spaced cones of the nearest measured cone's kind, from where the rows begin to where they
end. A tall cluster (TALL_M) is the side-by-side pair PAIR_M apart; the finish bay is every measured
green dome within FINISH_M of the finish zone. The drivable region is the lane between the rows from
BACK_M behind the start, joined to the finish bay, widened by MARGIN_M and smoothed.
"""

import json
import sys
from pathlib import Path

import contourpy
import numpy as np
from drivingbench.sim.course import COURSE, SITE, Ground, inside
from scipy import ndimage
from scipy.spatial import ConvexHull

LANE_M, CLEAR_M, ROW_TOL_M, OUTLIER_M, WINDOW_M, SMOOTH_M = 7.0, 0.75, 0.8, 1.0, 5.0, 2.0
SPACING_M, MIN_GAP_M, INNER = 2.0, 1.0, 0.85
TALL_M, PAIR_M, FINISH_M, SAME_M = 0.28, 0.5, 5.0, 0.6
BACK_M, MARGIN_M, CELL_M, SOFTEN_M = 6.0, 1.2, 0.1, 0.4


def frame(line: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arclength along the polyline and signed offset from it (left positive) of each point."""
    seg = np.diff(line, axis=0)
    length = np.linalg.norm(seg, axis=1)
    cum = np.r_[0, np.cumsum(length)]
    s, d = np.empty(len(xy)), np.empty(len(xy))
    for lo in range(0, len(xy), 4000):
        p = xy[lo : lo + 4000]
        rel = p[:, None] - line[None, :-1]
        t = np.clip((rel * seg).sum(2) / length**2, 0, 1)
        gap = np.linalg.norm(rel - t[..., None] * seg, axis=2)
        k, i = gap.argmin(1), np.arange(len(p))
        side = np.sign(seg[k, 0] * rel[i, k, 1] - seg[k, 1] * rel[i, k, 0])
        s[lo : lo + 4000], d[lo : lo + 4000] = cum[k] + t[i, k] * length[k], side * gap[i, k]
    return s, d


def at(line: np.ndarray, s: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The points at arclength s and offset d (left positive)."""
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    centre = np.stack([np.interp(s, cum, line[:, i]) for i in range(2)], 1)
    k = np.clip(np.searchsorted(cum, s) - 1, 0, len(line) - 2)
    tangent = line[k + 1] - line[k]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return centre + d[:, None] * np.stack([-tangent[:, 1], tangent[:, 0]], 1)


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(v)
    cum = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cum, cum[-1] / 2)])


def fit_row(s, d, w, grid, curvature, side) -> np.ndarray:
    """One row's offset at each grid arclength."""
    use = np.ones(len(s), bool)
    for _ in range(3):
        est = np.full(len(grid), np.nan)
        for i, g in enumerate(grid):
            near = use & (np.abs(s - g) <= WINDOW_M)
            if near.any():
                est[i] = weighted_median(d[near], w[near])
        known = ~np.isnan(est)
        est = np.interp(grid, grid[known], est[known])
        step = grid[1] - grid[0]
        est = ndimage.gaussian_filter1d(est, SMOOTH_M / step, mode="nearest")
        use = np.abs(d - np.interp(s, grid, est)) <= OUTLIER_M
    return side * np.minimum(side * est, INNER / np.maximum(side * curvature, 1e-9))


def place_row(line, grid, offset, cones, s, d, span) -> list[dict]:
    """The row's cones: measured ones on it, off ones moved onto it, gaps filled."""
    row = at(line, grid, offset)
    along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(row, axis=0), axis=1))]
    u = np.interp(s, grid, along)
    off = np.abs(d - np.interp(s, grid, offset))
    placed: list[tuple[float, dict]] = []
    for k in sorted(range(len(cones)), key=lambda k: (off[k] > ROW_TOL_M, -cones[k]["w"], off[k])):
        if all(abs(u[k] - v) >= MIN_GAP_M for v, _ in placed):
            c = cones[k] | {"source": "measured"}
            if off[k] > ROW_TOL_M:  # off its row: onto it
                x, y = at(line, s[k : k + 1], np.interp(s[k : k + 1], grid, offset))[0]
                c |= {"x": float(x), "y": float(y), "source": "moved"}
            placed.append((u[k], c))
    measured = sorted((v, c["kind"]) for v, c in placed if c["source"] == "measured")
    kind = lambda v: min(measured, key=lambda m: abs(m[0] - v))[1]  # noqa: E731
    lo, hi = np.interp(span, grid, along)
    anchors = sorted(v for v, _ in placed)
    ends = [v for v in (lo, hi) if min(abs(v - a) for a in anchors) > SPACING_M / 2]
    fills = list(ends)
    for a, b in zip(sorted(anchors + ends), sorted(anchors + ends)[1:]):
        if b - a > 1.5 * SPACING_M:
            n = round((b - a) / SPACING_M) - 1
            fills += [a + (b - a) * (j + 1) / (n + 1) for j in range(n)]
    for v in fills:
        x, y = (float(np.interp(v, along, row[:, i])) for i in range(2))
        placed.append((v, {"x": x, "y": y, "kind": kind(v), "colour": "red", "source": "placed"}))
    return [c for _, c in sorted(placed, key=lambda p: p[0])]


def drivable(line, grid, left, right, end_s, bay) -> np.ndarray:
    """The lane between the rows from BACK_M behind the start to end_s, joined to the bay's hull,
    widened by MARGIN_M and smoothed: a closed outline, counterclockwise."""
    tangent = (line[1] - line[0]) / np.linalg.norm(line[1] - line[0])
    behind = line[0] - tangent * np.arange(BACK_M, 0, -(grid[1] - grid[0]))[:, None]
    ext = np.r_[behind, line]
    ext_grid = np.r_[-np.arange(BACK_M, 0, -(grid[1] - grid[0])), grid]
    ends = at(line, np.array([end_s, end_s]), np.array([np.interp(end_s, grid, left),
                                                        np.interp(end_s, grid, right)]))  # fmt: skip
    hull = np.r_[bay, ends]
    hull = hull[ConvexHull(hull).vertices]
    lo, hi = ext.min(0) - LANE_M - MARGIN_M, np.maximum(ext.max(0), hull.max(0)) + LANE_M + MARGIN_M
    xs, ys = np.arange(lo[0], hi[0], CELL_M), np.arange(lo[1], hi[1], CELL_M)
    pts = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    s, d = frame(ext, pts)
    s -= BACK_M
    lane = (s <= end_s) & (d <= np.interp(s, ext_grid, np.r_[np.full(len(behind), left[0]), left]))
    lane &= d >= np.interp(s, ext_grid, np.r_[np.full(len(behind), right[0]), right])
    within = (lane | inside(hull, pts)).reshape(len(ys), len(xs))
    wide = ndimage.distance_transform_edt(~within) * CELL_M <= MARGIN_M
    soft = ndimage.gaussian_filter(wide.astype(float), SOFTEN_M / CELL_M) > 0.5
    labels, _ = ndimage.label(soft)
    soft = ndimage.binary_fill_holes(labels == np.bincount(labels.ravel())[1:].argmax() + 1)
    outline = max(contourpy.contour_generator(xs, ys, soft.astype(float)).lines(0.5), key=len)[::3]
    area = np.sum(
        outline[:, 0] * np.roll(outline[:, 1], -1) - np.roll(outline[:, 0], -1) * outline[:, 1]
    )
    return outline if area > 0 else outline[::-1]


def main():
    verified, *greens, cones_out, bounds_out = (Path(p) for p in sys.argv[1:])
    course = json.loads(COURSE.read_text())
    line, finish = np.array(course["centerline_enu_m"]), np.array(course["finish_zone_enu_m"])
    grid = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    d1 = np.gradient(line, grid, axis=0)
    d2 = np.gradient(d1, grid, axis=0)
    curvature = ndimage.gaussian_filter1d(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], 2)
    found = [{"x": c["x"], "y": c["y"], "kind": c["kind"], "height_m": c.get("height_m"),
              "colour": c["colour"], "w": 2.0 if c["run_confirmed"] else 1.0}
             for c in json.loads(verified.read_text()) if c["colour"] == "red"]  # fmt: skip
    tall = [c for c in found if (c["height_m"] or 0) >= TALL_M and c["w"] > 1]
    red = [c | {"kind": "dome" if c["kind"] == "unknown" else c["kind"]} for c in found
           if (c["height_m"] or 0) < TALL_M]  # fmt: skip
    s, d = frame(line, np.array([[c["x"], c["y"]] for c in red]))
    lane = (np.abs(d) >= CLEAR_M) & (np.abs(d) <= LANE_M)
    span = (s[lane].min(), s[lane].max())
    w = np.array([c["w"] for c in red])
    layout, offsets = [], {}
    for side in (1, -1):
        m = lane & (np.sign(d) == side)
        offsets[side] = fit_row(s[m], d[m], w[m], grid, curvature, side)
        layout += place_row(
            line, grid, offsets[side], [red[k] for k in np.flatnonzero(m)], s[m], d[m], span
        )
    if tall:  # the pair, side by side across the course
        ts, td = frame(line, np.array([[c["x"], c["y"]] for c in tall]))
        for c, a, b in zip(tall, ts, td):
            for x, y in at(line, np.array([a, a]), np.array([b - PAIR_M / 2, b + PAIR_M / 2])):
                layout.append(
                    c | {"x": float(x), "y": float(y), "kind": "tall", "source": "measured"}
                )
    kept = []  # measured cones first; a moved or placed one keeps MIN_GAP_M from every other
    for c in sorted(layout, key=lambda c: ("measured", "moved", "placed").index(c["source"])):
        if c["source"] == "measured" or all(
            np.hypot(o["x"] - c["x"], o["y"] - c["y"]) >= MIN_GAP_M for o in kept
        ):
            kept.append(c)
    layout = kept
    bay = []
    for c in (c for p in greens for c in json.loads(p.read_text())):
        if np.linalg.norm(finish.mean(0) - [c["x"], c["y"]]) <= FINISH_M and all(
            np.hypot(c["x"] - b["x"], c["y"] - b["y"]) > SAME_M for b in bay
        ):
            bay.append(
                {"x": c["x"], "y": c["y"], "kind": "dome", "colour": "green", "source": "measured"}
            )
    layout += bay
    ground = Ground.load()
    cones = [{"x": round(c["x"], 3), "y": round(c["y"], 3), "z": round(float(ground(c["x"], c["y"])), 3),
              "kind": c["kind"], "colour": c["colour"], "source": c["source"]}
             for c in sorted(layout, key=lambda c: (c["x"], c["y"]))]  # fmt: skip
    counts = {k: sum(c["source"] == k for c in cones) for k in ("measured", "moved", "placed")}
    cones_out.write_text(json.dumps({
        "name": SITE["name"],
        "frame": f"the course frame ({SITE['course']}), metres",
        "measured": "cones.py on the levelled reconstruction over "
                    + ", ".join("-".join(w) for w in SITE["cone_windows"])
                    + f", verified against the reference run ({'-'.join(SITE['reference_run'])}) "
                    "by verify_cones.py",
        "layout": __doc__.split("\n\n")[2].replace("\n", " ").strip(),
        "sources": counts,
        "cones": cones,
    }, indent=1))  # fmt: skip
    outline = drivable(
        line, grid, offsets[1], offsets[-1], span[1], np.array([[c["x"], c["y"]] for c in bay])
    )
    bounds_out.write_text(json.dumps({
        "name": "the simulator's drivable region",
        "rule": f"the lane between the cone rows from {BACK_M} m behind the start through the finish "
                f"bay, widened {MARGIN_M} m beyond the cones and smoothed; the car's body stays inside",
        "margin_m": MARGIN_M,
        "polygon_enu_m": np.round(outline, 3).tolist(),
    }))  # fmt: skip
    kinds = {k: sum(c["kind"] == k for c in cones) for k in ("cone", "dome", "tall")}
    print(
        f"{len(cones)} cones {kinds} {counts}; rows from s={span[0]:.1f} to {span[1]:.1f} m; "
        f"drivable outline of {len(outline)} points"
    )


if __name__ == "__main__":
    main()
