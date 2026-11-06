"""Level the reconstruction: one asphalt, not several.

    python -m drivingbench.sim.pipeline.level WORK

GLOMAP can register some passes 1–1.6 m above others over the same asphalt while GNSS altitude
shows no step; two copies of the ground fog the splat and throw cone measurements off. A pose-prior
bundle adjustment holds every camera's reconstructed east/north within XY_STD_M and pulls its height
toward the road camera's height above a near-flat lot (CAMERA_HEIGHT_M), first loosely, then
tighter (PASSES: height σ and iterations); the images settle everything else (points and poses
refined together, intrinsics fixed, a robust loss). Reads WORK/sparse/clean; writes
WORK/sparse/level and WORK/poses.json: each road-camera frame's name, minute, t_mono, wall, centre,
forward and down.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
from drivingbench.sim.render import CAMERA_HEIGHT_M

XY_STD_M, PASSES = 0.5, ((0.25, 50), (0.12, 200))


def heights(m: pycolmap.Reconstruction) -> str:
    z = np.array([i.projection_center()[2] for i in m.images.values()])
    return f"camera z p5/p50/p95 {np.percentile(z, [5, 50, 95]).round(2).tolist()}"


def adjust(m: pycolmap.Reconstruction, z_std: float, iterations: int):
    config, priors = pycolmap.BundleAdjustmentConfig(), []
    for image_id in m.reg_image_ids():
        image = m.image(image_id)
        config.add_image(image_id)
        x, y, _ = image.projection_center()
        priors.append(pycolmap.PosePrior(
            corr_data_id=image.data_id, position=np.array([x, y, CAMERA_HEIGHT_M]),
            position_covariance=np.diag([XY_STD_M**2, XY_STD_M**2, z_std**2]),
            coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN))  # fmt: skip
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = options.refine_extra_params = False
    options.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    options.ceres.solver_options.max_num_iterations = iterations
    prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
    pycolmap.create_pose_prior_bundle_adjuster(options, prior_options, config, priors, m).solve()
    print(heights(m), "; reprojection", round(m.compute_mean_reprojection_error(), 3), flush=True)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("work", type=Path)
    work = a.parse_args().work
    m = pycolmap.Reconstruction(work / "sparse/clean")
    print(m.num_reg_images(), "images;", heights(m))
    for z_std, iterations in PASSES:
        adjust(m, z_std, iterations)
    (work / "sparse/level").mkdir(parents=True, exist_ok=True)
    m.write(work / "sparse/level")
    priors = json.loads((work / "priors.json").read_text())
    poses = []
    for image in m.images.values():
        if image.name.startswith("narrow/") and image.has_pose:
            p = priors[image.name.split("/", 1)[1]]
            r = image.cam_from_world().rotation.matrix()
            poses.append({"name": image.name.split("/", 1)[1], "minute": p["minute"],
                          "t_mono": p["t_mono"], "wall": p["wall"],
                          "centre": [round(float(v), 4) for v in image.projection_center()],
                          "forward": [round(float(v), 6) for v in r[2]],
                          "down": [round(float(v), 6) for v in r[1]]})  # fmt: skip
    (work / "poses.json").write_text(json.dumps(sorted(poses, key=lambda p: p["wall"])))


if __name__ == "__main__":
    main()
