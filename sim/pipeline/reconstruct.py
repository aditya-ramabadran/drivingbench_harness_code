"""The day's reconstruction: both road cameras' frames mapped together, aligned to GPS, cleaned.

    python -m drivingbench.sim.pipeline.reconstruct WORK [--threads N --neighbors 24]

WORK holds images/{narrow,wide}/ and priors.json (frames.py). The wide fisheye sees the dashboard,
mount and border in every frame: pixels that barely change across frames are the car, so they are
masked, with side glass outside a central ellipse. SIFT features per camera from openpilot's comma
four intrinsics (road 1141.5 px, wide fisheye 425 px), matching between frames whose GPS priors
lie within 12 m, then GLOMAP global mapping, which never sees GPS. The largest model is aligned to
the priors by a similarity transform; frames left more than OFF_GPS_M from GPS or seeing fewer than
MIN_POINTS points are degenerate registrations (mostly fast returns to the start) and go, poorly
observed points go, and the model is aligned again. Writes WORK/sparse/clean and WORK/report.json.
One day (5,448 frames per camera) takes 23 min for features, 34 min for matching and 1 h 52 min for
mapping on an Apple-silicon laptop, peaking near 34 GB of memory.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image, ImageFilter

OFF_GPS_M, MIN_POINTS = 8.0, 50
CAMERAS = (("narrow", "SIMPLE_RADIAL", "1141.5,672,380,0"),
           ("wide", "OPENCV_FISHEYE", "425,425,672,380,0,0,0,0"))  # fmt: skip


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def wide_mask(work: Path):
    """The wide camera's cabin mask from frame variability, linked for every wide frame."""
    names = sorted((work / "images/wide").glob("*.jpg"))
    if not (work / "wide-mask.png").exists():
        stack = np.stack([np.asarray(Image.open(n).convert("L"), np.float32) for n in names[::4]])
        moving = Image.fromarray(
            (((stack.std(0) > 18) & (stack.mean(0) > 12)) * 255).astype(np.uint8)
        )
        keep = (
            np.asarray(
                moving.filter(ImageFilter.MedianFilter(15)).filter(ImageFilter.MinFilter(21))
            )
            > 127
        )
        h, w = keep.shape
        yy, xx = np.mgrid[:h, :w]
        keep &= ((xx - w / 2) / 600) ** 2 + ((yy - 330) / 330) ** 2 <= 1
        Image.fromarray((keep * 255).astype(np.uint8)).save(work / "wide-mask.png")
    masks = work / "masks/wide"
    masks.mkdir(parents=True, exist_ok=True)
    for name in names:
        if not (masks / f"{name.name}.png").exists():
            os.symlink(work / "wide-mask.png", masks / f"{name.name}.png")


def align(m: pycolmap.Reconstruction, priors: dict) -> np.ndarray:
    """Similarity-align to the GPS priors; each registered frame's distance from its prior."""
    names = [i.name for i in m.images.values() if i.has_pose]
    located = np.array([[*priors[k.split("/", 1)[1]]["enu"], 1.25] for k in names])
    m.transform(pycolmap.align_reconstruction_to_locations(
        m, names, located, min_common_images=3, ransac_options=pycolmap.RANSACOptions(max_error=4.0)))  # fmt: skip
    centres = np.array([m.find_image_with_name(k).projection_center() for k in names])
    return np.linalg.norm((centres - located)[:, :2], axis=1)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("work", type=Path)
    a.add_argument("--threads", type=int, default=-1)
    a.add_argument("--neighbors", type=int, default=24)
    args = a.parse_args()
    work = args.work.resolve()
    db_path, images = work / "database.db", work / "images"
    priors = json.loads((work / "priors.json").read_text())
    wide_mask(work)
    if not db_path.exists():
        sift = pycolmap.FeatureExtractionOptions()
        sift.sift.max_num_features, sift.num_threads = 6000, args.threads
        for cam, model, params in CAMERAS:
            reader = pycolmap.ImageReaderOptions()
            reader.camera_model, reader.camera_params = model, params
            reader.mask_path = str(work / "masks") if cam == "wide" else ""
            names = sorted(f"{cam}/{p.name}" for p in (images / cam).glob("*.jpg"))
            pycolmap.extract_features(db_path, images, image_names=names, reader_options=reader,
                                      camera_mode=pycolmap.CameraMode.SINGLE,
                                      extraction_options=sift)  # fmt: skip
            log("extracted", cam, len(names))
        with pycolmap.Database.open(db_path) as db:
            for image in db.read_all_images():
                p = priors[image.name.split("/", 1)[1]]
                acc = max(p["accuracy_m"], 1.0)
                db.write_pose_prior(pycolmap.PosePrior(
                    corr_data_id=image.data_id, position=np.array([*p["enu"], 1.25]),
                    position_covariance=np.diag([acc**2, acc**2, 1.0]),
                    coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN))  # fmt: skip
        pairing = pycolmap.SpatialPairingOptions()
        pairing.max_num_neighbors, pairing.max_distance, pairing.ignore_z = (
            args.neighbors,
            12.0,
            True,
        )
        pycolmap.match_spatial(db_path, pairing_options=pairing)
        log("matched")
    options = pycolmap.GlobalPipelineOptions()
    options.num_threads = args.threads
    (work / "sparse").mkdir(exist_ok=True)
    maps = pycolmap.global_mapping(db_path, images, work / "sparse", options=options)
    log("mapped", {i: m.num_reg_images() for i, m in maps.items()})
    m = max(maps.values(), key=lambda m: m.num_reg_images())
    registered = m.num_reg_images()
    off = dict(zip((i.name for i in m.images.values() if i.has_pose), align(m, priors)))
    drop = {i.frame_id for i in m.images.values()
            if i.has_pose and (off[i.name] > OFF_GPS_M or i.num_points3D < MIN_POINTS)}  # fmt: skip
    for frame in drop:
        m.deregister_frame(frame)
    m.filter_all_points3D(4.0, 1.5)
    residual = align(m, priors)
    (work / "sparse/clean").mkdir(exist_ok=True)
    m.write(work / "sparse/clean")
    report = {"models": sorted((n.num_reg_images() for n in maps.values()), reverse=True),
              "registered": registered, "dropped": len(drop), "kept": m.num_reg_images(),
              "points": m.num_points3D(), "reprojection_px": round(m.compute_mean_reprojection_error(), 3),
              "gps_m": {q: round(float(np.percentile(residual, v)), 2)
                        for q, v in (("median", 50), ("p95", 95), ("max", 100))}}  # fmt: skip
    (work / "report.json").write_text(json.dumps(report, indent=1))
    log(json.dumps(report))


if __name__ == "__main__":
    main()
