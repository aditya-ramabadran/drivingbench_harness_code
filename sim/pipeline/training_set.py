"""Training set for the splat: the levelled reconstruction's road-camera frames in one lighting
window (the site's local HH:MM), undistorted to pinhole, every HOLDOUTth frame held out.

    python -m drivingbench.sim.pipeline.training_set WORK START END

Writes WORK/splat/data (images and the undistorted model, what Brush trains on) and
WORK/splat/split.json.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pycolmap
from drivingbench.sim.course import TIMEZONE

HOLDOUT = 20  # as Brush's --eval-split-every (sim/pipeline/__main__.py)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("work", type=Path)
    a.add_argument("start")
    a.add_argument("end")
    args = a.parse_args()
    work = args.work
    priors = json.loads((work / "priors.json").read_text())
    m = pycolmap.Reconstruction(work / "sparse/level")

    def wanted(image):
        if not image.name.startswith("narrow/"):
            return False
        p = priors[image.name.split("/", 1)[1]]
        return (
            args.start <= datetime.fromtimestamp(p["wall"], TIMEZONE).strftime("%H:%M") < args.end
        )

    keep = {image.frame_id for image in m.images.values() if wanted(image)}
    for frame_id in list(m.reg_frame_ids()):
        if frame_id not in keep:
            m.deregister_frame(frame_id)
    selected = work / "splat/selected"
    selected.mkdir(parents=True, exist_ok=True)
    m.write(selected)
    out = work / "splat/data"
    pycolmap.undistort_images(out, selected, work / "images")
    names = sorted(i.name for i in m.images.values() if i.has_pose)
    test = names[::HOLDOUT]
    (work / "splat/split.json").write_text(json.dumps({"train": sorted(set(names) - set(test)),
                                                       "test": test}))  # fmt: skip
    undistorted = pycolmap.Reconstruction(out / "sparse")
    undistorted.write_text(out / "sparse")  # text files for any reader
    camera = next(iter(undistorted.cameras.values()))
    print(len(names), "frames;", len(test), "held out;", camera.model.name,
          [round(x, 1) for x in camera.params], camera.width, camera.height,
          undistorted.num_points3D(), "points")  # fmt: skip


if __name__ == "__main__":
    main()
