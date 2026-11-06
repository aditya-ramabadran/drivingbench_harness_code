"""Held-out splat quality: Brush's saved eval renders against the undistorted real frames.

    python -m drivingbench.sim.pipeline.score_splat WORK STEP

Reads WORK/splat/out/eval_STEP (Brush's renders of the held-out views) and WORK/splat/data's
frames; writes WORK/splat/heldout.json (PSNR, SSIM) and WORK/splat/heldout.jpg (real | splat).
"""

import json
import sys
from pathlib import Path

import numpy as np
from drivingbench.sim.faithfulness import scores
from PIL import Image


def main():
    splat, step = Path(sys.argv[1]) / "splat", sys.argv[2]
    renders, truth = splat / f"out/eval_{step}", splat / "data/images/narrow"
    rows, pairs = [], []
    for render in sorted(renders.glob("*.png")):
        a = Image.open(render).convert("RGB")
        b = np.asarray(Image.open(truth / f"{render.stem}.jpg").convert("RGB").resize(a.size))
        rows.append(scores(np.asarray(a), b))
        pairs.append((b, np.asarray(a)))
    psnr, sim = np.array([r["psnr"] for r in rows]), np.array([r["ssim"] for r in rows])
    score = {"views": len(rows),
             "psnr_db": {"mean": round(psnr.mean(), 2), "p10": round(np.percentile(psnr, 10), 2),
                         "p90": round(np.percentile(psnr, 90), 2)},
             "ssim": {"mean": round(sim.mean(), 3), "p10": round(np.percentile(sim, 10), 3)}}  # fmt: skip
    (splat / "heldout.json").write_text(json.dumps(score, indent=1))
    print(json.dumps(score))
    picks = np.linspace(0, len(pairs) - 1, 4).astype(int)
    image = Image.fromarray(np.concatenate([np.concatenate(pairs[i], axis=1) for i in picks]))
    image.resize((image.width // 2, image.height // 2)).save(splat / "heldout.jpg", quality=85)


if __name__ == "__main__":
    main()
