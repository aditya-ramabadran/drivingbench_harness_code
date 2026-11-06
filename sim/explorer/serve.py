"""Serve the lot explorer on this machine: the page, the site's splats and its course.

    uv run python -m drivingbench.sim.explorer.serve [--port 8890]   # http://127.0.0.1:8890

It serves the site's (sim/course.py `SITE`) course, cones, ground, drivable region and splats:
`clean` by default, `?file=full` for all of them, `?file=raw` for the splat with its real cones.
"""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from drivingbench.sim.course import BOUNDS, CONES, COURSE, GROUND, SITE, SPLATS

FILES = {
    "/": Path(__file__).with_name("index.html"),
    "/data/course.json": COURSE,
    "/data/cones.json": CONES,
    "/data/ground.json": GROUND,
    "/data/bounds.json": BOUNDS,
} | {f"/data/{role}.splat": path for role, path in SPLATS.items()}


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        return str(FILES.get(path.split("?", 1)[0], Path("/nonexistent")))

    def log_message(self, *args):  # quiet: the renderer fetches through this too
        pass


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--port", type=int, default=8890)
    args = a.parse_args()
    if not SPLATS["clean"].exists():
        raise SystemExit(
            f"the explorer shows the Gaussian splat {SITE['splats']['clean']}, which is not in the "
            "repository: build it with the pipeline (docs/sim.md)"
        )
    print(f"http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
