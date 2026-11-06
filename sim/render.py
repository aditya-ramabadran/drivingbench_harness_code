"""The simulated road camera: the lot explorer's splat and cones, rendered in headless Chrome.

`ChromeView` serves the explorer and drives it in `?render=1` mode through the Chrome DevTools
protocol (Google Chrome at its macOS location, a throwaway profile). `CourseCamera` is a
`SimCameras` renderer: it places the simulated car on the site's course (sim/course.py) and renders
what its road camera sees. The mount is measured from the reconstruction: 1.24 m above the ground,
4.6° up, 1.8° right of the direction of travel, the road camera's 1,158.8 px focal length.
"""

from __future__ import annotations

import base64
import io
import json
import math
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from drivingbench.sim.course import SITE, SPLATS, Course, Ground
from drivingbench.sim.explorer.serve import Handler
from PIL import Image

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WIDTH, HEIGHT, FOCAL_PX = 1344, 760, 1158.8
CAMERA_HEIGHT_M, PITCH_DEG, YAW_DEG = 1.24, 4.6, -1.8
ROAD_FOV_DEG = math.degrees(2 * math.atan(HEIGHT / 2 / FOCAL_PX))
WIDE_FOV_DEG = 60.0  # the wide camera is a fisheye; a wider pinhole view stands in for it
BACKGROUND = (159, 180, 200)  # the explorer's scene background (0x9fb4c8): no splat drawn there
EXPOSURE_LUMA = 89.5  # the road camera's mean frame luma over the real attempts


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ChromeView:
    """Headless Chrome running the explorer in render mode; one view at a time. A GPU reset can
    take the page's WebGL context, after which it would draw nothing: a render that fails reloads
    the page, waits for the splat again and retries once."""

    def __init__(self, file: str = "full", settle: int = 1):
        if not SPLATS[file].exists():
            raise FileNotFoundError(
                f"the simulated camera renders the Gaussian splat {SITE['splats'][file]}, "
                "which is not in the repository: build it with the pipeline (docs/sim.md)"
            )
        if not Path(CHROME).exists():
            raise FileNotFoundError(f"the simulated camera needs Google Chrome at {CHROME}")
        self.settle, self.next_id = settle, 0
        self.lock, self.rendering = threading.Lock(), threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", free_port()), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.profile = tempfile.mkdtemp(prefix="dbsim-chrome-")
        debug = free_port()
        url = f"http://127.0.0.1:{self.server.server_port}/?render=1&file={file}"
        self.chrome = subprocess.Popen(
            [CHROME, "--headless=new", f"--remote-debugging-port={debug}",
             f"--user-data-dir={self.profile}", f"--window-size={WIDTH},{HEIGHT}",
             "--no-first-run", "--no-default-browser-check", "--hide-scrollbars",
             "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
             "--disable-backgrounding-occluded-windows", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        self.socket = self._connect(debug)
        self._ready()

    def _ready(self):
        deadline = time.monotonic() + 120
        while not self.evaluate("window.explorerReady === true"):
            if time.monotonic() > deadline:
                raise TimeoutError("the explorer did not load in headless Chrome")
            time.sleep(0.25)
        # The splat reaches the GPU some frames after loading (seconds for 3 M splats): render until
        # it covers the view, not the bare background, and two frames agree.
        view, previous = self._view(0.0, 0.0, CAMERA_HEIGHT_M, 0.0, 0.0, ROAD_FOV_DEG), None
        while True:
            frame = self._frame(view).astype(np.int16)
            bare = np.mean(np.abs(frame[: HEIGHT // 2] - BACKGROUND).max(2) <= 2)
            if bare < 0.5 and previous is not None and np.abs(frame - previous).mean() < 0.5:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("the splat did not appear in headless Chrome")
            previous = frame

    def _connect(self, debug: int):
        import httpx
        from websockets.sync.client import connect

        deadline = time.monotonic() + 30
        while True:
            try:
                pages = httpx.get(f"http://127.0.0.1:{debug}/json", timeout=1).json()
                page = next(p for p in pages if p["type"] == "page")
                return connect(page["webSocketDebuggerUrl"], max_size=64 * 2**20)
            except (httpx.HTTPError, StopIteration, OSError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)

    def evaluate(self, expression: str):
        """A JavaScript expression's value (promises awaited)."""
        with self.lock:
            self.next_id += 1
            self.socket.send(json.dumps({"id": self.next_id, "method": "Runtime.evaluate",
                                         "params": {"expression": expression, "awaitPromise": True,
                                                    "returnByValue": True}}))  # fmt: skip
            while True:
                reply = json.loads(self.socket.recv())
                if reply.get("id") == self.next_id:
                    break
        result = reply.get("result", {})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"].get("text", "render failed"))
        return result.get("result", {}).get("value")

    def render(
        self, east, north, up, heading, pitch, fov_deg, *, forward=None, top=None
    ) -> np.ndarray:
        """RGB of the view from (east, north, up), heading from east CCW, pitch up (radians);
        or an exact orientation, `forward` and `top` unit vectors in the course frame."""
        view = self._view(east, north, up, heading, pitch, fov_deg, forward, top)
        with self.rendering:
            try:
                return self._frame(view)
            except RuntimeError:
                self.evaluate("window.explorerReady = false; location.reload()")
                self._ready()
                return self._frame(view)

    def _view(self, east, north, up, heading, pitch, fov_deg, forward=None, top=None) -> dict:
        view = {"e": east, "n": north, "u": up, "heading": heading, "pitch": pitch,
                "fov": fov_deg, "width": WIDTH, "height": HEIGHT, "settle": self.settle}  # fmt: skip
        if forward is not None:
            view |= {"forward": list(map(float, forward)), "up": list(map(float, top))}
        return view

    def _frame(self, view: dict) -> np.ndarray:
        jpeg = base64.b64decode(self.evaluate(f"renderAt({json.dumps(view)})"))
        return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))

    def close(self):
        self.socket.close()
        self.chrome.terminate()
        self.chrome.wait(5)
        self.server.shutdown()
        shutil.rmtree(self.profile, ignore_errors=True)


def auto_expose(frame: np.ndarray) -> np.ndarray:
    """The road camera's auto-exposure: the frame scaled to the camera's mean-luma target. The splat
    has the day's average look; the camera brightens toward shade and darkens toward bright sky."""
    luma = frame.reshape(-1, 3).mean(0) @ (0.299, 0.587, 0.114)
    return np.clip(frame * (EXPOSURE_LUMA / max(luma, 1.0)), 0, 255).astype(np.uint8)


class CourseCamera:
    """A `SimCameras` renderer: the simulated car on the course, seen by its road camera."""

    def __init__(self, view=None, ground=None, course: Course | None = None):
        self.view, self.ground = view or ChromeView(), ground or Ground.load()
        self.course = course or Course()

    def __call__(self, pose, camera: str) -> np.ndarray:
        p = self.course.place(pose)
        up = float(self.ground(p.east, p.north)) + CAMERA_HEIGHT_M
        fov = WIDE_FOV_DEG if camera == "wide" else ROAD_FOV_DEG
        heading, pitch = p.heading + math.radians(YAW_DEG), math.radians(PITCH_DEG)
        return auto_expose(self.view.render(p.east, p.north, up, heading, pitch, fov))

    def close(self):
        self.view.close()
