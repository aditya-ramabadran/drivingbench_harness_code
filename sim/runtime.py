"""The simulator behind the unchanged producer: real time, real interfaces.

    uv run python -m drivingbench.sim.runtime [--placeholder] [--port 8877]
                                              [--state-dir runs/sim/state]

`device/service.py` runs as on the comma. `RealTimeNative` replaces the native socket client
(`call`, `status`, `on_status`), advancing `SimNative` on this machine's monotonic clock as the
native loop runs on the comma's. `SimCameras` replaces camerad: each frame shows the pose at its
capture time, stamped on the camera clock, so capture-anchored plans and image ages behave as on
the car. Frames come from the site's Gaussian splat with its cones (sim/render.py), or plain
placeholder frames. Clients reach it as they reach the comma's producer (the gateway's
`producer_url`, 127.0.0.1:8877). The course's drivable region is a wall: a step that would take
the body outside it does not happen, and the car stands where it was.
"""

from __future__ import annotations

import argparse
import io
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np
from drivingbench.device import cameras
from drivingbench.shared.contracts import PROTOCOL
from drivingbench.sim.logs import REPO
from PIL import Image

STEP_S = 0.005  # how often the simulation catches up with the clock
HEARTBEAT_S = 0.05  # NativeClient's status poll
NATIVE_FPS = 20  # the comma's camera rate, for frame ids
SIZE = (1344, 760)  # the comma's road-camera frames


@dataclass(frozen=True)
class Pose:
    t: float  # monotonic seconds
    x: float
    y: float
    heading: float  # radians, counterclockwise from +x
    speed: float
    steer_deg: float


class RealTimeNative:
    """SimNative on the monotonic clock with NativeClient's interface. `contains(car)`, when set,
    says whether a car pose is allowed; `edge_at` is when one was first refused since the reset."""

    def __init__(self, sim):
        self.sim, self.contains, self.edge_at = sim, None, None
        self.lock = threading.Lock()
        self.on_status = None
        self.poses: deque[Pose] = deque(maxlen=4000)  # 20 s at the stepping rate
        self.stopped = threading.Event()
        self._thread = None

    def _catch_up(self):
        car = self.sim.car
        before = car.x, car.y, car.heading
        self.sim.advance(time.monotonic())
        if self.contains and not self.contains(car):
            car.x, car.y, car.heading = before
            car.speed = car.accel = 0.0
            self.edge_at = self.edge_at or self.sim.time
        self.poses.append(Pose(self.sim.time, car.x, car.y, car.heading, car.speed, car.steer_deg))

    def call(self, body, *, timeout=None):
        with self.lock:
            self._catch_up()
            return self.sim.call(body)

    def status(self):
        with self.lock:
            return dict(self.sim.snapshot)

    def pose(self, at: float) -> Pose:
        """The pose at monotonic time `at` (the latest one at or before it)."""
        with self.lock:
            self._catch_up()
            return next((p for p in reversed(self.poses) if p.t <= at), self.poses[0])

    def reset(self):
        """Stopped at the start, as a new episode begins."""
        c = self.sim.bridge.controller
        self.call({"protocol": PROTOCOL, "request_id": uuid.uuid4().hex, "operation": "stop",
                   "boot_id": c.boot_id, "command_epoch": c.command_epoch})  # fmt: skip
        with self.lock:
            car = self.sim.car
            car.x = car.y = car.heading = car.speed = car.accel = 0.0
            car.accel_queue = deque([0.0] * len(car.accel_queue))
            self.poses.clear()
            self.edge_at = None
            self._catch_up()

    def start(self):
        def run():
            next_status = time.monotonic()
            while not self.stopped.wait(STEP_S):
                with self.lock:
                    self._catch_up()
                    snapshot = dict(self.sim.snapshot)
                if self.on_status and time.monotonic() >= next_status:
                    next_status += HEARTBEAT_S
                    self.on_status(snapshot)

        self._thread = threading.Thread(target=run, name="sim-native", daemon=True)
        self._thread.start()

    def close(self):
        self.stopped.set()
        if self._thread:
            self._thread.join(1)


def placeholder(pose: Pose, camera: str) -> np.ndarray:
    """A plain frame for running without the splat: sky above the horizon, asphalt below, shaded
    by heading so that turns show."""
    width, height = SIZE
    image = np.empty((height, width, 3), np.uint8)
    image[: height // 2] = (150, 180, 215)
    shade = int(90 + 30 * np.cos(pose.heading))
    image[height // 2 :] = (shade, shade, shade)
    return image


class SimCameras:
    """camerad for the simulator: each frame rendered at the pose of its capture time."""

    def __init__(self, native: RealTimeNative, render=placeholder):
        self.native, self.render = native, render
        self.errors: dict[str, str] = {}
        self.generation = f"sim{time.monotonic_ns():x}"

    def start(self):
        pass

    def close(self):
        if close := getattr(self.render, "close", None):
            close()

    def frame(self, camera: str, adjustments) -> cameras.Frame:
        """The view now, stamped now on the camera clock; rendering, like the comma's encoder, only
        delays when the frame is ready."""
        at, captured = time.monotonic(), cameras.camera_clock()
        jpeg = io.BytesIO()
        Image.fromarray(self.render(self.native.pose(at), camera)).save(jpeg, "JPEG", quality=92)
        key = f"{self.generation}-{round(at * NATIVE_FPS)}"
        frame = cameras.Frame(key, captured, jpeg.getvalue())
        return frame if adjustments is None else cameras.render(frame, adjustments)


def build() -> RealTimeNative:
    """A ready SimNative on the monotonic clock: openpilot's Model 3 with the steer ratio,
    stiffness, angle offset and roll the car learned at the site (vehicleParameters, median)."""
    from drivingbench.sim.car import SimNative  # puts openpilot on the path
    from opendbc.car.tesla.interface import CarInterface

    cp = CarInterface.get_non_essential_params("TESLA_MODEL_3")
    sim = SimNative(cp, steer_ratio=12.45, stiffness=0.99, angle_offset=-1.88, roll=0.05,
                    time=time.monotonic())  # fmt: skip
    return RealTimeNative(sim)


def simulator_app(state_dir, render=placeholder, course=None):
    """The producer with the simulator below it, plus operator-only episode routes:
    GET /sim/episode (v1.1 progress, cone contacts, off-course time, finish, when the drivable
    region's edge first stopped the car) and POST /sim/reset."""
    from drivingbench.device.service import create_app
    from drivingbench.sim.course import Course
    from drivingbench.sim.episode import Episode

    native = build()
    course = course or Course()
    native.contains = lambda car: course.contains(course.place(car))
    episode, lock = Episode(course), threading.Lock()
    app = create_app(state_dir, native=native, cameras=SimCameras(native, render), release="sim")

    def track():  # v1.1's 5 Hz
        while not native.stopped.wait(0.2):
            pose = native.pose(time.monotonic())
            with lock:
                episode.update(pose.t, course.place(pose), pose.speed)

    threading.Thread(target=track, name="sim-episode", daemon=True).start()

    @app.get("/sim/episode")
    def episode_state():
        with lock:
            return episode.state() | {"edge_at": native.edge_at}

    @app.post("/sim/reset")
    def reset():
        native.reset()
        with lock:
            episode.reset()
            return episode.state() | {"edge_at": native.edge_at}

    app.state.native = native
    return app


def main():
    import uvicorn

    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--port", type=int, default=8877)
    a.add_argument("--state-dir", default=REPO / "runs/sim/state")
    a.add_argument("--placeholder", action="store_true", help="plain frames instead of the splat")
    args = a.parse_args()
    render = placeholder
    if not args.placeholder:
        from drivingbench.sim.render import CourseCamera

        try:
            render = CourseCamera()
        except FileNotFoundError as error:
            raise SystemExit(f"{error}; or run with --placeholder for plain frames") from None
    uvicorn.run(simulator_app(args.state_dir, render), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
