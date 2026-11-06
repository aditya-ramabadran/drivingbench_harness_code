"""Native-camera-shaped input for capture/HTTP tests; no comma or model API is contacted."""

import queue
import threading
from types import SimpleNamespace

from drivingbench.device.cameras import Cameras, Worker, _capture_frames, camera_clock


class CameraSource:
    """A conflated 20 Hz NV12 source with real capture identities and delayed metadata."""

    def __init__(self, camera, stop, data, width, height, hz=20):
        self.camera, self.stop, self.data = camera, stop, data
        self.width, self.height, self.hz = width, height, hz
        self.origin = camera_clock()
        self.next_at, self.connected = self.origin, False
        self.frame_id = -1

    def is_connected(self):
        return self.connected

    def connect(self, _blocking):
        self.connected = True
        return True

    def recv(self, _timeout):
        if self.stop.wait(max(0, self.next_at - camera_clock())):
            return None
        now = camera_clock()
        self.frame_id = max(self.frame_id + 1, int((now - self.origin) * self.hz))
        self.next_at = self.origin + (self.frame_id + 1) / self.hz
        self.timestamp_eof = round(now * 1e9)
        self.timestamp_sof = self.timestamp_eof - 1_000_000
        return SimpleNamespace(
            width=self.width,
            height=self.height,
            stride=self.width,
            uv_offset=self.width * self.height,
            data=self.data,
            frame_id=self.frame_id,
        )

    def metadata(self):
        # camerad publishes metadata after VisionIPC, not atomically with it.
        if camera_clock() < self.timestamp_eof * 1e-9 + 0.002:
            return None
        return SimpleNamespace(
            valid=True,
            **{
                f"{self.camera}RoadCameraState": SimpleNamespace(
                    frameId=self.frame_id,
                    timestampSof=self.timestamp_sof,
                    timestampEof=self.timestamp_eof,
                )
            },
        )


class CaptureCameras(Cameras):
    """Exercise the production capture loop and cache, replacing only native input I/O."""

    def __init__(self, data, width, height):
        super().__init__()
        self.data, self.width, self.height = data, width, height
        self.failures = []

    def start(self):
        for camera in ("narrow", "wide"):
            out = queue.Queue(maxsize=2)
            source = CameraSource(camera, self.stop, self.data, self.width, self.height)

            def run(camera=camera, source=source, out=out):
                try:
                    _capture_frames(camera, lambda: source, source.metadata, out, self.stop)
                except Exception as error:
                    self.failures.append(error)
                    self.stop.set()

            worker = threading.Thread(target=run, daemon=True)
            self.workers[camera] = Worker(out, worker)
            worker.start()

    def _spawn(self, _camera, _out):
        raise AssertionError(f"Capture test worker exited: {self.failures}")

    def close(self):
        self.stop.set()
        for worker in self.workers.values():
            worker.process.join(3)
            assert not worker.process.is_alive()
        assert not self.failures


class ParkedNative:
    on_status = None

    def start(self):
        pass

    def close(self):
        pass

    def status(self):
        return dict(state="held", reason="native_disengaged", speed_mps=0, enabled=False)

    def call(self, _body):
        raise AssertionError("Camera verification must never send a motion command")
