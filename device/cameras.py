"""Independent road-camera workers and a last-good-frame cache. No motion dependencies."""

import hashlib
import io
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from uuid import uuid4

from drivingbench.shared.contracts import ImageSettings
from PIL import Image, ImageEnhance, ImageStat

# A camera's encoder runs only this long after a request for it. A request that finds an older
# frame waits up to FRESH_WAIT_S for a capture at most FRESH_S old.
DEMAND_S, FRESH_S, FRESH_WAIT_S = 2.0, 0.5, 1.0


def camera_clock():
    return time.clock_gettime(getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC))


@dataclass(frozen=True)
class Frame:
    key: str
    captured_at: float
    jpeg: bytes


def enhance(image, settings):
    """Pillow's Brightness, Contrast and Color, the first two as byte-identical lookups."""
    ramp = Image.frombytes("L", (256, 1), bytes(range(256)))
    brightness = ImageEnhance.Brightness(ramp).enhance(2**settings.exposure_ev)
    image = image.point(list(brightness.tobytes()) * 3)
    # Pillow Contrast rounds the mean of its L conversion before blending. Build the lookup
    # through Pillow itself so clipping, float precision and truncation stay byte-identical.
    mean = int(ImageStat.Stat(image.convert("L")).mean[0] + 0.5)
    contrast = Image.blend(Image.new("L", ramp.size, mean), ramp, settings.contrast)
    image = image.point(list(contrast.tobytes()) * 3)
    return ImageEnhance.Color(image).enhance(settings.saturation)


def adjust(jpeg, settings):
    if settings == ImageSettings():
        return jpeg
    with Image.open(io.BytesIO(jpeg)) as original:
        image = enhance(original.convert("RGB"), settings)
        result = io.BytesIO()
        image.save(result, format="JPEG", quality=92, subsampling=0)
        return result.getvalue()


def render(frame, settings):
    """One canonical adjusted JPEG and identity, on either the comma or the laptop."""
    jpeg = adjust(frame.jpeg, settings)
    return Frame(frame.key + "-" + hashlib.sha256(jpeg).hexdigest()[:12], frame.captured_at, jpeg)


def encode_nv12(buffer):
    """Pinned camerad full-range YUV matrix; respect visible width and plane padding."""
    import numpy as np

    width, height, stride, offset = buffer.width, buffer.height, buffer.stride, buffer.uv_offset
    if not (
        0 < width * height <= 4_000_000
        and width % 2 == height % 2 == 0
        and stride >= width
        and offset >= stride * height
    ):
        raise ValueError("invalid_camera_layout")
    raw = bytes(buffer.data)
    if len(raw) < offset + stride * (height // 2):
        raise ValueError("truncated_camera_frame")
    y = np.frombuffer(raw, np.uint8, stride * height).reshape(height, stride)[:, :width]
    uv = np.frombuffer(raw, np.uint8, stride * (height // 2), offset).reshape(height // 2, stride)
    u = uv[:, :width:2].repeat(2, 0).repeat(2, 1).astype(np.float32) - 128
    v = uv[:, 1:width:2].repeat(2, 0).repeat(2, 1).astype(np.float32) - 128
    rgb = np.stack((y + 1.13983 * v, y - 0.39465 * u - 0.58060 * v, y + 2.03211 * u), axis=2)
    output = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(output, "JPEG", quality=92)
    return output.getvalue()


def _latest(out, value):
    """Publish the newest value, evicting the oldest when the queue is full."""
    try:
        out.put_nowait(value)
    except queue.Full:
        try:
            out.get_nowait()
        except queue.Empty:
            return
        try:
            out.put_nowait(value)
        except queue.Full:
            pass


def _match_metadata(receive, topic, valid, frame_id, timestamps, stop):
    """Join independent image/metadata streams without accepting an unvalidated frame."""
    deadline = time.monotonic() + 0.05
    while not stop.is_set():
        event = receive()
        if event is not None:
            state = getattr(event, topic)
            valid[state.frameId] = (state.timestampSof, state.timestampEof) if event.valid else None
            while len(valid) > 64:
                valid.popitem(last=False)
        if frame_id in valid:
            return valid[frame_id] == timestamps
        if time.monotonic() >= deadline:
            return False
        if event is None:
            stop.wait(0.001)
    return False


def _capture_frames(camera, make_client, receive_metadata, out, stop, wanted_until=None):
    """Encode validated native captures as they arrive, while someone reads this camera.

    Capture identity and time come from camerad, never the request or encoding clock.
    The EOF watermark spans reconnects; a new worker starts a new clock domain.
    """
    topic = f"{camera}RoadCameraState"
    last_eof = 0
    while not stop.is_set():
        client = make_client()
        valid = OrderedDict()
        generation = uuid4().hex
        try:
            while not stop.is_set():
                if wanted_until is not None and camera_clock() >= wanted_until.value:
                    stop.wait(0.05)  # nobody is reading this camera: no conversion or JPEG
                    continue
                if not client.is_connected():
                    if not client.connect(False):
                        stop.wait(0.1)
                        continue
                    valid.clear()
                    generation = uuid4().hex
                buffer = client.recv(100)
                if buffer is None:
                    continue
                frame_id, sof, eof = client.frame_id, client.timestamp_sof, client.timestamp_eof
                # CameraState validity is authoritative; this camerad leaves VIP.valid unset.
                if (
                    not 0 < sof <= eof <= camera_clock() * 1e9
                    or eof <= last_eof
                    or buffer.frame_id != frame_id
                ):
                    continue
                if not _match_metadata(receive_metadata, topic, valid, frame_id, (sof, eof), stop):
                    continue
                if buffer.frame_id != frame_id:
                    continue
                jpeg = encode_nv12(buffer)
                if buffer.frame_id != frame_id:
                    continue  # Shared buffer was reused during the copy.
                _latest(out, Frame(f"{generation}-{frame_id}", eof * 1e-9, jpeg))
                last_eof = eof
        except Exception as error:
            _latest(out, str(error))
            stop.wait(0.5)
        finally:
            del client


def _capture(camera, out, stop, parent_pid, wanted_until):
    # Native VisionIPC can hold the GIL: each road stream has a separate process.
    if os.name == "posix" and hasattr(time, "CLOCK_BOOTTIME"):
        import ctypes
        import signal

        ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG: die with parent
        if os.getppid() != parent_pid:
            return
    from msgq.visionipc import VisionIpcClient
    from openpilot.cereal import messaging
    from openpilot.cereal.visionipc import VisionStreamType

    events = messaging.sub_sock(f"{camera}RoadCameraState")
    stream = getattr(VisionStreamType, f"VISION_STREAM_{camera.upper()}_ROAD")
    _capture_frames(
        camera,
        lambda: VisionIpcClient("camerad", stream, True),
        lambda: messaging.recv_one_or_none(events),
        out,
        stop,
        wanted_until,
    )


@dataclass
class Worker:
    out: object
    process: object
    restarts: int = 0
    retry_at: float = 0.0


class Cameras:
    RESPAWN_MAX_DELAY_S = 30.0

    def __init__(self):
        self.frames = {}
        self.errors = {}
        self.cache = {}  # camera -> ((capture key, settings), rendered frame)
        self.rendering = {}  # in-flight renders by (camera, key), joined by identical requests
        self.latest_render = {}  # camera -> its newest render, the only one that may fill cache
        self.lock = threading.Lock()
        self.workers = {}
        self.wanted_until = {}
        self.context = mp.get_context("spawn")
        self.stop = self.context.Event()
        self.render_pool = None

    def _renderer(self):
        # Pillow's blend/encode work can hold the GIL. Keep it out of the HTTP/native
        # heartbeat process; requests still share one job per capture/settings identity.
        return ProcessPoolExecutor(2, mp_context=self.context)

    def _spawn(self, camera, out):
        process = self.context.Process(
            target=_capture,
            args=(camera, out, self.stop, os.getpid(), self.wanted_until[camera]),
            daemon=True,
        )
        process.start()
        return process

    def start(self):
        self.render_pool = self._renderer()
        for camera in ("narrow", "wide"):
            self.wanted_until[camera] = self.context.Value("d", 0.0, lock=False)
            out = self.context.Queue(maxsize=2)
            self.workers[camera] = Worker(out, self._spawn(camera, out))

    def put(self, camera, frame):
        """Also used by deterministic camera adapters in tests."""
        with self.lock:
            self.frames[camera] = frame

    def _poll(self, name, worker):
        if not worker.process.is_alive():
            self.errors[name] = "camera_worker_exited"
            now = camera_clock()
            if now >= worker.retry_at:  # Respawn with capped exponential backoff.
                worker.restarts += 1
                worker.retry_at = now + min(self.RESPAWN_MAX_DELAY_S, 2.0**worker.restarts)
                worker.process = self._spawn(name, worker.out)
        while True:
            try:
                value = worker.out.get_nowait()
            except queue.Empty:
                break
            if isinstance(value, Frame):
                self.frames[name] = value
                self.errors.pop(name, None)
            else:
                self.errors[name] = value

    def _demand(self, camera):
        """Keep this camera's encoder running; after it idled, wait briefly for a new capture."""
        wanted = self.wanted_until.get(camera)
        if wanted is None:
            return
        asked = camera_clock()
        wanted.value = asked + DEMAND_S
        while True:
            with self.lock:
                for name, worker in self.workers.items():
                    self._poll(name, worker)
                frame = self.frames.get(camera)
            now = camera_clock()
            if (frame and now - frame.captured_at <= FRESH_S) or now >= asked + FRESH_WAIT_S:
                return
            time.sleep(0.02)

    def frame(self, camera, adjustments):
        self._demand(camera)
        with self.lock:
            for name, worker in self.workers.items():
                self._poll(name, worker)
            frame = self.frames.get(camera)
            if frame is None:
                return None
            key = (frame.key, adjustments.model_dump_json())
            cached = self.cache.get(camera)
            if cached and cached[0] == key:
                return cached[1]
            job = (camera, key)
            pending = self.rendering.get(job)
            owner = pending is None
            if owner:
                pending = self.rendering[job] = Future()
                self.latest_render[camera] = pending
            pool = self.render_pool
        # Join only an identical capture/settings render. Different frames/cameras can progress,
        # and no rendering or waiter holds the shared frame/status lock. Work stays in callers;
        # there is no background queue and only active requests retain in-flight results.
        if not owner:
            return pending.result()
        try:
            rendered = (
                render(frame, adjustments)
                if pool is None
                else pool.submit(render, frame, adjustments).result()
            )
        except BaseException as error:
            with self.lock:
                if (
                    pool is not None
                    and isinstance(error, BrokenProcessPool)
                    and self.render_pool is pool
                ):
                    self.render_pool = self._renderer()
                    pool.shutdown(wait=False, cancel_futures=True)
                del self.rendering[job]
                if self.latest_render.get(camera) is pending:
                    del self.latest_render[camera]
            pending.set_exception(error)
            raise
        with self.lock:
            if self.latest_render.get(camera) is pending:
                self.cache[camera] = (key, rendered)
                del self.latest_render[camera]
            del self.rendering[job]
        pending.set_result(rendered)
        return rendered

    def close(self):
        self.stop.set()
        for worker in self.workers.values():
            worker.process.join(timeout=1)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=1)
            worker.out.cancel_join_thread()
            worker.out.close()
        if self.render_pool is not None:
            self.render_pool.shutdown(wait=True, cancel_futures=True)
