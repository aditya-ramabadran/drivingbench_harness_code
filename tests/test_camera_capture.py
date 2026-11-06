import asyncio
import io
import queue
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from camera_source import CaptureCameras, ParkedNative
from drivingbench.device import cameras
from drivingbench.device.service import create_app as producer_app
from drivingbench.gateway.app import create_app as gateway_app
from fastapi.testclient import TestClient
from PIL import Image


class ScriptedSource:
    def __init__(self, entries, stop, now):
        self.entries, self.stop, self.now = iter(entries), stop, now
        self.connected = False

    def is_connected(self):
        return self.connected

    def connect(self, _blocking):
        self.connected = True
        return True

    def recv(self, _timeout):
        try:
            entry = next(self.entries)
        except StopIteration:
            self.stop.set()
            return None
        at, self.frame_id, self.timestamp_eof, self.valid = entry
        self.now[0] = at
        self.timestamp_sof = self.timestamp_eof - 1
        self.buffer = SimpleNamespace(frame_id=self.frame_id)
        return self.buffer

    def metadata(self):
        return SimpleNamespace(
            valid=self.valid,
            narrowRoadCameraState=SimpleNamespace(
                frameId=self.frame_id,
                timestampSof=self.timestamp_sof,
                timestampEof=self.timestamp_eof,
            ),
        )


def capture(monkeypatch, entries, encoder=None, source_type=ScriptedSource, wanted_until=None):
    now = [100.0]
    stop, out = threading.Event(), queue.Queue()
    source = source_type(entries, stop, now)
    monkeypatch.setattr(cameras, "camera_clock", lambda: now[0])
    monkeypatch.setattr(cameras, "encode_nv12", encoder or (lambda b: str(b.frame_id).encode()))
    cameras._capture_frames("narrow", lambda: source, source.metadata, out, stop, wanted_until)
    return list(out.queue)


def test_capture_preserves_native_rate_and_does_not_reencode_a_duplicate(monkeypatch):
    entries = []
    for i in range(20):
        row = (100 + i * 0.05, i, round((100 + i * 0.05) * 1e9), True)
        entries.extend([row, row])
    frames = capture(monkeypatch, entries)
    assert len(frames) == 20
    assert len({f.key for f in frames}) == 20
    assert [round(f.captured_at, 2) for f in frames] == [100 + i * 0.05 for i in range(20)]
    assert frames[-1].captured_at - frames[0].captured_at == pytest.approx(0.95)


def test_invalid_captures_are_not_labeled_fresh(monkeypatch):
    entries = [(100, 1, 100_000_000_000, False), (100.1, 2, 101_000_000_000, True)]
    assert capture(monkeypatch, entries) == []


def test_out_of_order_capture_cannot_replace_the_latest_frame(monkeypatch):
    frames = capture(
        monkeypatch,
        [(100, 7, 100_000_000_000, True), (100.1, 6, 99_900_000_000, True)],
    )
    assert len(frames) == 1 and frames[0].captured_at == 100


def test_shared_buffer_reuse_during_encoding_cannot_publish(monkeypatch):
    def overwritten(buffer):
        buffer.frame_id += 1
        return b"reused"

    assert capture(monkeypatch, [(100, 7, 100_000_000_000, True)], overwritten) == []


def test_shared_buffer_reuse_during_metadata_join_never_reaches_the_encoder(monkeypatch):
    class Reused(ScriptedSource):
        def metadata(self):
            event = super().metadata()
            self.buffer.frame_id += 1
            return event

    def encode(_buffer):
        raise AssertionError("A reused native buffer must not be encoded")

    assert (
        capture(
            monkeypatch,
            [(100, 7, 100_000_000_000, True)],
            encoder=encode,
            source_type=Reused,
        )
        == []
    )


def test_reconnection_does_not_reuse_a_cache_identity(monkeypatch):
    class Reconnecting(ScriptedSource):
        def metadata(self):
            event = super().metadata()
            self.connected = False
            return event

    frames = capture(
        monkeypatch,
        [(100, 7, 100_000_000_000, True), (100.1, 7, 100_100_000_000, True)],
        source_type=Reconnecting,
    )
    assert len(frames) == 2 and frames[0].key != frames[1].key
    assert frames[1].captured_at - frames[0].captured_at == pytest.approx(0.1)


def test_reconnection_rejects_older_eof_but_a_new_worker_accepts_a_new_clock(monkeypatch):
    class Reconnecting(ScriptedSource):
        def metadata(self):
            event = super().metadata()
            self.connected = False
            return event

    frames = capture(
        monkeypatch,
        [
            (100, 7, 100_000_000_000, True),
            (100.1, 1, 99_900_000_000, True),
            (100.2, 2, 100_200_000_000, True),
        ],
        source_type=Reconnecting,
    )
    assert [frame.captured_at for frame in frames] == pytest.approx([100, 100.2])
    restarted = capture(monkeypatch, [(5, 7, 5_000_000_000, True)])
    assert len(restarted) == 1 and restarted[0].captured_at == 5
    assert restarted[0].key != frames[0].key


def test_exception_recovery_does_not_forget_the_last_published_eof(monkeypatch):
    class Recovering(ScriptedSource):
        failed = False

        def recv(self, timeout):
            if getattr(self, "frame_id", None) == 7 and not self.failed:
                self.failed, self.connected = True, False
                raise RuntimeError("transport reset")
            return super().recv(timeout)

    values = capture(
        monkeypatch,
        [
            (100, 7, 100_000_000_000, True),
            (100.1, 1, 99_900_000_000, True),
            (100.2, 2, 100_200_000_000, True),
        ],
        source_type=Recovering,
    )
    assert values[1] == "transport reset"
    assert [
        frame.captured_at for frame in values if isinstance(frame, cameras.Frame)
    ] == pytest.approx([100, 100.2])


async def exercise_http_source(tmp_path, data, width, height, duration=2.4):
    """Actual capture/encoding/cache/producer/gateway; native I/O and networking are emulated."""
    source = CaptureCameras(data, width, height)
    producer = producer_app(tmp_path, native=ParkedNative(), cameras=source)
    samples = {name: [] for name in ("narrow", "wide")}
    with TestClient(producer):
        async with httpx.AsyncClient(
            base_url="http://producer", transport=httpx.ASGITransport(app=producer)
        ) as upstream:
            gateway = gateway_app(client=upstream)
            async with httpx.AsyncClient(
                base_url="http://127.0.0.1:8766", transport=httpx.ASGITransport(app=gateway)
            ) as client:
                response = await client.patch(
                    "/api/settings",
                    json={
                        "image_adjustments": {
                            "exposure_ev": 0.5,
                            "contrast": 1.1,
                            "saturation": 1.05,
                        }
                    },
                )
                assert response.status_code == 200
                # Warm both capture workers before starting the measurement window.
                await asyncio.sleep(0.3)
                stop_at = asyncio.get_running_loop().time() + duration

                async def sample(camera):
                    while asyncio.get_running_loop().time() < stop_at:
                        response = await client.get("/api/camera/" + camera)
                        assert response.status_code == 200
                        with Image.open(io.BytesIO(response.content)) as image:
                            assert image.size == (width, height)
                        samples[camera].append(
                            (
                                response.headers["X-Frame-ID"],
                                float(response.headers["X-Captured-At"]),
                                asyncio.get_running_loop().time(),
                            )
                        )
                        await asyncio.sleep(0.05)  # Probe faster than the five-Hz requirement.

                async def ui_load():
                    while asyncio.get_running_loop().time() < stop_at:
                        await client.get("/api/status")
                        await client.get("/api/observe")  # JPEG attachment + disk recording load.
                        await asyncio.sleep(0.1)

                await asyncio.gather(sample("narrow"), sample("wide"), ui_load())
    result = {}
    for name, observations in samples.items():
        captured = {}
        for key, at, received in observations:
            if key in captured:
                assert captured[key][0] == at  # Polls must not invent new capture times.
            else:
                captured[key] = (at, received)
        times = sorted(at for at, _ in captured.values())
        received = sorted(arrival for _, arrival in captured.values())
        assert len(times) == len(set(times))
        windows = np.arange(received[0], received[-1] - 1, 0.1)
        result[name] = dict(
            requests=len(observations),
            distinct_frames=len(captured),
            capture_span_s=times[-1] - times[0],
            distinct_hz=(len(captured) - 1) / (times[-1] - times[0]),
            max_capture_gap_s=float(max(np.diff(times))),
            min_distinct_delivered_in_1s=int(
                min(sum(start <= t < start + 1 for t in received) for start in windows)
            ),
        )
    return result


async def test_capture_http_path_sustains_five_distinct_frames_per_second_under_load(tmp_path):
    width, height = 640, 480
    rng = np.random.default_rng(42)
    data = rng.integers(32, 224, width * height * 3 // 2, dtype=np.uint8).tobytes()
    result = await exercise_http_source(tmp_path, data, width, height)
    for stats in result.values():
        assert stats["distinct_hz"] >= 5, result
        assert stats["min_distinct_delivered_in_1s"] >= 5, result
        assert stats["max_capture_gap_s"] < 0.3, result


def test_an_unread_camera_neither_converts_nor_encodes_until_someone_asks(monkeypatch):
    class Unread(ScriptedSource):
        def recv(self, timeout):
            raise AssertionError("an unread camera must not even receive frames")

    wanted = SimpleNamespace(value=0.0)  # demand expired long ago
    stop, now = threading.Event(), [100.0]
    source = Unread([], stop, now)
    monkeypatch.setattr(cameras, "camera_clock", lambda: now[0])
    waits = []

    def wait(seconds):
        waits.append(seconds)
        if len(waits) == 3:
            stop.set()
        return stop.is_set()

    monkeypatch.setattr(stop, "wait", wait)
    out = queue.Queue()
    cameras._capture_frames("narrow", lambda: source, source.metadata, out, stop, wanted)
    assert waits == [0.05] * 3 and out.empty()
    # A request renews demand: the same capture loop now encodes every validated frame.
    wanted.value = 1000.0
    entries = [(100 + i * 0.05, i, round((100 + i * 0.05) * 1e9), True) for i in range(5)]
    assert len(capture(monkeypatch, entries, wanted_until=wanted)) == 5


def test_a_request_renews_demand_and_waits_only_briefly_for_a_fresh_capture(monkeypatch):
    now = [50.0]
    monkeypatch.setattr(cameras, "camera_clock", lambda: now[0])
    monkeypatch.setattr(cameras.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    cache = cameras.Cameras()
    cache.wanted_until["narrow"] = SimpleNamespace(value=0.0)
    cache.put("narrow", cameras.Frame("stale", 40.0, b"old"))
    cache._demand("narrow")
    assert cache.wanted_until["narrow"].value == 50.0 + cameras.DEMAND_S
    assert now[0] == pytest.approx(50.0 + cameras.FRESH_WAIT_S, abs=0.03)  # gave up, kept old
    cache.put("narrow", cameras.Frame("fresh", now[0] - 0.1, b"new"))
    asked = now[0]
    cache._demand("narrow")
    assert now[0] == asked  # a fresh capture is served at once
