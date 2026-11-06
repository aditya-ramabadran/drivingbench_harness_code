"""Byte-exact image adjustment and capture encoding, and adversarial rendering concurrency."""

import io
import os
import signal
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

import numpy as np
import pytest
from drivingbench.device import cameras
from drivingbench.shared.contracts import ImageSettings
from PIL import Image, ImageEnhance, ImageStat


def reference_rgb(y, uv):
    """camerad's full-range YUV matrix on separately sliced planes: an independent oracle."""
    u = uv[:, 0::2].repeat(2, 0).repeat(2, 1).astype(np.float32) - 128
    v = uv[:, 1::2].repeat(2, 0).repeat(2, 1).astype(np.float32) - 128
    rgb = np.stack((y + 1.13983 * v, y - 0.39465 * u - 0.58060 * v, y + 2.03211 * u), axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def pillow_enhance(image, settings):
    image = ImageEnhance.Brightness(image).enhance(2**settings.exposure_ev)
    image = ImageEnhance.Contrast(image).enhance(settings.contrast)
    return ImageEnhance.Color(image).enhance(settings.saturation)


def reference_adjust(jpeg, settings):
    if settings == ImageSettings():
        return jpeg
    with Image.open(io.BytesIO(jpeg)) as original:
        output = io.BytesIO()
        pillow_enhance(original.convert("RGB"), settings).save(
            output, "JPEG", quality=92, subsampling=0
        )
        return output.getvalue()


@pytest.mark.parametrize("exposure", [-2, -1.3, 0, 0.25, 1.3, 2, 3])
@pytest.mark.parametrize("contrast", [0.5, 0.999, 1, 1.16, 2])
def test_saved_adjustment_operations_preserve_rgb_at_clipping_and_rounding_boundaries(
    exposure, contrast
):
    data = np.random.default_rng(87).integers(0, 256, (256, 512, 3), dtype=np.uint8)
    image = Image.fromarray(data)
    for saturation in (0, 0.5, 1, 1.6, 2):
        settings = ImageSettings(exposure_ev=exposure, contrast=contrast, saturation=saturation)
        assert (
            cameras.enhance(image, settings).tobytes() == pillow_enhance(image, settings).tobytes()
        )


def test_contrast_lookup_covers_every_mean_and_channel_value():
    values = np.arange(256, dtype=np.uint8)
    colors = np.stack((values, values[::-1], values ^ 85), axis=1)
    for mean in range(256):
        data = np.full((256, 257, 3), mean, np.uint8)
        data[0, :256] = colors
        image = Image.fromarray(data)
        assert int(ImageStat.Stat(image.convert("L")).mean[0] + 0.5) == mean
        for factor in (0.5, 0.999, 1.16, 2):
            settings = ImageSettings(contrast=factor)
            assert (
                cameras.enhance(image, settings).tobytes()
                == pillow_enhance(image, settings).tobytes()
            )


@pytest.mark.parametrize("padding", [0, 16, 64])
@pytest.mark.parametrize(
    "settings", [ImageSettings(), ImageSettings(exposure_ev=1.3, contrast=1.16, saturation=1.6)]
)
def test_padded_capture_and_its_adjusted_jpeg_are_byte_exact(padding, settings):
    width, height = 1344, 760
    stride, offset = width + padding, (width + padding) * height + padding
    rng = np.random.default_rng(83)
    raw = rng.integers(0, 256, offset + stride * (height // 2), dtype=np.uint8).tobytes()
    buffer = SimpleNamespace(width=width, height=height, stride=stride, uv_offset=offset, data=raw)
    y = np.frombuffer(raw, np.uint8, stride * height).reshape(height, stride)[:, :width]
    uv = np.frombuffer(raw, np.uint8, stride * (height // 2), offset).reshape(height // 2, stride)[
        :, :width
    ]
    output = io.BytesIO()
    Image.fromarray(reference_rgb(y, uv)).save(output, "JPEG", quality=92)
    expected = reference_adjust(output.getvalue(), settings)
    assert cameras.adjust(cameras.encode_nv12(buffer), settings) == expected  # and its frame hash


@pytest.mark.parametrize(
    "width,height,stride,offset,size,error",
    [
        (0, 2, 2, 4, 8, "invalid_camera_layout"),
        (3, 2, 4, 8, 12, "invalid_camera_layout"),  # odd width cannot carry 2x2 chroma
        (4, 2, 2, 8, 12, "invalid_camera_layout"),  # stride narrower than the image
        (4, 2, 4, 4, 12, "invalid_camera_layout"),  # chroma overlapping luma
        (4, 2, 4, 8, 11, "truncated_camera_frame"),
    ],
)
def test_encoder_rejects_unsafe_native_layouts(width, height, stride, offset, size, error):
    buffer = SimpleNamespace(
        width=width, height=height, stride=stride, uv_offset=offset, data=bytes(size)
    )
    with pytest.raises(ValueError, match=error):
        cameras.encode_nv12(buffer)


def test_concurrent_identical_renders_share_one_result_and_one_failure_then_retry(monkeypatch):
    cache = cameras.Cameras()
    frame = cameras.Frame("native-42", 123.456, b"raw-image")
    cache.put("narrow", frame)
    settings = ImageSettings(exposure_ev=1)

    for fail in (True, False):
        entered, release, joined = threading.Event(), threading.Event(), threading.Event()
        count_lock = threading.Lock()
        waiting = 0
        calls = []

        class JoinedFuture(Future):
            def result(self, timeout=None):
                nonlocal waiting
                with count_lock:
                    waiting += 1
                    if waiting == 3:
                        joined.set()
                return super().result(timeout)

        def adjust(*_args):
            calls.append(1)
            entered.set()
            assert release.wait(2)
            if fail:
                raise ValueError("render failed")
            return b"adjusted-image"

        monkeypatch.setattr(cameras, "Future", JoinedFuture)
        monkeypatch.setattr(cameras, "adjust", adjust)
        with ThreadPoolExecutor(4) as pool:
            owner = pool.submit(cache.frame, "narrow", settings)
            try:
                assert entered.wait(2)
                waiters = [pool.submit(cache.frame, "narrow", settings) for _ in range(3)]
                assert joined.wait(2)  # All three actually joined while the owner was blocked.
                assert len(calls) == 1
            finally:
                release.set()
            if fail:
                for result in (owner, *waiters):
                    with pytest.raises(ValueError, match="render failed"):
                        result.result(timeout=2)
                assert not cache.cache
            else:
                rendered = owner.result(timeout=2)
                assert all(result.result(timeout=2) is rendered for result in waiters)
                assert rendered.captured_at == frame.captured_at
                assert rendered.key.startswith(frame.key + "-")
                assert cache.frame("narrow", settings) is rendered
                assert len(calls) == 1
        assert not cache.rendering and not cache.latest_render


@pytest.mark.parametrize("change", ["capture", "settings", "camera"])
def test_held_render_cannot_block_new_inputs_or_overwrite_newer_cache(monkeypatch, change):
    cache = cameras.Cameras()
    settings = ImageSettings(exposure_ev=1)
    cache.put("narrow", cameras.Frame("old", 10, b"old"))
    entered, release = threading.Event(), threading.Event()

    def adjust(jpeg, adjustments):
        if jpeg == b"old" and adjustments == settings:
            entered.set()
            assert release.wait(2)
        return jpeg + adjustments.model_dump_json().encode()

    monkeypatch.setattr(cameras, "adjust", adjust)
    with ThreadPoolExecutor(2) as pool:
        old = pool.submit(cache.frame, "narrow", settings)
        try:
            assert entered.wait(2)
            camera, new_settings = "narrow", settings
            if change == "capture":
                cache.put("narrow", cameras.Frame("new", 11, b"new"))
            elif change == "settings":
                new_settings = ImageSettings(contrast=1.2)
            else:
                camera = "wide"
                cache.put("wide", cameras.Frame("wide", 11, b"wide"))
            newer = pool.submit(cache.frame, camera, new_settings).result(timeout=2)
            assert not old.done()  # New work completed without waiting for the held render.
        finally:
            release.set()
        previous = old.result(timeout=2)
    assert cache.frame(camera, new_settings) is newer
    assert previous.captured_at == 10
    assert not cache.rendering and not cache.latest_render


@pytest.mark.parametrize("kill_worker", [False, True])
def test_spawned_render_workers_preserve_bytes_and_recover_from_process_failure(
    monkeypatch, kill_worker
):
    class CaptureProcess:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            pass

        def terminate(self):
            self.alive = False

    cache = cameras.Cameras()
    monkeypatch.setattr(cache, "_spawn", lambda *_args: CaptureProcess())
    data = io.BytesIO()
    Image.new("RGB", (1344, 760), (40, 90, 140)).save(data, "JPEG", quality=92)
    frame = cameras.Frame("native-capture", 100.5, data.getvalue())
    settings = ImageSettings(exposure_ev=1.3, contrast=1.16, saturation=1.6)
    expected = cameras.render(frame, settings)
    cache.put("narrow", frame)
    cache.start()
    try:
        pool = cache.render_pool
        if kill_worker:
            worker_pid = pool.submit(os.getpid).result(timeout=5)
            os.kill(worker_pid, signal.SIGKILL)
            with pytest.raises(BrokenProcessPool):
                cache.frame("narrow", settings)
            assert cache.render_pool is not pool
            assert not cache.rendering and not cache.latest_render
        assert cache.frame("narrow", settings) == expected
        assert cache.cache["narrow"][1] == expected
    finally:
        cache.close()
