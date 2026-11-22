"""The drivingbench_sandbox tools as Python functions for Prime Agent.

Same gateway calls, results and images as the MCP server (mcp/tools.py, kept equal by
tests/test_prime_agent.py), because Prime Agent's MCP bridge drops image results: observe
attaches the camera frames to the model's context itself. For code, read() is observe returning
the frames as images instead of attaching them, frames holds the latest observation's images and
history() replays every call as it came back. Past frames wait on disk and are decoded only when
looked at, so a loop reading many times a second keeps memory flat. Argument limits are the
producer's; an invalid command comes back as an error and nothing moves.
"""

import asyncio
import atexit
import base64
import io
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

GATEWAY = os.environ.get("DRIVINGBENCH_GATEWAY_URL", "http://127.0.0.1:8766")
CLIENT = "prime-agent"
ATTACHMENT = "application/vnd.prime-agent.attachment+json"
MAX_DIMENSION, MAX_DATA_CHARS = 1200, 350_000  # Prime Agent's attachment limits
SUMMARY = (
    "timestamp",
    "state",
    "speed_mps",
    "steering_percent",
    "remaining_s",
    "image_age_s",
    "camera_reason",
    "reason",
    "error",
)


def _stamped(value):
    return {"timestamp": datetime.now(UTC).isoformat(), **value}


def _summary(packet):
    summary = {key: packet[key] for key in SUMMARY if key in packet}
    if images := packet.get("images"):
        summary["cameras"] = [image["camera"] for image in images]
    return summary


def _request(method, path, body=None):
    headers = {"X-Client": CLIENT}
    data = None
    if method != "GET":
        headers |= {"X-Request-ID": uuid.uuid4().hex, "Content-Type": "application/json"}
        data = json.dumps(body).encode()
    try:
        request = urllib.request.Request(GATEWAY + path, data, headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read()
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("invalid response")
        if status >= 400 and "error" not in result:
            return {"error": "request_rejected"}
        return result
    except (OSError, ValueError):
        # Never repeat a write: the device may have accepted it already.
        return {"error": "observation_unavailable" if method == "GET" else "outcome_unknown"}


def _fit(image):
    """The frame within Prime Agent's attachment limits, shrunk only when it exceeds them."""
    from PIL import Image

    data = image["data"]
    with Image.open(io.BytesIO(base64.b64decode(data))) as frame:
        if max(frame.size) <= MAX_DIMENSION and len(data) <= MAX_DATA_CHARS:
            return image["mime_type"], data
        frame = frame.convert("RGB")
        limit = MAX_DIMENSION
        while True:  # like Prime Agent's attach_image: lower quality, then a smaller frame
            frame.thumbnail((limit, limit))
            for quality in (82, 72, 60, 48, 36):
                buffer = io.BytesIO()
                frame.save(buffer, "JPEG", quality=quality)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                if len(encoded) <= MAX_DATA_CHARS:
                    return "image/jpeg", encoded
            limit = int(limit * 0.75)


def _attach(image, observation_id):
    from rlm import emit

    mime_type, data = _fit(image)
    camera = image["camera"]
    raw_size = _image(base64.b64decode(image["data"])).size
    display_size = _image(base64.b64decode(data)).size
    payload = {
        "mime_type": mime_type,
        "data": data,
        "path": f"drivingbench://{observation_id}/{camera}",
    }
    text = (
        f"{observation_id}, {camera}; raw pixels {raw_size}, displayed pixels {display_size}. "
        f"Exact saved frames: drivingbench_sandbox.observation({observation_id!r}). "
        "Pixel coordinates refer to the stated image size."
    )
    emit({ATTACHMENT: payload, "text/plain": text})


def _image(raw):
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as frame:
        return frame.convert("RGB")


class _Frames(Mapping):
    """One observation's frames as camera → PIL image, each read from disk when looked up."""

    def __init__(self, paths):
        self._paths = paths
        self.observation_id = (
            f"observation-{next(iter(paths.values())).name.split('-')[0]}" if paths else None
        )

    def __getitem__(self, camera):
        image = _image(self._paths[camera].read_bytes())
        image.info.update(observation_id=self.observation_id, camera=camera)
        return image

    def metadata(self, camera):
        raw = self._paths[camera].read_bytes()
        _, displayed = _fit({"mime_type": "image/jpeg", "data": base64.b64encode(raw).decode()})
        return {
            "observation_id": self.observation_id,
            "camera": camera,
            "raw_size": _image(raw).size,
            "display_size": _image(base64.b64decode(displayed)).size,
        }

    def attach(self, camera):
        """Attach this exact saved frame, with its identity and coordinate dimensions."""
        _attach(
            {
                "camera": camera,
                "mime_type": "image/jpeg",
                "data": base64.b64encode(self._paths[camera].read_bytes()).decode(),
            },
            self.observation_id,
        )

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)


frames = {}  # the latest observation's camera frames as PIL images
_log = []  # (tool, arguments, result, frame files by camera) for every call, oldest first
_folder = None  # this kernel's frame files, removed when it exits


def _store(images):
    """Each base64 frame written to this kernel's folder as received; camera → path."""
    global _folder
    if _folder is None:
        _folder = Path(tempfile.mkdtemp(prefix="drivingbench-frames-"))
        atexit.register(shutil.rmtree, _folder, True)
    paths = {camera: _folder / f"{len(_log):06d}-{camera}" for camera in images}
    for camera, path in paths.items():
        path.write_bytes(base64.b64decode(images[camera]))
    return paths


def _logged(tool, arguments, result, images=None):
    global frames
    paths = None
    if images is not None:
        paths = _store(images)
        frames = _Frames(paths)
    _log.append((tool, arguments, result, paths))
    return result


async def _observe(tool):
    packet = await asyncio.to_thread(_request, "GET", "/api/observe")
    images = {image["camera"]: image["data"] for image in packet.get("images", [])}
    return packet, _logged(tool, {}, _stamped(_summary(packet)), images)


async def _command(tool, path, body):
    result = await asyncio.to_thread(_request, "POST", path, body)
    state = _stamped({key: result[key] for key in ("status", "error") if key in result})
    return json.dumps(_logged(tool, body, state), separators=(",", ":"))


async def observe() -> str:
    """See road images, actual speed/steering, and command state. Motion continues as you think."""
    packet, state = await _observe("observe")
    saved = frames
    for image in packet.get("images", []):
        _attach(image, saved.observation_id)
    return json.dumps(state, separators=(",", ":"))


async def read() -> tuple[dict, dict]:
    """observe() for code: the same state as a dict and each camera frame as a PIL image.

    Nothing is attached to your context, so loops can call it often; it is recorded like observe().
    """
    _, state = await _observe("read")
    return state, frames


def history() -> list[dict]:
    """Every call this session made, oldest first, as it came back to you.

    Each entry has timestamp, tool, arguments and result; observations also have frames
    (camera name to PIL image, read from disk when looked up). It persists across attempts in
    this session.
    """
    return [
        {"timestamp": result["timestamp"], "tool": tool, "arguments": arguments, "result": result}
        | ({} if paths is None else {"frames": _Frames(paths)})
        for tool, arguments, result, paths in _log
    ]


def observation(observation_id):
    """Retrieve the immutable saved frame set identified in an attached road image."""
    for *_, paths in _log:
        if paths is not None:
            saved = _Frames(paths)
            if saved.observation_id == observation_id:
                return saved
    raise KeyError(observation_id)


async def set_motion(
    direction: str,
    steering_percent: float,
    speed_mps: float,
    duration_s: float,
    reason: str = "",
) -> str:
    """Replace motion now. Percent is 0–100 of the shared wheel-angle setting; zero is straight.

    Speed 0.5–3.5 m/s, typically 1.0; above the operator's shared ceiling it is rejected. Duration 5–60 s. Duration is the lifetime, not a minimum replacement interval. It includes steering buildup and launch delay. Expiry starts braking, not
    guaranteed standstill. Compare timestamps across calls to learn your own latency.

    reason: optional short note on what this command is for; recorded, never executed.
    """  # noqa: E501 (the MCP description, verbatim)
    body = {
        "direction": direction,
        "steering_percent": steering_percent,
        "speed_mps": speed_mps,
        "duration_s": duration_s,
        "reason": reason,
    }
    return await _command("set_motion", "/api/motion", body)


async def stop_now(reason: str = "") -> str:
    """Cancel motion and begin braking. Observe to confirm standstill.

    reason: optional short note on why you are stopping; recorded, never executed.
    """
    return await _command("stop_now", "/api/stop", {"reason": reason})
