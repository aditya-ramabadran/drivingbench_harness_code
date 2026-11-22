"""The Prime Agent skill must behave exactly like the MCP tools, against a stub and the real stack."""

import asyncio
import base64
import inspect
import io
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from drivingbench.mcp.tools import make_server
from drivingbench.shared.contracts import INSTRUCTIONS
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp/prime_agent/src"))
import drivingbench_sandbox as skill  # noqa: E402


def jpeg(width, height, noise=False):
    buffer = io.BytesIO()
    image = Image.effect_noise((width, height), 90) if noise else Image.new("L", (width, height))
    image.convert("RGB").save(buffer, "JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode()


SMALL = jpeg(64, 40)


class Gateway:
    """A local gateway both clients really call, recording every request."""

    def __init__(self):
        self.requests, self.motion = [], (200, {"status": "accepted"})
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                gateway.requests.append(("GET", self.path, self.headers["X-Client"], None))
                images = [
                    {"camera": c, "mime_type": "image/jpeg", "data": SMALL}
                    for c in ("narrow", "wide")
                ]
                observed = {"timestamp": "t", "state": "held", "speed_mps": 0, "internal": 1}
                self.reply(200, observed | {"steering_percent": 0, "images": images})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                gateway.requests.append(("POST", self.path, self.headers["X-Client"], body))
                if not self.headers["X-Request-ID"]:
                    self.reply(400, {"error": "missing_request_id"})
                elif self.path == "/api/motion":
                    self.reply(*gateway.motion)
                else:
                    self.reply(200, {"status": "stopping"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def gateway(monkeypatch):
    served = Gateway()
    monkeypatch.setattr(skill, "GATEWAY", served.url)
    yield served
    served.server.shutdown()


@pytest.fixture
def emitted(monkeypatch):
    payloads = []
    monkeypatch.setitem(sys.modules, "rlm", SimpleNamespace(emit=payloads.append))
    return payloads


def without_time(text):
    return {k: v for k, v in json.loads(text).items() if k != "timestamp"}


async def both(served, name, args):
    async with httpx.AsyncClient(base_url=served.url) as client:
        mcp = await make_server(client, "prime-agent").call_tool(name, args)
    return mcp, await getattr(skill, name)(**args)


MOTION = {
    "direction": "left",
    "steering_percent": 30,
    "speed_mps": 1.0,
    "duration_s": 10,
    "reason": "ease left",
}


@pytest.mark.parametrize(
    "name,args", [("observe", {}), ("set_motion", MOTION), ("stop_now", {"reason": "cone"})]
)
def test_the_skill_sends_and_returns_exactly_what_the_mcp_does(gateway, emitted, name, args):
    mcp, mine = asyncio.run(both(gateway, name, args))
    first, second = gateway.requests
    assert first == second  # method, path, client label and body
    texts = [part for part in mcp if part.type == "text"]
    assert without_time(mine) == without_time(texts[0].text)
    images = [part.data for part in mcp if part.type == "image"]
    assert [p[skill.ATTACHMENT]["data"] for p in emitted] == images
    assert all(p[skill.ATTACHMENT]["mime_type"] == "image/jpeg" for p in emitted)


def test_read_is_observe_for_code_with_pixels_and_nothing_attached(gateway, emitted, monkeypatch):
    observed = without_time(asyncio.run(skill.observe()))
    emitted.clear()
    state, frames = asyncio.run(skill.read())
    first, second = gateway.requests
    assert first == second and emitted == []
    assert {k: v for k, v in state.items() if k != "timestamp"} == observed
    assert {name: frame.size for name, frame in frames.items()} == {
        "narrow": (64, 40),
        "wide": (64, 40),
    }
    assert frames["narrow"].getpixel((0, 0)) == (0, 0, 0)
    monkeypatch.setattr(skill, "GATEWAY", "http://127.0.0.1:1")
    state, frames = asyncio.run(skill.read())
    assert without_time(json.dumps(state)) == {"error": "observation_unavailable"} and frames == {}


def test_history_replays_every_call_as_returned_and_frames_is_the_latest(
    gateway, emitted, monkeypatch
):
    monkeypatch.setattr(skill, "_log", [])
    monkeypatch.setattr(skill, "frames", {})
    observed = json.loads(asyncio.run(skill.observe()))
    assert {name: frame.size for name, frame in skill.frames.items()} == {
        "narrow": (64, 40),
        "wide": (64, 40),
    }
    moved = json.loads(asyncio.run(skill.set_motion(**MOTION)))
    monkeypatch.setattr(skill, "GATEWAY", "http://127.0.0.1:1")
    failed, _ = asyncio.run(skill.read())
    assert skill.frames == {}
    first, second, third = skill.history()
    assert (first["tool"], first["arguments"], first["result"]) == ("observe", {}, observed)
    assert first["timestamp"] == observed["timestamp"] and first["frames"]["wide"].size == (64, 40)
    assert second == {
        "timestamp": moved["timestamp"],
        "tool": "set_motion",
        "arguments": MOTION,
        "result": moved,
    }
    assert (third["tool"], third["result"], third["frames"]) == ("read", failed, {})
    assert failed["error"] == "observation_unavailable"


def test_a_fast_reading_loop_keeps_frames_on_disk_and_decodes_only_what_is_looked_at(
    gateway, monkeypatch
):
    monkeypatch.setattr(skill, "_log", [])

    async def loop():
        for _ in range(50):
            await skill.read()

    asyncio.run(loop())
    files = [p for *_, paths in skill._log for p in paths.values()]  # memory holds only paths
    assert len(files) == 100 and all(isinstance(p, Path) and p.is_file() for p in files)
    decoded = []
    monkeypatch.setattr(
        skill, "_image", lambda raw: decoded.append(raw) or Image.new("RGB", (1, 1))
    )
    calls = skill.history()
    assert len(calls) == 50 and decoded == []  # nothing decoded until a frame is looked at
    assert set(calls[7]["frames"]) == {"narrow", "wide"} and decoded == []
    calls[7]["frames"]["wide"]
    assert decoded == [base64.b64decode(SMALL)]  # exactly the bytes the gateway sent


def test_descriptions_and_instructions_are_the_mcp_ones():
    async def tools():
        async with httpx.AsyncClient() as client:
            return {t.name: t.description for t in await make_server(client).list_tools()}

    for name, description in asyncio.run(tools()).items():
        assert inspect.cleandoc(getattr(skill, name).__doc__) == inspect.cleandoc(description)
    assert INSTRUCTIONS in (ROOT / "mcp/prime_agent/SKILL.md").read_text()


def test_large_frames_are_shrunk_to_fit_and_small_ones_pass_untouched():
    assert skill._fit({"mime_type": "image/jpeg", "data": SMALL}) == ("image/jpeg", SMALL)
    big = jpeg(1928, 1208, noise=True)
    assert len(big) > skill.MAX_DATA_CHARS
    mime, data = skill._fit({"mime_type": "image/jpeg", "data": big})
    with Image.open(io.BytesIO(base64.b64decode(data))) as image:
        assert max(image.size) <= 1200 and image.size[0] / image.size[1] == pytest.approx(
            1928 / 1208, 1e-2
        )
    assert mime == "image/jpeg" and len(data) <= skill.MAX_DATA_CHARS


def test_rejections_pass_through_and_writes_are_never_retried(gateway, emitted, monkeypatch):
    gateway.motion = (422, {"error": "invalid_request"})
    bad = MOTION | {"speed_mps": 9}
    assert without_time(asyncio.run(skill.set_motion(**bad))) == {"error": "invalid_request"}
    gateway.motion = (500, b"not json")
    assert without_time(asyncio.run(skill.set_motion(**MOTION))) == {"error": "outcome_unknown"}
    assert len([r for r in gateway.requests if r[1] == "/api/motion"]) == 2
    monkeypatch.setattr(skill, "GATEWAY", "http://127.0.0.1:1")
    assert without_time(asyncio.run(skill.observe())) == {"error": "observation_unavailable"}
    assert without_time(asyncio.run(skill.stop_now())) == {"error": "outcome_unknown"}
    assert emitted == []


def test_saved_frame_identity_survives_new_reads_and_attachment_reports_coordinate_sizes(
    gateway, emitted, monkeypatch
):
    monkeypatch.setattr(skill, "_log", [])
    large = jpeg(1928, 1208, noise=True)
    monkeypatch.setattr(sys.modules[__name__], "SMALL", large)
    _, seen = asyncio.run(skill.read())
    identity = seen.observation_id
    metadata = seen.metadata("narrow")
    assert metadata["raw_size"] == (1928, 1208)
    assert max(metadata["display_size"]) <= 1200
    monkeypatch.setattr(sys.modules[__name__], "SMALL", jpeg(64, 40))
    asyncio.run(skill.observe())
    assert skill.frames.observation_id != identity
    assert seen["narrow"].size == (1928, 1208)
    assert seen["narrow"].info["observation_id"] == identity
    assert skill.observation(identity)["narrow"].size == (1928, 1208)
    seen.attach("narrow")
    attached = emitted[-1]
    assert (
        identity in attached["text/plain"]
        and str(metadata["display_size"]) in attached["text/plain"]
    )
    assert attached[skill.ATTACHMENT]["path"] == f"drivingbench://{identity}/narrow"


async def test_the_skill_drives_the_real_gateway_producer_and_controller(
    tmp_path, emitted, monkeypatch
):
    """The stack tests/test_e2e.py builds: only the cameras and the car's readings are fixtures."""
    from drivingbench.device.ipc import NativeClient
    from drivingbench.device.service import create_app as producer_app
    from drivingbench.gateway.app import create_app as gateway_app
    from test_e2e import QUARTER, MemoryCameras, NativeLoop, http_server, wait_status

    monkeypatch.setattr(skill, "_log", [])
    with tempfile.TemporaryDirectory(prefix="db-prime-", dir="/tmp") as local:
        socket_path = Path(local) / "native.sock"
        with (
            NativeLoop(socket_path) as loop,
            http_server(
                producer_app(tmp_path, native=NativeClient(socket_path), cameras=MemoryCameras())
            ) as producer,
            http_server(gateway_app(producer)) as gateway,
        ):
            monkeypatch.setattr(skill, "GATEWAY", gateway)
            async with httpx.AsyncClient(base_url=gateway, timeout=5) as ui:
                await wait_status(ui, lambda state: state.get("gear") == "drive")
                loop.update(brake_pressed=False, enabled=True, longitudinal_active=True)
                await wait_status(ui, lambda state: state.get("enabled") is True)

                observed = json.loads(await skill.observe())
                assert observed["state"] == "held" and len(emitted) == 1  # narrow_only default
                state, frames = await skill.read()
                assert state["state"] == "held" and set(frames) == {"narrow"}

                command = dict(direction="left", steering_percent=50, speed_mps=0.6, duration_s=5)
                assert json.loads(await skill.set_motion(**command))["status"] == "accepted"
                moving = await wait_status(ui, lambda state: state.get("command") is not None)
                assert moving["command"]["target_angle_deg"] == 2 * QUARTER
                assert json.loads(await skill.stop_now(reason="cone"))["status"] == "stopping"
                await wait_status(ui, lambda state: state.get("command") is None)
                loop.update(brake_pressed=True)
                await wait_status(ui, lambda state: state.get("brake_pressed") is True)
                denied = json.loads(await skill.set_motion(**command))
                assert denied["error"] == "operator_brake"

                calls = (await ui.get("/api/history")).json()["calls"]
                assert {call["client"] for call in calls} == {"prime-agent"}
                assert [call["tool"] for call in skill.history()] == [
                    "observe",
                    "read",
                    "set_motion",
                    "stop_now",
                    "set_motion",
                ]
