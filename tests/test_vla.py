import asyncio
import base64
import io
import json
import math

import httpx
import pytest
from drivingbench.gateway.setup import load_policies
from drivingbench.harness import vla
from drivingbench.shared.contracts import Chunk
from PIL import Image
from pydantic import ValidationError

URL = "http://127.0.0.1:9000/act"


def jpeg(width=1344, height=760, color=(40, 90, 140)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def quadrants():
    """A narrow frame whose sky band is green and whose road halves are red and blue."""
    image = Image.new("RGB", (1344, 760), (0, 200, 0))
    image.paste((220, 0, 0), (0, 2 * vla.ROAD_TOP, 672, 760))
    image.paste((0, 0, 220), (672, 2 * vla.ROAD_TOP, 1344, 760))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def decoded(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def response(age, *, captured_at=1000.0, content=None):
    return httpx.Response(
        200,
        content=content or jpeg(),
        headers={
            "X-Image-Age-S": str(age),
            "X-Frame-ID": "source-1",
            "X-Captured-At": str(captured_at),
        },
    )


def test_shape_drops_elapsed_steps_and_makes_every_speed_executable():
    chunk = [[0.01 * i, v] for i, v in enumerate([1.0, 1.0, 0.1, 0.3, 2.9, 1.2, 1.4])]
    body = vla.shape(chunk, skip=2, ceiling=2.0)
    # stop, crawl floor, ceiling; five steps left hold the last until there are HOLD_STEPS
    assert body["speed_mps"] == [0.0, 0.5, 2.0, 1.2, 1.4, 1.4, 1.4, 1.4]
    assert body["curvature_inv_m"] == pytest.approx(
        [0.02, 0.03, 0.04, 0.05, 0.06, 0.06, 0.06, 0.06]
    )
    Chunk.model_validate(body)
    wild = vla.shape([[5, 1], [-5, 1]] * 40, skip=0, ceiling=3.5)
    assert len(wild["speed_mps"]) == 60 and max(map(abs, wild["curvature_inv_m"])) == 1
    for bad in (
        None,
        {"actions": [[0, 1]]},
        [],
        [[0, 1, 2]],
        [0.1, 1.0],
        [[math.nan, 1]],
        [[0, math.inf]],
        [[True, 1.0]],
        [["0.1", 1.0]],
        [[0.1, None]],
        [[0.0, 1.0], [0.0]],
    ):
        with pytest.raises(ValueError):
            vla.shape(bad, skip=0, ceiling=3.5)


def test_a_start_from_standstill_crawls_and_a_late_chunk_holds_its_last_step_briefly():
    start = vla.shape([[0.0, 0.0], [0.0, 0.1], [0.0, 0.2], [0.0, 0.35]], skip=1, ceiling=3.5)
    assert start["speed_mps"][:3] == [0.0, 0.5, 0.5]  # 0.1 still a stop; 0.2 and 0.35 crawl
    assert len(start["speed_mps"]) == vla.HOLD_STEPS
    late = vla.shape([[0.02 * i, 1.0 + 0.1 * i] for i in range(10)], skip=12, ceiling=3.5)
    assert late["curvature_inv_m"] == [pytest.approx(0.18)] * vla.HOLD_STEPS  # its final step
    assert late["speed_mps"] == [pytest.approx(1.9)] * vla.HOLD_STEPS
    too_late = vla.shape([[0.0, 1.0]] * 10, skip=10 + vla.LATE_STEPS, ceiling=3.5)
    assert too_late["speed_mps"] == []  # a stop


def test_late_plan_holds_only_within_the_one_second_allowance():
    chunk = [[0.01 * i, 1.0] for i in range(30)]
    for late in (0, 5, 9):  # through the last permitted step
        body = vla.shape(chunk, skip=len(chunk) + late, ceiling=1.0)
        assert body["curvature_inv_m"] == pytest.approx([0.29] * vla.HOLD_STEPS)
        assert body["speed_mps"] == [1.0] * vla.HOLD_STEPS
    assert vla.shape(chunk, skip=len(chunk) + vla.LATE_STEPS, ceiling=1.0)["speed_mps"] == []


def test_a_chunk_starts_at_capture_and_skips_what_inference_and_the_car_delay_consumed():
    clock, seen = Clock(), {}

    def predict(images, state, instruction):
        seen.update(size=decoded(images["front"]).size, state=state, instruction=instruction)
        clock.now += 0.25  # inference time
        return [[0.001 * i, 1.0] for i in range(30)]

    policy = vla.ChunkPolicy("demo", predict, clock)
    status = {"speed_mps": 0.9, "curvature_inv_m": -0.05, "settings": {"speed_limit_mps": 3.0}}
    path, body, text, evidence = asyncio.run(policy.act(status, {"front": response(0.35)}, "go", 4))
    assert seen == {"size": (vla.WIDTH, vla.HEIGHT), "state": [-0.05, 0.9], "instruction": "go"}
    assert evidence == {"predicted_steps": 30, "skipped_steps": 8}  # 0.35 + 0.25 + 0.2 lead
    # The whole plan goes, anchored to the frame's capture on the comma's camera clock less the
    # lead: the comma picks the row due, so transit cannot shift it. The laptop's estimate only
    # names the row it expects (step 8) and decides whether the plan is too late.
    assert path == "chunk" and body["curvature_inv_m"][0] == 0.0 and len(body["speed_mps"]) == 30
    assert body["anchor_s"] == pytest.approx(1000.0 - vla.ACTUATOR_LEAD_S)
    assert body["reason"].startswith("demo r4:")
    Chunk.model_validate(body)
    assert "skipped 8" in text and "curvature +0.008/m" in text


def test_two_cameras_reach_the_policy_and_the_older_frame_sets_the_skip():
    seen = {}

    def predict(images, state, instruction):
        seen.update({name: decoded(image).size for name, image in images.items()})
        return [[0.0, 1.0]] * 32

    predict.cameras = ("front", "wide")
    policy = vla.ChunkPolicy("two", predict, Clock())
    frames = {"front": response(0.2), "wide": response(0.6, captured_at=999.6)}
    _, body, _, evidence = asyncio.run(policy.act({}, frames, "", 1))
    assert seen == {"front": (vla.WIDTH, vla.HEIGHT), "wide": (vla.WIDTH, vla.HEIGHT)}
    assert evidence["skipped_steps"] == 8  # the 0.6 s wide frame plus the 0.2 s lead
    assert body["anchor_s"] == pytest.approx(999.6 - vla.ACTUATOR_LEAD_S)  # and it dates the plan


def test_the_road_views_are_the_band_below_the_sky_split_down_the_middle():
    narrow = response(0.1, content=quadrants())
    images, state = vla.policy_input(
        {}, dict.fromkeys(("front", "road_left", "road_right"), narrow)
    )
    assert state == [0.0, 0.0]
    whole = decoded(images["front"])
    assert whole.size == (vla.WIDTH, vla.HEIGHT)
    assert whole.getpixel((10, 10)) == pytest.approx((0, 200, 0), abs=8)
    expected = {"road_left": (220, 0, 0), "road_right": (0, 0, 220)}
    for name, color in expected.items():
        view = decoded(images[name])
        assert view.size == (vla.WIDTH // 2, vla.HEIGHT - vla.ROAD_TOP)
        for corner in ((4, 4), (view.width - 5, view.height - 5)):
            assert view.getpixel(corner) == pytest.approx(color, abs=8)


def test_a_prediction_that_arrives_after_its_own_chunk_brakes():
    clock = Clock()

    def slow(images, state, instruction):
        clock.now += 2.0
        return [[0.0, 1.0]] * 10

    path, body, text, _ = asyncio.run(
        vla.ChunkPolicy("late", slow, clock).act({}, {"front": response(0.1)}, "", 1)
    )
    assert path == "stop" and "expired during inference" in body["reason"]
    assert text.startswith("late by")


def test_a_marginal_late_plan_reports_holding_its_final_target():
    def predict(images, state, instruction):
        return [[0.02, 1.0]] * 30

    policy = vla.ChunkPolicy("held", predict, Clock())
    path, body, text, evidence = asyncio.run(policy.act({}, {"front": response(3.6)}, "", 1))
    assert path == "chunk" and len(body["speed_mps"]) == 30  # the comma holds its final row
    assert evidence["holding_last"] and "holding final target" in text
    expired = asyncio.run(policy.act({}, {"front": response(3.8)}, "", 1))
    assert expired[0] == "stop" and "holding_last" not in expired[3]


def server(reply):
    """A policy server: `reply(request body)` returns an httpx.Response or raises."""
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return reply(requests[-1])

    return httpx.MockTransport(handle), requests


def test_http_sends_each_view_as_base64_jpeg_with_the_state_and_instruction():
    transport, requests = server(lambda body: httpx.Response(200, json={"actions": [[0.01, 1]]}))
    predict = vla.HttpPolicy(URL, ("road_left", "road_right"), transport)
    policy = vla.ChunkPolicy("served", predict, Clock())
    status = {"curvature_inv_m": 0.03, "speed_mps": 1.2}
    narrow = response(0.1)
    path, body, _, _ = asyncio.run(
        policy.act(status, {"road_left": narrow, "road_right": narrow}, "follow the cones", 2)
    )
    (sent,) = requests
    assert set(sent) == {"images", "state", "instruction"}
    assert sent["state"] == [0.03, 1.2] and sent["instruction"] == "follow the cones"
    for name in ("road_left", "road_right"):
        image = decoded(base64.b64decode(sent["images"][name]))
        assert image.size == (vla.WIDTH // 2, vla.HEIGHT - vla.ROAD_TOP)
    assert path == "chunk" and body["speed_mps"] == [1.0] * vla.HOLD_STEPS
    policy.close()
    assert predict.client.is_closed


@pytest.mark.parametrize(
    "reply",
    [
        lambda body: httpx.Response(500, text="model crashed"),
        lambda body: httpx.Response(404, json={"actions": [[0.0, 1.0]]}),
        lambda body: httpx.Response(200, text="not json"),
        lambda body: httpx.Response(200, json=[[0.0, 1.0]]),
        lambda body: httpx.Response(200, json={"action": [[0.0, 1.0]]}),
        lambda body: httpx.Response(200, json={"actions": []}),
        lambda body: httpx.Response(200, json={"actions": [[0.0, 1.0, 2.0]]}),
        lambda body: httpx.Response(200, json={"actions": [0.0, 1.0]}),
        lambda body: httpx.Response(200, content=b'{"actions": [[NaN, 1.0]]}'),
        lambda body: httpx.Response(200, content=b'{"actions": [[0.0, Infinity]]}'),
        lambda body: httpx.Response(200, json={"actions": [["left", 1.0]]}),
        lambda body: (_ for _ in ()).throw(httpx.ReadTimeout("policy too slow")),
        lambda body: (_ for _ in ()).throw(httpx.ConnectError("no server")),
    ],
)
def test_a_bad_reply_raises_so_the_driver_sends_nothing(reply):
    transport, requests = server(reply)
    policy = vla.ChunkPolicy("served", vla.HttpPolicy(URL, transport=transport), Clock())
    with pytest.raises((httpx.HTTPError, ValueError)):
        asyncio.run(policy.act({}, {"front": response(0.1)}, "go", 1))
    with pytest.raises((httpx.HTTPError, ValueError)):
        policy.probe("go")
    assert len(requests) == 2


def test_the_probe_sends_black_views_of_every_camera_at_policy_size():
    transport, requests = server(lambda body: httpx.Response(200, json={"actions": [[0, 0]]}))
    cameras = ("front", "wide", "road_left", "road_right")
    vla.ChunkPolicy("served", vla.HttpPolicy(URL, cameras, transport)).probe("go straight")
    (sent,) = requests
    assert sent["state"] == [0.0, 0.0] and sent["instruction"] == "go straight"
    sizes = {name: decoded(base64.b64decode(data)).size for name, data in sent["images"].items()}
    road = (vla.WIDTH // 2, vla.HEIGHT - vla.ROAD_TOP)
    assert sizes == {"front": (vla.WIDTH, vla.HEIGHT), "wide": (vla.WIDTH, vla.HEIGHT)} | {
        "road_left": road,
        "road_right": road,
    }


def test_the_registry_is_the_operators_file_and_rejects_anything_unexpected(tmp_path):
    path = tmp_path / "policies.json"
    assert load_policies(path) == {}
    path.write_text(
        json.dumps(
            {
                "cones": {"url": URL, "objective": "Follow the cones."},
                "steer": {
                    "url": "https://policy.example/v1/act",
                    "cameras": ["road_left", "road_right"],
                    "steerable": True,
                },
            }
        )
    )
    policies = load_policies(path)
    assert policies["cones"].cameras == ("front",) and not policies["cones"].steerable
    assert policies["cones"].objective == "Follow the cones."
    assert policies["steer"].cameras == ("road_left", "road_right") and policies["steer"].steerable
    built = vla.build("steer", policies["steer"])
    assert built.commands == vla.COMMANDS and built.cameras == ("road_left", "road_right")
    assert vla.build("cones", policies["cones"]).commands is None
    good = {"url": URL}
    for bad in (
        [good],
        {"x": URL},
        {"x": {}},
        {"x": good | {"kind": "grpc"}},
        {"x": good | {"url": "ftp://policy"}},
        {"x": good | {"url": "policy:9000"}},
        {"x": good | {"cameras": []}},
        {"x": good | {"cameras": ["rear"]}},
        {"x": good | {"cameras": "front"}},
        {"x": good | {"steerable": "true"}},
        {"x": good | {"objective": 3}},
        {"x": good | {"api_key": "secret"}},
        {"": good},
        {"x" * 65: good},
    ):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValidationError):
            load_policies(path)
    path.write_text("{not json")
    with pytest.raises(ValidationError):
        load_policies(path)
