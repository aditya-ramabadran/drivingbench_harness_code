import asyncio
import base64
import io
import json
import threading

import httpx
import pytest
from drivingbench.gateway.app import create_app
from drivingbench.harness import driver as fast
from drivingbench.harness import runner as runs
from drivingbench.harness import supervisor as chat_loop
from drivingbench.harness import vla
from drivingbench.shared.contracts import PROTOCOL, Chunk
from PIL import Image


def jpeg(width=1344, height=760):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 90, 90)).save(buffer, "JPEG")
    return buffer.getvalue()


FRAME = jpeg()


class Producer:
    """A fake comma: status, frames, observe, and every command or session call it accepts."""

    def __init__(self, reason=None, protocol=PROTOCOL):
        self.reason, self.protocol = reason, protocol
        self.commands, self.down, self.refuse_session = [], False, False
        self.cameras = []

    def handle(self, request):
        if self.down:
            raise httpx.ConnectError("comma unreachable")
        path = request.url.path
        if path == "/status":
            return httpx.Response(
                200,
                json={
                    "protocol": self.protocol,
                    "state": "executing",
                    "reason": self.reason,
                    "speed_mps": 0.8,
                    "steering_deg": 54,
                    "settings": {"max_steering_angle_deg": 180},
                },
            )
        if path in ("/camera/narrow", "/camera/wide"):
            self.cameras.append(path.rsplit("/", 1)[1])
            return httpx.Response(
                200,
                content=FRAME,
                headers={"Content-Type": "image/jpeg", "X-Captured-At": "1000.0"},
            )
        if path == "/observe":
            image = {
                "camera": "narrow",
                "mime_type": "image/jpeg",
                "data": base64.b64encode(FRAME).decode(),
            }
            return httpx.Response(
                200,
                json={
                    "timestamp": "2026-10-11T19:00:00+00:00",
                    "state": "executing",
                    "speed_mps": 0.8,
                    "images": [image],
                },
            )
        body = json.loads(request.content)
        self.commands.append((path.strip("/"), request.headers.get("x-client"), body))
        if path == "/session/start" and self.refuse_session:
            return httpx.Response(422, json={"detail": "invalid_session"})
        return httpx.Response(200, json={"status": "accepted" if path == "/motion" else "ok"})

    def sent(self, client, path=None):
        return [b for p, c, b in self.commands if c == client and (path is None or p == path)]


class Decider:
    """Returns the queued answers in turn, then repeats the last; records every request."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    async def __call__(self, model, state, questions):
        self.calls.append((model, state, questions))
        steering, speed = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        runner_up = "keep" if steering != "keep" else "straight"
        return {
            "steering": {"probabilities": {steering: 0.7, runner_up: 0.3}},
            "speed": {"probabilities": {speed: 0.8, "crawl" if speed != "crawl" else "slow": 0.2}},
        }


def tool_call(name, arguments=None, call_id=None):
    return {
        "id": call_id or name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {})},
    }


class Chat:
    """A scripted supervisor: each queued reply is a list of tool calls, or text that ends the turn."""

    def __init__(self, *replies):
        self.replies, self.bodies = list(replies), []

    async def __call__(self, body):
        self.bodies.append(json.loads(json.dumps(body)))
        reply = self.replies.pop(0) if self.replies else "done"
        message = (
            {"role": "assistant", "content": reply}
            if isinstance(reply, str)
            else {"role": "assistant", "content": None, "tool_calls": reply}
        )
        return {
            "choices": [{"message": message}],
            "usage": {"cost": 0.01},
            "service_tier": body.get("service_tier", "default"),
        }


START = {
    "supervisor": "openai/gpt-6-astra",
    "tier": "ultrafast",
    "driver": "typesafe/jev-1.14",
    "objective": "park in the blue box",
    "prompt": "PROMPT",
    "reflection": "REFLECT",
    "continuation": "CONTINUE",
}


@pytest.fixture(autouse=True)
def fast_ticks(monkeypatch):
    monkeypatch.setattr(fast.DecisionPolicy, "tick_s", 0.01)
    monkeypatch.setattr(vla.ChunkPolicy, "tick_s", 0.01)


async def until(predicate, timeout=2.0):
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


async def test_driver_accepts_a_slow_prediction_without_extending_its_motion(monkeypatch):
    monkeypatch.setattr(fast, "DECISION_TIMEOUT_S", fast.DECISION_TIMEOUT_S / 100)

    class SlowDecider(Decider):
        async def __call__(self, *args):
            await asyncio.sleep(0.08)  # eight seconds on the scaled timeout budget
            return await super().__call__(*args)

    producer = Producer()
    async with gateway(producer, SlowDecider(("straight", "slow")), None) as client:
        await client.post("/runner/start", json=START | {"supervisor": "none"})
        try:
            await until(lambda: producer.sent("driver", "motion"))
            assert producer.sent("driver", "motion")[0]["duration_s"] == 5.0
        finally:
            await client.post("/runner/stop", json={"outcome": "aborted"})


def gateway(producer, decider, chat, repo=None, policies=None):
    upstream = httpx.AsyncClient(
        base_url="http://device", transport=httpx.MockTransport(producer.handle)
    )
    app = create_app(
        client=upstream, runs=repo and repo / "runs", decider=decider, chat=chat, policies=policies
    )
    return httpx.AsyncClient(
        base_url="http://127.0.0.1:8766", transport=httpx.ASGITransport(app=app)
    )


async def test_a_run_labels_a_session_and_both_layers_act_through_the_attempts(tmp_path):
    producer, decider = Producer(), Decider(("left_60", "slow"))
    chat = Chat(
        [tool_call("observe")],
        [tool_call("set_brief", {"brief": "Turn left at the cone.", "reason": "first bend"})],
        "Briefed.",
    )
    async with gateway(producer, decider, chat, tmp_path) as client:
        started = (await client.post("/runner/start", json=START)).json()
        assert started["attempt"] == 1 and started["driver"]["state"] == "driving"
        label = producer.sent("runner", "session/start")[0]
        assert (
            label["model"] == "gpt-6-astra (ultrafast) + jev-1.14" and label["harness"] == "other"
        )
        await until(lambda: producer.sent("driver", "motion"))
        motion = producer.sent("driver", "motion")[0]
        assert (
            motion["direction"],
            motion["steering_percent"],
            motion["speed_mps"],
            motion["duration_s"],
        ) == ("left", 60, 1.0, 5.0)
        assert motion["reason"] == "driver r1: left_60 0.70, slow 0.80"
        model, state, questions = decider.calls[0]
        assert model == "typesafe/jev-1.14" and questions is fast.QUESTIONS
        assert "park in the blue box" in state[0] and "measured steering 30% left" in state[1]
        image = base64.b64decode(state[2]["image_url"]["url"].split(",", 1)[1])
        assert Image.open(io.BytesIO(image)).width <= fast.FRAME_WIDTH
        await until(lambda: len(chat.bodies) >= 3)
        status = (await client.get("/runner/status")).json()
        assert (
            status["driver"]["revision"] == 2
            and status["driver"]["brief"] == "Turn left at the cone."
        )
        assert status["supervisor"]["state"] == "waiting"
        first = chat.bodies[0]
        assert first["service_tier"] == "ultrafast" and first["reasoning"] == {"effort": "medium"}
        assert [t["function"]["name"] for t in first["tools"]] == [
            "observe",
            "set_motion",
            "stop_now",
            "set_brief",
        ]
        assert first["messages"] == [{"role": "user", "content": "PROMPT"}]
        observed = chat.bodies[1]["messages"]
        tool_result = json.loads(observed[2]["content"])
        assert tool_result["driver"]["brief"].startswith("Objective: park in the blue box")
        assert observed[3]["content"][1]["type"] == "image_url"  # images follow the tool result

        ended = (await client.post("/runner/end-attempt")).json()
        assert ended["between_attempts"] and producer.sent("driver", "stop")
        await until(
            lambda: any(
                m == {"role": "user", "content": "REFLECT"} for m in chat.bodies[-1]["messages"]
            )
        )
        assert (await client.post("/runner/end-attempt")).status_code == 409
        resumed = (await client.post("/runner/next-attempt")).json()
        assert resumed["attempt"] == 2 and resumed["driver"]["brief"].startswith("Objective:")
        await until(
            lambda: any(
                m == {"role": "user", "content": "CONTINUE"} for m in chat.bodies[-1]["messages"]
            )
        )
        stopped = (
            await client.post("/runner/stop", json={"outcome": "completed", "note": "fine"})
        ).json()
        assert not stopped["running"]
        assert producer.sent("runner", "session/end")[-1] == {
            "outcome": "completed",
            "note": "fine",
            "request_id": producer.sent("runner", "session/end")[-1]["request_id"],
        }
        run = started["run"]
    kinds = [
        json.loads(line)["kind"]
        for line in (tmp_path / "runs/runner" / run / "events.jsonl").read_text().splitlines()
    ]
    for kind in (
        "start",
        "session_start",
        "attempt_start",
        "decision",
        "supervisor_tool",
        "brief",
        "attempt_end",
        "stop",
    ):
        assert kind in kinds
    assert kinds.count("attempt_start") == 2


async def test_supervisor_takeover_pauses_the_driver_until_the_command_ends_or_a_new_brief():
    producer, decider = Producer(), Decider(("straight", "crawl"))
    chat = Chat(
        [
            tool_call(
                "set_motion",
                {
                    "direction": "right",
                    "steering_percent": 100,
                    "speed_mps": 1,
                    "duration_s": 30,
                    "reason": "hard right now",
                },
            )
        ],
        "Took over.",
    )
    async with gateway(producer, decider, chat) as client:
        await client.post("/runner/start", json=START | {"tier": "normal"})
        await until(lambda: producer.sent("supervisor", "motion"))
        status = (await client.get("/runner/status")).json()
        assert (status["driver"]["state"], status["driver"]["reason"]) == (
            "paused",
            "supervisor_command",
        )
        assert "service_tier" not in chat.bodies[0]
        sent = len(producer.sent("driver"))
        await asyncio.sleep(0.1)
        assert len(producer.sent("driver")) == sent
        chat.replies = [
            [tool_call("set_brief", {"brief": "Carry on along the aisle."})],
            "Back to you.",
        ]
        await client.post("/runner/message", json={"text": "Continue"})
        await until(
            lambda: any(
                b["reason"].startswith("driver r2:") for b in producer.sent("driver", "motion")
            )
        )
        assert {"role": "user", "content": "Continue"} in chat.bodies[-1]["messages"]
        await client.post("/api/stop", json={}, headers={"X-Client": "UI"})
        status = (await client.get("/runner/status")).json()
        assert (status["driver"]["state"], status["driver"]["reason"]) == ("paused", "stopped")
        assert (await client.post("/runner/resume")).json()["driver"]["state"] == "driving"
        await client.post("/runner/stop", json={"outcome": "aborted"})


async def test_keep_holds_the_last_steering_and_stop_answers_brake():
    producer = Producer()
    decider = Decider(("right_30", "crawl"), ("keep", "crawl"), ("straight", "stop"))
    async with gateway(producer, decider, Chat("ok")) as client:
        await client.post("/runner/start", json=START)
        await until(lambda: producer.sent("driver", "stop"))
        await client.post("/runner/stop", json={"outcome": "aborted"})
    first, second = producer.sent("driver", "motion")[:2]
    assert (first["direction"], first["steering_percent"], first["speed_mps"]) == ("right", 30, 0.6)
    assert (second["direction"], second["steering_percent"]) == ("right", 30)
    assert producer.sent("driver", "stop")[0]["reason"].startswith(
        "driver r1: straight 0.70, stop 0.80"
    )


async def test_operator_states_and_protocol_mismatch_hold_the_driver():
    for producer, reason in (
        (Producer(reason="operator_brake"), "operator_brake"),
        (Producer(protocol=99), "protocol_mismatch"),
    ):
        decider = Decider(("straight", "crawl"))
        async with gateway(producer, decider, Chat("ok")) as client:
            started = await client.post("/runner/start", json=START)
            await asyncio.sleep(0.05)
            status = (await client.get("/runner/status")).json()
            if reason == "protocol_mismatch":
                assert started.status_code == 409
                assert "protocol_mismatch" in started.json()["error"]
                assert not status["running"]
            else:
                assert (status["driver"]["state"], status["driver"]["reason"]) == (
                    "waiting",
                    reason,
                )
            assert decider.calls == [] and producer.sent("driver", "motion") == []
            await client.post("/runner/stop", json={"outcome": "aborted"})


async def test_bad_answers_bad_tool_calls_and_model_errors_send_nothing(monkeypatch):
    producer = Producer()

    class Broken:
        async def __call__(self, model, state, questions):
            return {"steering": {"probabilities": {"sideways": 1.0}}, "speed": {"value": "crawl"}}

    chat = Chat(
        [
            tool_call("fly"),
            tool_call(
                "set_motion",
                {"direction": "left", "steering_percent": 30, "speed_mps": 0.4, "duration_s": 10},
                "slow",
            ),
            {
                "id": "x",
                "type": "function",
                "function": {"name": "set_brief", "arguments": "{oops"},
            },
        ],
        "ok",
    )
    async with gateway(producer, Broken(), chat) as client:
        await client.post("/runner/start", json=START)
        await until(lambda: len(chat.bodies) >= 2)
        results = [
            json.loads(m["content"]) for m in chat.bodies[1]["messages"] if m["role"] == "tool"
        ]
        assert [r["error"] for r in results] == [
            "unknown_tool",
            "invalid_arguments",
            "invalid_arguments",
        ]
        await client.post("/runner/stop", json={"outcome": "aborted"})
    assert producer.sent("driver", "motion") == [] and producer.sent("supervisor") == []

    class Failing:
        def __init__(self, failures):
            self.failures, self.calls = failures, 0

        async def __call__(self, body):
            self.calls += 1
            if self.calls <= self.failures:
                raise httpx.ConnectError("openrouter down")
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_loop, "RETRY_S", 0)
    passing = Failing(1)  # one failed completion is asked again and the turn goes on
    async with gateway(Producer(), Decider(("straight", "crawl")), passing) as client:
        await client.post("/runner/start", json=START)
        await until(lambda: passing.calls == 2)
        await asyncio.sleep(0.05)
        supervisor = (await client.get("/runner/status")).json()["supervisor"]
        assert supervisor["state"] == "waiting" and supervisor["transcript"][-1]["text"] == "ok"
        await client.post("/runner/stop", json={"outcome": "aborted"})
    down = Failing(2)  # failing twice ends the turn, and the Runner says it stalled and why
    async with gateway(Producer(), Decider(("straight", "crawl")), down) as client:
        await client.post("/runner/start", json=START)
        await until(lambda: down.calls == 2)
        await asyncio.sleep(0.05)
        supervisor = (await client.get("/runner/status")).json()["supervisor"]
        assert supervisor["transcript"][-1]["kind"] == "error"
        assert supervisor["state"] == "stalled" and "openrouter down" in supervisor["reason"]
        await client.post("/runner/message", json={"text": "Continue"})
        assert (await client.get("/runner/status")).json()["supervisor"]["state"] != "stalled"
        await client.post("/runner/stop", json={"outcome": "aborted"})


async def test_start_validation_and_missing_key(monkeypatch):
    monkeypatch.setattr(runs, "keychain_key", lambda *item: None)
    monkeypatch.setattr(runs, "keychain_has_key", lambda *item: False)
    async with gateway(Producer(), None, None) as client:
        assert (await client.post("/runner/start", json=START | {"prompt": ""})).status_code == 422
        missing = await client.post("/runner/start", json=START)
        assert (missing.status_code, missing.json()["error"]) == (503, "no_openrouter_key")
        assert (await client.post("/runner/stop", json={"outcome": "aborted"})).status_code == 409
        assert (await client.post("/runner/next-attempt")).status_code == 409
        options = (await client.get("/runner/options")).json()
        assert (
            options["key_present"] is False
            and options["keychain_item"] == "drivingbench-openrouter"
        )
        assert "openai/gpt-6-astra" in options["supervisors"] and "ultrafast" in options["tiers"]
        assert options["drivers"] == ["none", "typesafe/jev-1.14", "openai/gpt-6-luna-decisions"]
    async with gateway(Producer(), Decider(("straight", "crawl")), Chat("ok")) as client:
        await client.post("/runner/start", json=START)
        assert (await client.post("/runner/start", json=START)).status_code == 409
        assert (await client.post("/runner/stop", json={"outcome": "maybe"})).status_code == 422
        assert (await client.post("/runner/message", json={"text": " "})).status_code == 422
        await client.post("/runner/stop", json={"outcome": "aborted"})


async def test_stop_still_ends_the_run_when_the_producer_is_unreachable(tmp_path):
    producer = Producer()
    async with gateway(producer, Decider(("straight", "crawl")), Chat("ok"), tmp_path) as client:
        run = (await client.post("/runner/start", json=START)).json()["run"]
        await until(lambda: producer.sent("driver", "motion"))
        producer.down = True
        stopped = await client.post("/runner/stop", json={"outcome": "aborted"})
        assert stopped.status_code == 200 and not stopped.json()["running"]
        assert (await client.post("/runner/start", json=START)).json()[
            "error"
        ] == "producer_unavailable"
    last = json.loads(
        (tmp_path / "runs/runner" / run / "events.jsonl").read_text().splitlines()[-1]
    )
    assert last["kind"] == "stop" and last["session"] == {"error": "ConnectError"}
    assert len(list((tmp_path / "runs/runner").iterdir())) == 1  # the failed start created no run


class Gated(Decider):
    """Holds each decision until released, so a test can act while one is in flight."""

    def __init__(self, *answers):
        super().__init__(*answers)
        self.release, self.asked = asyncio.Event(), 0

    async def __call__(self, model, state, questions):
        self.asked += 1
        await self.release.wait()
        return await super().__call__(model, state, questions)


async def test_a_stop_or_brief_during_a_decision_drops_that_decision():
    producer, decider = Producer(), Gated(("left_100", "normal"))
    async with gateway(producer, decider, Chat("ok")) as client:
        await client.post("/runner/start", json=START)
        await until(lambda: decider.asked)
        await client.post("/api/stop", json={}, headers={"X-Client": "UI"})
        decider.release.set()
        await asyncio.sleep(0.1)
        assert producer.sent("driver", "motion") == []
        status = (await client.get("/runner/status")).json()["driver"]
        assert status["state"] == "paused" and status["decisions"][-1]["result"] == "discarded"
        await client.post("/runner/resume")
        await until(lambda: producer.sent("driver", "motion"))
        await client.post("/runner/stop", json={"outcome": "aborted"})

    producer, decider = Producer(), Gated(("straight", "crawl"))
    chat = Chat([tool_call("set_brief", {"brief": "New plan."})], "ok")
    async with gateway(producer, decider, chat) as client:
        await client.post("/runner/start", json=START)
        await until(lambda: len(chat.bodies) >= 2 and decider.asked)
        decider.release.set()
        await until(lambda: producer.sent("driver", "motion"))
        reasons = [b["reason"] for b in producer.sent("driver", "motion")]
        assert all(r.startswith("driver r2:") for r in reasons)
        await client.post("/runner/stop", json={"outcome": "aborted"})


async def test_a_refused_session_starts_nothing(tmp_path):
    producer = Producer()
    producer.refuse_session = True
    async with gateway(producer, Decider(("straight", "crawl")), Chat("ok"), tmp_path) as client:
        refused = await client.post("/runner/start", json=START)
        assert refused.status_code == 409
        assert refused.json()["error"] == "session_start_failed: invalid_session"
        await asyncio.sleep(0.05)
        assert not (await client.get("/runner/status")).json()["running"]
    assert [p for p, _, _ in producer.commands] == ["session/start"]
    assert not (tmp_path / "runs/runner").exists() or not any((tmp_path / "runs/runner").iterdir())


def test_most_likely_reads_probabilities_or_a_bare_choice():
    assert fast.most_likely({"probabilities": {"a": 0.2, "b": 0.8}}) == ("b", 0.8)
    assert fast.most_likely({"choice": "stop"}) == ("stop", 1.0)
    with pytest.raises(ValueError):
        fast.most_likely({})


async def test_a_chat_model_alone_drives_with_exactly_the_public_tools():
    producer = Producer()
    chat = Chat(
        [tool_call("observe")],
        [
            tool_call(
                "set_motion",
                {"direction": "left", "steering_percent": 30, "speed_mps": 1, "duration_s": 8},
            )
        ],
        "Driving.",
    )
    solo = START | {"driver": "none", "tier": "normal"}
    async with gateway(producer, None, chat) as client:
        started = await client.post("/runner/start", json=solo)
        assert started.status_code == 200 and started.json()["driver"]["state"] == "stopped"
        assert producer.sent("runner", "session/start")[0]["model"] == "gpt-6-astra (normal)"
        await until(lambda: producer.sent("supervisor", "motion"))
        tools = [t["function"]["name"] for t in chat.bodies[0]["tools"]]
        assert tools == ["observe", "set_motion", "stop_now"]  # no set_brief: nothing to brief
        observed = json.loads(
            next(m for m in chat.bodies[1]["messages"] if m["role"] == "tool")["content"]
        )
        assert "driver" not in observed and observed["state"] == "executing"
        assert producer.sent("driver") == []
        await until(lambda: len(chat.bodies) >= 3)
        ended = (await client.post("/runner/end-attempt")).json()
        assert ended["between_attempts"] and producer.sent("driver", "stop")  # the car stops
        assert {"role": "user", "content": "REFLECT"} in chat.bodies[-1]["messages"]
        assert not (await client.post("/runner/next-attempt")).json()["between_attempts"]
        assert {"role": "user", "content": "CONTINUE"} in chat.bodies[-1]["messages"]
        await client.post("/runner/stop", json={"outcome": "aborted"})
    neither = START | {"supervisor": "none", "driver": "none"}
    async with gateway(Producer(), None, None) as client:
        assert (await client.post("/runner/start", json=neither)).status_code == 422


async def test_gemini_needs_only_its_own_key_and_rejects_wrong_tiers_before_session(monkeypatch):
    asked = []
    monkeypatch.setattr(runs, "keychain_key", lambda *item: asked.append(item) or None)
    producer = Producer()
    async with gateway(producer, None, None) as client:
        body = START | {"supervisor": "gemini/gemini-5-preview", "driver": "none", "tier": "normal"}
        missing = await client.post("/runner/start", json=body)
        assert (missing.status_code, missing.json()["error"]) == (503, "no_gemini_key")
        assert (await client.get("/runner/gemini/models")).status_code == 503
        assert (
            await client.post("/runner/start", json=body | {"tier": "ultrafast"})
        ).status_code == 422
        assert (
            await client.post("/runner/start", json=body | {"supervisor": "gemini/../models"})
        ).status_code == 422
    assert asked == [("drivingbench-gemini",), ("drivingbench-gemini",)]
    assert producer.commands == []


async def test_gemini_uses_the_real_supervisor_loop_images_signatures_and_attempt_context(
    tmp_path, monkeypatch
):
    from drivingbench.harness import gemini

    requests = []
    signature = {"google": {"thought_signature": "opaque"}}
    move = {"direction": "straight", "steering_percent": 0, "speed_mps": 1, "duration_s": 5}

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "gemini-4-argon"
        assert body["service_tier"] == "priority"
        assert body["reasoning_effort"] == "medium"
        assert {t["function"]["name"] for t in body["tools"]} == {
            "observe",
            "set_motion",
            "stop_now",
        }
        message = {"role": "assistant", "content": "Finished turn.", "extra_content": signature}
        if len(requests) <= 2:
            call = tool_call(
                "observe" if len(requests) == 1 else "set_motion",
                {} if len(requests) == 1 else move,
            )
            call["extra_content"] = signature
            message["tool_calls"] = [call]
        return httpx.Response(
            200,
            json={
                "id": f"resp-{len(requests)}",
                "model": "gemini-4-argon",
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": 45, "completion_tokens": 7},
            },
            headers={"x-gemini-service-tier": "standard"},
        )

    api = gemini.Gemini("test-key", httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(gemini, "Gemini", lambda key: api)
    monkeypatch.setattr(
        runs,
        "keychain_key",
        lambda item: (
            "test-key"
            if item == gemini.KEYCHAIN_ITEM
            else pytest.fail("OpenRouter must not be used")
        ),
    )
    producer = Producer()
    async with gateway(producer, None, None, repo=tmp_path) as client:
        body = START | {"supervisor": "gemini/gemini-4-argon", "driver": "none", "tier": "fast"}
        response = await client.post("/runner/start", json=body)
        assert response.is_success
        await until(lambda: len(requests) == 3)
        status = (await client.get("/runner/status")).json()["supervisor"]
        assert status["cost_usd"] is None  # missing billing is not $0
        history = requests[1]["messages"]
        assert history[1]["tool_calls"][0]["extra_content"] == signature
        assert history[1]["extra_content"] == signature
        assert history[-1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        motions = producer.sent("supervisor", "motion")
        assert len(motions) == 1 and {k: motions[0][k] for k in move} == move
        await client.post("/runner/end-attempt")
        await until(lambda: len(requests) == 4)
        assert requests[-1]["messages"][-1] == {"role": "user", "content": "REFLECT"}
        await client.post("/runner/next-attempt")
        await until(lambda: len(requests) == 5)
        assert requests[-1]["messages"][-1] == {"role": "user", "content": "CONTINUE"}
        await client.post("/runner/stop", json={"outcome": "aborted"})
    assert api.client is None
    events = [
        json.loads(line)
        for line in next((tmp_path / "runs/runner").glob("*/events.jsonl")).read_text().splitlines()
    ]
    responses = [e for e in events if e["kind"] == "supervisor_response"]
    assert responses[0]["provider"] == "Google Gemini API"
    assert responses[0]["requested_service_tier"] == "priority"
    assert responses[0]["service_tier"] == "standard"
    assert responses[0]["usage"]["prompt_tokens"] == 45


async def test_end_attempt_stops_after_a_chat_command_already_in_flight():
    """A command past the attempt gate lands before End attempt's stop, and none after."""

    class Slow(Producer):
        def __init__(self):
            super().__init__()
            self.arrived, self.release = asyncio.Event(), asyncio.Event()

        async def handle(self, request):
            if request.url.path == "/motion" and request.headers.get("x-client") == "codex":
                self.arrived.set()
                await self.release.wait()  # still on its way to the comma
            return super().handle(request)

    producer = Slow()
    move = {"direction": "left", "steering_percent": 30, "speed_mps": 1, "duration_s": 60}
    chat = Chat("Driving.", "Reflected.")
    async with gateway(producer, None, chat) as client:
        await client.post("/runner/start", json=START | {"driver": "none", "tier": "normal"})
        await until(lambda: len(chat.bodies) >= 1)
        codex = {"X-Client": "codex"}
        moving = asyncio.create_task(client.post("/api/motion", json=move, headers=codex))
        await producer.arrived.wait()
        ending = asyncio.create_task(client.post("/runner/end-attempt"))
        await asyncio.sleep(0.05)
        assert not ending.done()  # End attempt waits for the command in flight
        producer.release.set()
        assert (await moving).is_success and (await ending).json()["between_attempts"]
        order = [(p, c) for p, c, _ in producer.commands if p in {"motion", "stop"}]
        assert order.index(("motion", "codex")) < order.index(("stop", "driver"))
        late = await client.post("/api/motion", json=move, headers=codex)
        assert late.status_code == 409 and len(producer.sent("codex", "motion")) == 1
        await client.post("/runner/stop", json={"outcome": "aborted"})


async def test_between_attempts_only_the_operator_moves_the_car():
    producer = Producer()
    move = {"direction": "left", "steering_percent": 30, "speed_mps": 1, "duration_s": 8}
    chat = Chat("Driving.", [tool_call("set_motion", move), tool_call("stop_now")], "Reflected.")
    solo = START | {"driver": "none", "tier": "normal"}
    async with gateway(producer, None, chat) as client:
        await client.post("/runner/start", json=solo)
        await until(lambda: len(chat.bodies) >= 1)
        await client.post("/runner/end-attempt")
        await until(lambda: len(chat.bodies) >= 3)  # the reflection turn tried to drive
        refused = json.loads(chat.bodies[2]["messages"][-2]["content"])
        assert (
            refused["error"] == "between_attempts" and producer.sent("supervisor", "motion") == []
        )
        assert producer.sent("supervisor", "stop")  # stopping is always allowed
        for client_name, allowed in (("claude", False), ("codex", False), (None, True)):
            headers = {"X-Client": client_name} if client_name else {}
            sent = await client.post("/api/motion", json=move, headers=headers)
            assert (sent.status_code == 409) == (not allowed), client_name
        assert (await client.post("/api/stop", json={}, headers={"X-Client": "claude"})).is_success
        await client.post("/runner/next-attempt")
        assert (
            await client.post("/api/motion", json=move, headers={"X-Client": "claude"})
        ).is_success
        await client.post("/runner/stop", json={"outcome": "aborted"})
        assert (
            await client.post("/api/motion", json=move, headers={"X-Client": "codex"})
        ).is_success


GOOD = [[0.02, 1.0]] * 5 + [[-0.03, 1.5]] * 25


def policy(**fields):
    return vla.PolicySpec(url="http://policy.test/act", **fields)


class PolicyServer:
    """A trained policy behind HTTP: replies in turn, then repeats the last; records each request.

    A reply is a list of actions, an httpx.Response, or an exception to raise."""

    def __init__(self, *replies, gate=None):
        self.replies, self.requests, self.gate = list(replies), [], gate

    def handle(self, request):
        self.requests.append(json.loads(request.content))
        if self.gate:
            self.gate()
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return (
            reply
            if isinstance(reply, httpx.Response)
            else httpx.Response(200, json={"actions": reply})
        )


@pytest.fixture
def serve(monkeypatch):
    """Route every trained policy's HTTP client to a PolicyServer; returns the clients built."""
    built, real = [], vla.HttpPolicy

    def route(server):
        def http(url, cameras):
            built.append(real(url, cameras, httpx.MockTransport(server.handle)))
            return built[-1]

        monkeypatch.setattr(vla, "HttpPolicy", http)
        return built

    return route


def events(tmp_path):
    (log,) = (tmp_path / "runs/runner").glob("*/events.jsonl")
    return [json.loads(line) for line in log.read_text().splitlines()]


async def test_a_trained_driver_alone_sends_anchored_chunks_and_restarts_each_attempt(
    tmp_path, monkeypatch, serve
):
    monkeypatch.setattr(runs, "keychain_key", lambda *i: pytest.fail("no keys for a policy alone"))
    server = PolicyServer(GOOD)
    clients = serve(server)
    producer = Producer()
    policies = {"cones": policy(objective="Follow the cones.")}
    start = {"supervisor": "none", "driver": "cones", "objective": "Follow the cones."}
    async with gateway(producer, None, None, tmp_path, policies) as client:
        options = (await client.get("/runner/options")).json()
        assert options["drivers"][-1] == "cones"
        assert options["trained"] == {
            "cones": {"steerable": False, "objective": "Follow the cones."}
        }
        started = await client.post("/runner/start", json=start | {"brief": "ignored"})
        assert started.status_code == 200 and not started.json()["supervisor"]["model"]
        assert started.json()["driver"]["brief"] == "Follow the cones."
        assert producer.sent("runner", "session/start")[0]["model"] == "cones"
        probe = server.requests[0]  # Start checked the endpoint before the session
        assert probe["state"] == [0.0, 0.0] and probe["instruction"] == "Follow the cones."
        await until(lambda: producer.sent("driver", "chunk"))
        first = {k: v for k, v in producer.sent("driver", "chunk")[0].items() if k != "request_id"}
        Chunk.model_validate(first)
        assert first["curvature_inv_m"][0] == 0.02 and first["reason"].startswith("cones r1:")
        assert first["anchor_s"] == pytest.approx(1000.0 - vla.ACTUATOR_LEAD_S)
        assert server.requests[1]["state"] == [0.0, 0.8]  # measured curvature (absent), speed
        assert producer.sent("driver", "motion") == []
        talk = await client.post("/runner/message", json={"text": "go"})
        assert (talk.status_code, talk.json()["error"]) == (409, "no_supervisor")
        # Another client's chunk pauses the driver for its length; a stop until resumed.
        other = {"curvature_inv_m": [0.0] * 30, "speed_mps": [1.0] * 30}
        await client.post("/api/chunk", json=other, headers={"X-Client": "UI"})
        driver = (await client.get("/runner/status")).json()["driver"]
        assert (driver["state"], driver["reason"]) == ("paused", "operator_command")
        await client.post("/api/stop", json={}, headers={"X-Client": "UI"})
        sent = len(producer.sent("driver", "chunk"))
        await asyncio.sleep(0.1)
        assert len(producer.sent("driver", "chunk")) == sent
        assert (await client.post("/runner/end-attempt")).status_code == 200
        assert (await client.post("/runner/next-attempt")).status_code == 200
        assert clients[0].client.is_closed and len(clients) == 2  # a fresh client per attempt
        await until(lambda: len(producer.sent("driver", "chunk")) > sent)
        assert producer.sent("driver", "chunk")[sent]["reason"].startswith("cones r2:")
        await client.post("/runner/stop", json={"outcome": "completed"})
    assert clients[1].client.is_closed


async def test_a_failing_policy_sends_the_car_nothing_and_the_run_shows_why(tmp_path, serve):
    server = PolicyServer(
        GOOD,  # Start's check
        GOOD,
        httpx.Response(500, text="out of memory"),
        httpx.Response(200, content=b'{"actions": [[NaN, 1.0]]}'),
        httpx.Response(200, json={"actions": [[0.01, 1.0, 3.0]]}),
        httpx.Response(200, json={"actions": []}),
        httpx.Response(200, json={"plan": GOOD}),
        httpx.ReadTimeout("policy too slow"),
        GOOD,
        httpx.Response(503, text="gone"),
    )
    serve(server)
    producer = Producer()
    start = {"supervisor": "none", "driver": "flaky", "objective": "cones"}
    async with gateway(producer, None, None, tmp_path, {"flaky": policy()}) as client:
        assert (await client.post("/runner/start", json=start)).status_code == 200
        await until(lambda: len(server.requests) >= 12)
        decisions = (await client.get("/runner/status")).json()["driver"]["decisions"]
        assert {d["result"] for d in decisions} >= {"error"}
        await client.post("/runner/stop", json={"outcome": "aborted"})
    chunks = producer.sent("driver", "chunk")
    assert len(chunks) == 2  # only the two good replies reached the car
    for chunk in chunks:
        Chunk.model_validate({k: v for k, v in chunk.items() if k != "request_id"})
    assert producer.sent("driver", "motion") == []
    errors = [e["error"] for e in events(tmp_path) if e["kind"] == "driver_error"]
    for expected in ("500", "finite", "finite", "finite", "actions", "too slow", "503"):
        assert any(expected in error for error in errors), expected


async def test_a_broken_policy_fails_start_before_any_session(tmp_path, serve):
    producer = Producer()
    policies = {"down": policy()}
    async with gateway(producer, None, None, tmp_path, policies) as client:
        unknown = {"supervisor": "none", "driver": "nobody", "objective": "park"}
        assert (await client.post("/runner/start", json=unknown)).status_code == 422
        for reply in (httpx.ConnectError("refused"), httpx.Response(200, json={"actions": "go"})):
            clients = serve(PolicyServer(reply))
            broken = await client.post("/runner/start", json=unknown | {"driver": "down"})
            assert broken.status_code == 409 and broken.json()["error"].startswith("down: ")
            assert clients[-1].client.is_closed
        assert not (await client.get("/runner/status")).json()["running"]
    assert producer.commands == []


def test_a_policy_cannot_take_the_name_of_another_driver():
    for name in ("none", "typesafe/jev-1.14"):
        with pytest.raises(ValueError, match="must differ"):
            create_app(client=httpx.AsyncClient(), policies={name: policy()})


async def test_views_come_from_one_frame_per_camera_and_a_decision_model_reads_only_narrow(
    tmp_path, serve
):
    server = PolicyServer(GOOD)
    serve(server)
    policies = {
        "two": policy(cameras=("front", "wide")),
        "road": policy(cameras=("road_left", "road_right")),
    }
    sizes = {}
    for name in policies:
        producer = Producer()
        start = {"supervisor": "none", "driver": name, "objective": "cones"}
        async with gateway(producer, None, None, tmp_path, policies) as client:
            await client.post("/runner/start", json=start)
            await until(lambda: producer.sent("driver", "chunk"))
            await client.post("/runner/stop", json={"outcome": "aborted"})
        images = server.requests[-1]["images"]
        sizes |= {v: Image.open(io.BytesIO(base64.b64decode(images[v]))).size for v in images}
        expected = {"two": {"narrow", "wide"}, "road": {"narrow"}}[name]
        assert set(producer.cameras) == expected
    assert sizes == {
        "front": (vla.WIDTH, vla.HEIGHT),
        "wide": (vla.WIDTH, vla.HEIGHT),
        "road_left": (vla.WIDTH // 2, vla.HEIGHT - vla.ROAD_TOP),
        "road_right": (vla.WIDTH // 2, vla.HEIGHT - vla.ROAD_TOP),
    }
    producer = Producer()
    async with gateway(producer, Decider(("straight", "slow")), None) as client:
        await client.post("/runner/start", json=START | {"supervisor": "none"})
        await until(lambda: producer.sent("driver", "motion"))
        await client.post("/runner/stop", json={"outcome": "aborted"})
    assert set(producer.cameras) == {"narrow"}


async def test_a_steerable_policy_starts_on_a_command_and_its_supervisor_may_choose_only_those(
    tmp_path, serve
):
    server = PolicyServer(GOOD)
    serve(server)
    chat = Chat(
        [tool_call("set_brief", {"brief": "Turn left at the orange cone."}, "free")],
        [tool_call("set_brief", {"brief": "sharp left", "reason": "tight bend"}, "listed")],
    )
    producer = Producer()
    policies = {"steer": policy(steerable=True), "fixed": policy()}
    async with gateway(producer, None, chat, tmp_path, policies) as client:
        refused = await client.post("/runner/start", json=START | {"driver": "fixed"})
        assert refused.status_code == 422 and "steerable" in refused.json()["error"]
        started = await client.post("/runner/start", json=START | {"driver": "steer"})
        assert started.status_code == 200
        assert started.json()["driver"]["brief"] == "go straight"  # the objective is no command
        assert server.requests[0]["instruction"] == "go straight"
        await until(lambda: len(chat.bodies) >= 3)
        brief = next(t for t in chat.bodies[0]["tools"] if t["function"]["name"] == "set_brief")
        assert brief["function"]["parameters"]["properties"]["brief"]["enum"] == list(vla.COMMANDS)
        results = [m["content"] for m in chat.bodies[-1]["messages"] if m["role"] == "tool"]
        assert "not_a_command" in results[0] and "not_a_command" not in results[1]
        driver = (await client.get("/runner/status")).json()["driver"]
        assert (driver["brief"], driver["revision"]) == ("sharp left", 2)
        await until(lambda: server.requests[-1]["instruction"] == "sharp left")
        await until(
            lambda: any(
                c["reason"].startswith("steer r2:") for c in producer.sent("driver", "chunk")
            )
        )
        await client.post("/runner/stop", json={"outcome": "aborted"})


async def test_start_brings_the_comma_online_only_when_it_is_offline(tmp_path, serve):
    serve(PolicyServer(GOOD))
    producer = Producer()
    producer.down = True
    connected = []

    def online():
        connected.append(True)
        producer.down = False
        return {"status": "online"}

    async with httpx.AsyncClient(
        base_url="http://device", transport=httpx.MockTransport(producer.handle)
    ) as upstream:
        runner = runs.Runner(upstream, tmp_path, policies={"trained": policy()}, online=online)
        body = {"supervisor": "none", "driver": "trained", "objective": "cones"}
        await runner.start(body)
        assert connected == [True]
        await runner.stop("aborted")
        await runner.start(body)
        assert connected == [True]
        await runner.stop("aborted")


async def test_stop_and_duplicate_start_during_the_policy_check_cannot_start_motion(
    tmp_path, serve
):
    entered, release = threading.Event(), threading.Event()

    def hold():
        entered.set()
        assert release.wait(3)

    serve(PolicyServer(GOOD, gate=hold))
    producer = Producer()
    async with gateway(producer, None, None, tmp_path, {"trained": policy()}) as client:
        body = {"supervisor": "none", "driver": "trained", "objective": "cones"}
        task = asyncio.create_task(client.post("/runner/start", json=body))
        try:
            await until(entered.is_set)
            status = (await client.get("/runner/status")).json()
            assert status["preparing"] and not status["running"]
            duplicate = await client.post("/runner/start", json=body)
            assert duplicate.json()["error"] == "preparation_in_progress"
            assert (await client.post("/api/stop", json={})).is_success
        finally:
            release.set()
        response = await task
        assert response.json()["error"] == "start_cancelled_by_stop"
    assert not producer.sent("driver", "chunk")
    assert not producer.sent("runner", "session/start")
