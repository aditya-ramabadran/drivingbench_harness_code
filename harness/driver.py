"""The driver's fast loop: a policy turns the latest frame into the next command, tick by tick.

It runs inside the laptop gateway, so it sees every other client's motion or stop and yields to it.
It is a client of the same producer API the chat tools use, labelled `driver`, and never touches
the comma. Each command is short (a 5 s motion or a chunk of a few seconds), so the car brakes
soon after the loop stalls. Policies: a decision model (below) or a trained policy (vla.py).
"""

import asyncio
import base64
import io
import math
import time
from collections import deque
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from drivingbench.shared.contracts import (
    CHUNK_STEP_S,
    POLICY_CAMERAS,
    PROTOCOL,
    STEERING_SCALE_DEG,
)
from PIL import Image

CLIENT = "driver"
TICK_S = 0.5
COMMAND_DURATION_S = 5.0  # the producer's minimum; doubles as the dead-man if the driver stalls
DECISION_TIMEOUT_S = 15.0  # model response budget; the comma's command expiry is independent
FRAME_WIDTH = 672  # a small frame keeps decision requests fast and within image size limits
BRIEF_MAX_CHARS = 1000
STEERING = {
    "straight": 0,
    "left_30": 30,
    "left_60": 60,
    "left_100": 100,
    "right_30": -30,
    "right_60": -60,
    "right_100": -100,
    "keep": None,
}
SPEEDS = {"stop": 0.0, "crawl": 0.6, "slow": 1.0, "normal": 1.5}
# The controller's block reasons while the operator has the car: send nothing, resume when clear.
NOT_READY = {
    "gear_not_drive",
    "operator_brake",
    "operator_intervention",
    "native_disengaged",
    "producer_unavailable",
    "emergency_speed_exceeded",
    "native_offline",
}
QUESTIONS = {
    "steering": {
        "type": "choice",
        "instructions": (
            "You are the low-level driver of a real car on a cone course in a parking lot, choosing "
            "every half second from the front camera image. A supervisor writes your brief: follow it. "
            "Steering is a percent of full lock; full lock turns in a circle of very roughly 5.5 m radius, "
            "nearly a U-turn, and the wheel reaches it within about 1.5 s, also at standstill. Choose "
            "the steering for the next second."
        ),
        "criteria": {
            "straight": "Wheel straight",
            "left_30": "Gentle left",
            "left_60": "Medium left",
            "left_100": "Full left",
            "right_30": "Gentle right",
            "right_60": "Medium right",
            "right_100": "Full right",
            "keep": "Keep the current steering",
        },
    },
    "speed": {
        "type": "choice",
        "instructions": (
            "Choose the speed for the next second. Choose stop if anything is in your path, the brief "
            "does not cover what you see, or you are unsure."
        ),
        "criteria": {
            "stop": "Brake to a stop",
            "crawl": "0.6 m/s",
            "slow": "1.0 m/s",
            "normal": "1.5 m/s",
        },
    },
}
DEFAULT_BRIEF = (
    "Objective: {objective}\n"
    "Keep the whole car between the cone lines and away from curbs, islands and parked cars.\n"
    "Drive at crawl speed until told otherwise.\n"
    "Stop if you cannot see which way the course continues."
)


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def frame_part(jpeg: bytes) -> dict:
    image = Image.open(io.BytesIO(jpeg))
    image.thumbnail((FRAME_WIDTH, FRAME_WIDTH))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=85)
    data = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def most_likely(answer: dict) -> tuple[str, float]:
    """The highest-probability option of a choice answer; a bare value counts as certain."""
    probabilities = answer.get("probabilities") or {}
    if probabilities:
        option = max(probabilities, key=probabilities.get)
        return option, float(probabilities[option])
    option = answer.get("choice") or answer.get("value")
    if option is None:
        raise ValueError("answer without probabilities or choice")
    return str(option), 1.0


async def command(upstream: httpx.AsyncClient, client: str, path: str, body: dict) -> dict:
    """POST motion, a chunk or stop to the producer the way the gateway proxy does."""
    request_id = uuid4().hex
    response = await upstream.post(
        "/" + path,
        json={**body, "request_id": request_id},
        headers={
            "X-Client": client,
            "X-Drivingbench-Protocol": str(PROTOCOL),
            "X-Request-ID": request_id,
        },
    )
    result = response.json()
    if response.is_error and "error" not in result:
        result = {"error": result.get("detail") or f"http_{response.status_code}"}
    return result


class DecisionPolicy:
    """A decision model picks one steering and one speed option per frame (OpenRouter decide)."""

    tick_s = TICK_S
    cameras = ("front",)

    def __init__(self, decide, model: str):
        self.decide, self.name = decide, model
        self.steering = 0  # last accepted signed percent, for "keep"

    async def act(self, status: dict, frames: dict, brief: str, revision: int):
        scale = (status.get("settings") or {}).get("max_steering_angle_deg") or STEERING_SCALE_DEG
        degrees = status.get("steering_deg")
        measured = round(100 * degrees / scale) if degrees is not None else None
        side = "left" if (measured or 0) >= 0 else "right"
        state = [
            f"BRIEF (revision {revision}, written by your supervisor):\n{brief}",
            f"CAR: speed {status.get('speed_mps') or 0:.1f} m/s; measured steering "
            f"{'unknown' if measured is None else f'{abs(measured)}% {side}'}; "
            f"last commanded steering {abs(self.steering)}% "
            f"{'left' if self.steering > 0 else 'right' if self.steering < 0 else 'straight'}.",
            frame_part(frames["front"].content),
        ]
        answers = await self.decide(self.name, state, QUESTIONS)
        steering, p_steering = most_likely(answers["steering"])
        speed, p_speed = most_likely(answers["speed"])
        if steering not in STEERING or speed not in SPEEDS:
            raise ValueError(f"unexpected answer {steering!r}, {speed!r}")
        reason = f"driver r{revision}: {steering} {p_steering:.2f}, {speed} {p_speed:.2f}"
        text = f"{steering} {p_steering:.0%} · {speed} {p_speed:.0%}"
        if speed == "stop":
            return "stop", {"reason": reason}, text, {"answers": answers}
        percent = self.steering if STEERING[steering] is None else STEERING[steering]
        body = {
            "direction": "left" if percent > 0 else "right" if percent < 0 else "straight",
            "steering_percent": abs(percent),
            "speed_mps": SPEEDS[speed],
            "duration_s": COMMAND_DURATION_S,
            "reason": reason,
        }
        return "motion", body, text, {"answers": answers}

    def accepted(self, path: str, body: dict):
        if path == "motion":
            sign = {"left": 1, "right": -1, "straight": 0}[body["direction"]]
            self.steering = sign * body["steering_percent"]


class Driver:
    def __init__(self, upstream: httpx.AsyncClient, record, clock=None):
        self.upstream, self.record = upstream, record
        self.clock = clock or time.monotonic
        self.policy = None
        self.task: asyncio.Task | None = None
        self.brief = ""
        self.revision = 0
        self.paused_until, self.pause_reason = 0.0, None
        self.handovers = 0  # pauses, resumes and briefs; a decision spanning one is dropped
        self.waiting: str | None = None
        self.decisions: deque = deque(maxlen=10)

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def status(self) -> dict:
        state, reason = "driving", None
        if not self.running:
            state = "stopped"
        elif self.clock() < self.paused_until:
            state, reason = "paused", self.pause_reason
        elif self.waiting:
            state, reason = "waiting", self.waiting
        return {
            "state": state,
            "reason": reason,
            "model": self.policy.name if self.policy else "",
            "brief": self.brief,
            "revision": self.revision,
            "decisions": list(self.decisions),
        }

    def start(self, policy, brief: str):
        self.policy = policy
        self.brief, self.revision = brief[:BRIEF_MAX_CHARS], self.revision + 1
        self.paused_until, self.pause_reason, self.waiting = 0.0, None, None
        self.handovers += 1
        self.decisions.clear()
        self.record("driver_start", model=policy.name, brief=self.brief, revision=self.revision)
        self.task = asyncio.create_task(self.control())

    async def stop(self, reason: str) -> dict:
        if self.running:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        try:
            result = await command(self.upstream, CLIENT, "stop", {"reason": reason[:200]})
        except (httpx.HTTPError, ValueError) as exc:  # the operator can still stop from the UI
            result = {"error": type(exc).__name__}
        self.record("driver_stop", reason=reason, result=result)
        return result

    def set_brief(self, brief: str, reason: str = "", source: str = "supervisor") -> dict:
        self.brief, self.revision = brief[:BRIEF_MAX_CHARS], self.revision + 1
        self.paused_until, self.pause_reason = 0.0, None  # a new brief hands the car back
        self.handovers += 1
        self.record("brief", revision=self.revision, brief=self.brief, reason=reason, source=source)
        return {"revision": self.revision, "timestamp": now_utc()}

    def resume(self):
        self.paused_until, self.pause_reason = 0.0, None
        self.handovers += 1
        self.record("driver_resume")

    def yield_to(self, client: str, path: str, body: dict):
        """Another client's accepted command pauses the driver until it ends; a stop until resumed."""
        if not self.running or client == CLIENT:
            return
        if path in ("motion", "chunk"):
            seconds = body.get("duration_s") or len(body.get("speed_mps") or ()) * CHUNK_STEP_S
            self.paused_until = self.clock() + float(seconds)
            self.pause_reason = (
                "supervisor_command" if client == "supervisor" else "operator_command"
            )
        else:
            self.paused_until, self.pause_reason = math.inf, "stopped"
        self.handovers += 1
        self.record("driver_pause", reason=self.pause_reason, client=client)

    async def control(self):
        while True:
            started = self.clock()
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # a failed tick sends nothing; the last command expires
                text = f"{type(error).__name__}: {error}"[:300]
                self.record("driver_error", error=text)
                latency_ms = round(1000 * (self.clock() - started))
                self.decisions.append(
                    {
                        "timestamp": now_utc(),
                        "text": text,
                        "latency_ms": latency_ms,
                        "result": "error",
                    }
                )
            await asyncio.sleep(max(0.0, self.policy.tick_s - (self.clock() - started)))

    async def frames(self):
        """Fetch each physical camera once; the road views share the narrow frame."""
        sources = tuple(dict.fromkeys(POLICY_CAMERAS[name] for name in self.policy.cameras))
        fetched = {source: await self.upstream.get(f"/camera/{source}") for source in sources}
        return {name: fetched[POLICY_CAMERAS[name]] for name in self.policy.cameras}

    def ready(self, status):
        blocked = status.get("reason") if status.get("reason") in NOT_READY else None
        if status.get("protocol") != PROTOCOL:
            blocked = "protocol_mismatch"
        if blocked != self.waiting:
            self.waiting = blocked
            self.record("driver_waiting" if blocked else "driver_ready", reason=blocked)
        return blocked is None

    async def tick(self):
        if self.clock() < self.paused_until:
            return
        status = (await self.upstream.get("/status")).json()
        if not self.ready(status):
            return
        frames = await self.frames()
        if any(frame.status_code != 200 for frame in frames.values()):
            return
        sources = {POLICY_CAMERAS[name]: frame for name, frame in frames.items()}
        self.record(
            "driver_frame",
            frame_ids={name: frame.headers.get("X-Frame-ID") for name, frame in sources.items()},
            source_captured_at={
                name: frame.headers.get("X-Captured-At") for name, frame in sources.items()
            },
            image_age_s={
                name: frame.headers.get("X-Image-Age-S") for name, frame in sources.items()
            },
        )
        revision, handovers = self.revision, self.handovers
        asked = self.clock()
        path, body, text, evidence = await asyncio.wait_for(
            self.policy.act(status, frames, self.brief, revision), DECISION_TIMEOUT_S
        )
        latency_ms = round(1000 * (self.clock() - asked))
        if handovers != self.handovers:
            result = {"status": "discarded"}
        else:
            result = await command(self.upstream, CLIENT, path, body)
            if "error" not in result:
                self.policy.accepted(path, body)
        self.decisions.append(
            {
                "timestamp": now_utc(),
                "text": text,
                "latency_ms": latency_ms,
                "result": result.get("status") or result.get("error"),
                **evidence,
            }
        )
        self.record(
            "decision",
            revision=revision,
            latency_ms=latency_ms,
            path=path,
            body=body,
            result=result,
            **evidence,
        )
