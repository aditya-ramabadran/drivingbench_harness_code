"""Trained policies in the driver loop: the latest frames in, a timed (curvature, speed) chunk out.

A policy sees its views of the producer's frames (resized and cut below), the measured curvature
and speed, and its instruction. Its chunk runs through the same controller and limits as
set_motion, anchored to the older frame's capture less the car's steering response delay: the
comma runs the row due at each 0.1 s step on its own clock, so transit cannot shift rows. The
laptop's estimate of what has already passed only decides whether a plan arrived too late.

Policies are served over HTTP and named in the operator's registry (docs/runner.md).
"""

import asyncio
import base64
import io
import math
import time
from typing import Annotated, Literal

import httpx
from drivingbench.shared.contracts import (
    CHUNK_MAX_STEPS,
    CHUNK_STEP_S,
    POLICY_CAMERAS,
    SPEED_RANGE_MPS,
)
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, TypeAdapter

ACTUATOR_LEAD_S = 0.2  # the steering's response delay, measured on a Corolla: rows run this early
# A predicted speed this low is a stop; above it, at least the car's 0.5 m/s crawl. A model
# starting from standstill predicts a gentle 0.2-0.4 m/s first: a higher threshold never starts.
STOP_BELOW_MPS = 0.15
# Frames arrive ~0.5 s old on the car's network, so much of a short chunk has passed when it is
# sent: one shorter than HOLD_STEPS holds its last step that long, and one that ended less than
# LATE_STEPS ago holds its last step; later than that is a stop.
HOLD_STEPS, LATE_STEPS = 8, 10
REQUEST_TIMEOUT_S = 10.0  # a plan this late has expired anyway; the car brakes at its chunk's end

WIDTH, HEIGHT = 672, 380  # half the comma four's 1344×760 road cameras, their true geometry
ROAD_TOP = 140  # rows above this are sky and treetops in every lot
ROAD_VIEWS = ("road_left", "road_right")
JPEG_QUALITY = 90
# A steerable policy's instructions, each describing the car's next 1.5 s (docs/runner.md).
COMMANDS = (
    "go straight",
    "curve left",
    "curve right",
    "sharp left",
    "sharp right",
    "slow down",
    "stop",
)


class PolicySpec(BaseModel):
    """One entry of the operator's policy registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: StrictStr = Field(pattern=r"^https?://\S+$")
    cameras: tuple[Literal[tuple(POLICY_CAMERAS)], ...] = Field(default=("front",), min_length=1)
    objective: StrictStr = ""  # the instruction it was trained on, prefilled in the Runner
    steerable: StrictBool = False  # trained on COMMANDS; a chat model may choose among them


REGISTRY = TypeAdapter(dict[Annotated[str, Field(min_length=1, max_length=64)], PolicySpec])


def view(frame: Image.Image, name: str) -> Image.Image:
    """A policy view of a policy-size frame: the frame itself, or a road view, the band below the
    sky split down the middle. Each road view is nearly square, so a 224×224 model input shows
    the cones twice as large as the whole frame does."""
    if name not in ROAD_VIEWS:
        return frame
    left = 0 if name == "road_left" else WIDTH // 2
    return frame.crop((left, ROAD_TOP, left + WIDTH // 2, HEIGHT))


def jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def policy_input(status: dict, frames: dict) -> tuple[dict[str, bytes], list[float]]:
    """Each view as a JPEG, decoding each camera's frame once, and the measured state."""
    decoded = {}
    for name, response in frames.items():
        source = POLICY_CAMERAS[name]
        if source not in decoded:
            with Image.open(io.BytesIO(response.content)) as image:
                rgb = image.convert("RGB")
            decoded[source] = rgb.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    images = {name: jpeg(view(decoded[POLICY_CAMERAS[name]], name)) for name in frames}
    state = [float(status.get("curvature_inv_m") or 0), float(status.get("speed_mps") or 0)]
    return images, state


def rows(actions) -> list[tuple[float, float]]:
    """A policy's prediction, checked: a non-empty list of finite [curvature, speed] rows."""
    if not (
        isinstance(actions, list)
        and actions
        and all(
            isinstance(row, list | tuple)
            and len(row) == 2
            and all(
                isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)
                for x in row
            )
            for row in actions
        )
    ):
        raise ValueError("a policy returns a non-empty list of finite [curvature, speed] rows")
    return [(float(k), float(v)) for k, v in actions]


def shape(actions, skip: int, ceiling: float) -> dict:
    """The sendable part of a predicted chunk: elapsed steps dropped, speeds made executable."""
    predicted = rows(actions)
    steps = predicted[skip : skip + CHUNK_MAX_STEPS]
    if not steps and skip < len(predicted) + LATE_STEPS:
        steps = predicted[-1:]
    steps += steps[-1:] * (HOLD_STEPS - len(steps))
    return {
        "curvature_inv_m": [min(max(k, -1.0), 1.0) for k, _ in steps],
        "speed_mps": [
            0.0 if v < STOP_BELOW_MPS else min(max(v, SPEED_RANGE_MPS[0]), ceiling)
            for _, v in steps
        ],
    }


class HttpPolicy:
    """A policy server: POST {"images", "state", "instruction"}, reply {"actions"}."""

    def __init__(self, url: str, cameras=("front",), transport=None):
        self.url, self.cameras = url, tuple(cameras)
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT_S, trust_env=False, transport=transport)

    def __call__(self, images: dict[str, bytes], state: list[float], instruction: str):
        response = self.client.post(
            self.url,
            json={
                "images": {name: base64.b64encode(data).decode() for name, data in images.items()},
                "state": state,
                "instruction": instruction,
            },
        )
        response.raise_for_status()
        reply = response.json()
        if not isinstance(reply, dict) or "actions" not in reply:
            raise ValueError('a policy replies {"actions": [[curvature, speed], ...]}')
        return reply["actions"]

    def close(self):
        self.client.close()


class ChunkPolicy:
    """A trained policy for the driver loop, around `predict(images, state, instruction)`."""

    tick_s = 0.3  # a new chunk about three times a second; each runs to its end if none follows

    def __init__(self, name: str, predict, clock=None, commands=None):
        self.name, self.predict, self.commands = name, predict, commands
        self.cameras = tuple(getattr(predict, "cameras", ("front",)))
        self.clock = clock or time.monotonic

    def close(self):
        if close := getattr(self.predict, "close", None):
            close()

    def probe(self, instruction: str):
        """One prediction from black views: an unreachable or malformed policy fails at Start."""
        black = Image.new("RGB", (WIDTH, HEIGHT))
        images = {name: jpeg(view(black, name)) for name in self.cameras}
        rows(self.predict(images, [0.0, 0.0], instruction))

    async def act(self, status: dict, frames: dict[str, httpx.Response], brief: str, revision: int):
        """The anchored chunk for the latest frames, or a stop when it came too late."""
        asked = self.clock()  # before preprocessing: everything from here on delays the chunk
        age = max(float(f.headers.get("X-Image-Age-S") or 0) for f in frames.values())
        # The older frame dates the plan, on the comma's camera clock.
        captured_at = min(float(f.headers["X-Captured-At"]) for f in frames.values())
        actions = await asyncio.to_thread(
            lambda: self.predict(*policy_input(status, frames), brief)
        )
        ceiling = (status.get("settings") or {}).get("speed_limit_mps") or SPEED_RANGE_MPS[1]
        skip = round((age + self.clock() - asked + ACTUATOR_LEAD_S) / CHUNK_STEP_S)
        due = shape(actions, skip, ceiling)  # the laptop's estimate, for the late decision
        predicted = len(actions)
        evidence = {"predicted_steps": predicted, "skipped_steps": skip}
        if not due["speed_mps"]:  # inference outlasted its own chunk: brake, like an expiry
            reason = f"{self.name} r{revision}: chunk expired during inference"
            return "stop", {"reason": reason}, f"late by {skip - predicted} steps", evidence
        holding = skip >= predicted
        if holding:
            evidence["holding_last"] = True
        body = shape(actions, 0, ceiling)
        body["anchor_s"] = captured_at - ACTUATOR_LEAD_S  # the producer converts it to its clock
        body["reason"] = f"{self.name} r{revision}: {len(body['speed_mps'])} steps, {skip} elapsed"
        text = (
            f"curvature {due['curvature_inv_m'][0]:+.3f}/m · {due['speed_mps'][0]:.2f} m/s"
            f" · {len(body['speed_mps'])} steps (skipped {skip})"
        )
        if holding:
            text += " · holding final target"
        return "chunk", body, text, evidence

    def accepted(self, path: str, body: dict):
        """Chunks carry nothing from one tick to the next."""


def build(name: str, spec: PolicySpec) -> ChunkPolicy:
    """A registry entry's policy; a steerable one takes only COMMANDS as instructions."""
    commands = COMMANDS if spec.steerable else None
    return ChunkPolicy(name, HttpPolicy(spec.url, spec.cameras), commands=commands)
