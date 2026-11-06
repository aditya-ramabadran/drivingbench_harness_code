"""The /runner chat model: it writes the driver's brief, or drives alone (solo) with no driver.

Its observe, set_motion and stop_now are the drivingbench_sandbox tools (same schemas and wording);
set_brief is its only addition, and solo runs get exactly the public tools. Like a chat app, it
works until it ends its turn and then waits for the operator (a message, the reflection prompt or
the continuation prompt). Models are on OpenRouter or Google's Gemini API.
"""

import asyncio
import json
import time
from collections import deque

import httpx
from drivingbench.harness.driver import BRIEF_MAX_CHARS, command, now_utc
from drivingbench.mcp.tools import OBSERVE_KEYS, make_server
from drivingbench.shared.contracts import REASON_MAX_CHARS
from pydantic import ValidationError

CLIENT = "supervisor"
REASONING_EFFORT = "medium"
RETRY_S = 2.0  # a failed completion is asked once more after this, then the turn stalls
SET_BRIEF = {
    "type": "function",
    "function": {
        "name": "set_brief",
        "description": (
            "Replace the driver's instructions. A fast driver model steers the car about twice a "
            "second from the front camera, choosing steering (straight, left or right at 30/60/100%, "
            "or keep) and speed (stop, 0.6, 1.0, 1.5 m/s). It sees only the latest frame, the car's "
            "speed and steering, and your brief; it has no memory between decisions and no map. Write "
            "for that driver: name visible landmarks, which side the cones should stay on, when to "
            "start turning and how hard, and the speed. It takes effect at the next decision and hands "
            "the car back to the driver if your own command was overriding it.\n\n"
            "reason: optional short note on what this brief is for; recorded, never shown to the driver."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "brief": {"type": "string", "minLength": 1, "maxLength": BRIEF_MAX_CHARS},
                "reason": {"type": "string", "maxLength": REASON_MAX_CHARS},
            },
            "required": ["brief"],
            "additionalProperties": False,
        },
    },
}


SANDBOX = make_server(None)  # definitions and argument checks only; never calls a gateway


def assistant_message(message: dict) -> dict:
    """Keep provider continuation data, including Gemini's message and tool-call signatures."""
    return {
        k: message[k]
        for k in ("role", "content", "tool_calls", "reasoning_details", "extra_content")
        if k in message
    }


def command_brief(commands) -> dict:
    """set_brief for a steerable trained driver: the brief is one of the commands it learned."""
    function = {
        "name": "set_brief",
        "description": (
            "Choose the driver's next command. A trained driving model steers the car several times a "
            "second from its camera views and follows exactly one of these commands, kept until you "
            f"choose another: {', '.join(commands)}. Each describes what the car does over about the "
            "next 1.5 seconds, as in the driving it learned from: go straight (little or no "
            "steering); curve left or right (a gentle turn, radius 12 to 50 m); sharp left or right (a "
            "tight turn, radius under 12 m, down to full lock at about 5.5 m); slow down (lose about a "
            "quarter of the speed while going straight); stop (come to a standstill within about 1.5 "
            "seconds). The driver sets the exact steering and speed itself from what it sees; you "
            "choose the maneuver. Your turns take seconds, so choose each command just before the car "
            "needs it, when the cones that start the turn are a car length or two ahead, and go back "
            "to go straight as it comes out. It takes effect at the driver's next chunk and hands the "
            "car back to the driver if your own command was overriding it.\n\n"
            "reason: optional short note on why; recorded, never shown to the driver."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "brief": {"type": "string", "enum": list(commands)},
                "reason": {"type": "string", "maxLength": REASON_MAX_CHARS},
            },
            "required": ["brief"],
            "additionalProperties": False,
        },
    }
    return {"type": "function", "function": function}


async def sandbox_tools(solo: bool = False, commands=None) -> list[dict]:
    """The drivingbench_sandbox tool definitions as chat-completion function tools."""
    tools = await SANDBOX.list_tools()
    functions = []
    for tool in tools:
        description = tool.description
        if tool.name == "observe" and not solo:
            description += " Also reports the driver's brief, state and latest decisions."
        functions.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": description,
                    "parameters": tool.inputSchema,
                },
            }
        )
    return functions if solo else functions + [command_brief(commands) if commands else SET_BRIEF]


class Supervisor:
    def __init__(self, upstream: httpx.AsyncClient, driver, record, may_move=lambda: True):
        self.upstream, self.driver, self.record = upstream, driver, record
        self.may_move = may_move  # false between attempts: only the operator moves the car then
        self.chat = None
        self.model = self.tier = ""
        self.solo = False
        self.commands = None
        self.tools: list[dict] = []
        self.messages: list[dict] = []
        self.task: asyncio.Task | None = None
        self.transcript: deque = deque(maxlen=60)
        self.cost = 0.0
        self.cost_known = True
        self.stalled = ""  # the error that ended the last turn, until the operator speaks

    @property
    def thinking(self) -> bool:
        return self.task is not None and not self.task.done()

    def status(self) -> dict:
        state = "thinking" if self.thinking else "waiting" if self.messages else "stopped"
        return {
            "model": self.model,
            "tier": self.tier,
            "state": "stalled" if self.stalled and not self.thinking else state,
            "reason": self.stalled or None,
            "cost_usd": round(self.cost, 4) if self.cost_known else None,
            "transcript": list(self.transcript),
        }

    def note(self, kind: str, text: str):
        self.transcript.append({"timestamp": now_utc(), "kind": kind, "text": text[:2000]})

    async def start(self, chat, model: str, tier: str, prompt: str, solo: bool = False):
        self.chat, self.model, self.tier, self.solo = chat, model, tier, solo
        self.commands = None if solo else getattr(self.driver.policy, "commands", None)
        self.tools = await sandbox_tools(solo, self.commands)
        self.messages, self.cost = [], 0.0
        self.cost_known = True
        self.transcript.clear()
        await self.say(prompt, "prompt")

    async def say(self, text: str, kind: str = "operator"):
        """Interrupt any turn in progress, add the operator's message, and start a new turn."""
        await self.interrupt()
        self.stalled = ""
        self.messages.append({"role": "user", "content": text})
        self.note(kind, text)
        self.record("supervisor_message", source=kind, text=text)
        self.task = asyncio.create_task(self.turn())

    async def interrupt(self):
        if self.thinking:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.record("supervisor_interrupted")

    async def stop(self):
        await self.interrupt()
        self.messages = []
        close = getattr(getattr(self.chat, "__self__", None), "close", None)
        if close:
            await close()

    async def turn(self):
        try:
            while True:
                body = {
                    "model": self.model,
                    "messages": self.messages,
                    "tools": self.tools,
                    "reasoning": {"effort": REASONING_EFFORT},
                }
                if self.tier != "normal":
                    body["service_tier"] = self.tier
                started = time.monotonic()
                response = await self.complete(body)
                usage = response.get("usage") or {}
                self.cost_known &= usage.get("cost") is not None
                self.cost += float(usage.get("cost") or 0)
                message = response["choices"][0]["message"]
                self.record(
                    "supervisor_response",
                    latency_ms=round(1000 * (time.monotonic() - started)),
                    usage=usage,
                    service_tier=response.get("service_tier"),
                    provider=response.get("provider"),
                    model=response.get("model"),
                    response_id=response.get("id"),
                    requested_service_tier=response.get("requested_service_tier"),
                    content=message.get("content"),
                    tool_calls=message.get("tool_calls"),
                    extra_content=message.get("extra_content"),
                )
                if message.get("content"):
                    self.note("assistant", message["content"])
                calls = message.get("tool_calls") or []
                kept = assistant_message(message)
                results, images = [], []
                for call in calls:
                    result, parts = await self.execute(call)
                    results.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
                    )
                    images += parts
                # Append the turn only once its tools ran: an interruption never leaves a call
                # unanswered.
                self.messages += [kept, *results]
                if images:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "observe images:"}, *images],
                        }
                    )
                if not calls:
                    self.record("supervisor_turn_ended")
                    return
        except asyncio.CancelledError:
            raise
        except Exception as error:  # the driver keeps its brief; the operator can message again
            self.stalled = f"{type(error).__name__}: {error}"[:300]
            self.note("error", self.stalled)
            self.record("supervisor_error", error=self.stalled)

    async def complete(self, body: dict) -> dict:
        """One chat completion, asked twice: a provider's passing error must not end the turn."""
        try:
            return await self.chat(body)
        except Exception as error:
            self.record("supervisor_retry", error=f"{type(error).__name__}: {error}"[:300])
            await asyncio.sleep(RETRY_S)
            return await self.chat(body)

    async def execute(self, call: dict) -> tuple[dict, list]:
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except ValueError:
            return self.result(name, {}, {"error": "invalid_arguments"}), []
        images = []
        if name == "observe":
            packet = (await self.upstream.get("/observe", headers={"X-Client": CLIENT})).json()
            result = {k: packet[k] for k in OBSERVE_KEYS if k in packet}
            images = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{i['mime_type']};base64,{i['data']}"},
                }
                for i in packet.get("images", [])
            ]
            if images:
                result["cameras"] = [i["camera"] for i in packet["images"]]
            if not self.solo:
                driver = self.driver.status()
                result["driver"] = {
                    "state": driver["state"],
                    "reason": driver["reason"],
                    "revision": driver["revision"],
                    "brief": driver["brief"],
                    "decisions": driver["decisions"][-5:],
                }
        elif name in ("set_motion", "stop_now"):
            try:  # exactly what the public MCP accepts, e.g. its speed floor
                SANDBOX._tool_manager.get_tool(name).fn_metadata.arg_model.model_validate(args)
            except ValidationError:
                return self.result(name, args, {"error": "invalid_arguments"}), []
            path = "motion" if name == "set_motion" else "stop"
            if path == "motion" and not self.may_move():
                return self.result(name, args, {"error": "between_attempts"}), []
            if path == "stop":  # pause first, so no driver command lands after the stop
                self.driver.yield_to(CLIENT, path, args)
            outcome = await command(self.upstream, CLIENT, path, args)
            if path == "motion" and "error" not in outcome:
                self.driver.yield_to(CLIENT, path, args)
            result = {k: outcome[k] for k in ("status", "error") if k in outcome}
        elif name == "set_brief":
            brief = str(args.get("brief") or "").strip()
            if not self.driver.running:
                result = {"error": "driver_not_running"}
            elif not brief or len(brief) > BRIEF_MAX_CHARS:
                result = {"error": "invalid_brief"}
            elif self.commands and brief not in self.commands:
                result = {"error": "not_a_command", "commands": list(self.commands)}
            else:
                result = self.driver.set_brief(
                    brief, str(args.get("reason") or "")[:REASON_MAX_CHARS]
                )
        else:
            result = {"error": "unknown_tool"}
        return self.result(name, args, {"timestamp": now_utc(), **result}), images

    def result(self, name, args, result):
        self.note("tool", f"{name}({json.dumps(args)[:300]}) → {json.dumps(result)[:300]}")
        self.record("supervisor_tool", name=name, arguments=args, result=result)
        return result
