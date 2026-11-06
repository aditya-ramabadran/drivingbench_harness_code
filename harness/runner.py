"""Runs without a chat app, attempt by attempt, behind the Runner tab: a driver (a decision model or
a trained policy), a chat model writing its brief, or both; a chat model with no driver drives
alone with exactly the public tools.

Start labels a session and begins attempt 1; End attempt stops the car (and sends the chat model the
reflection prompt); Next attempt restarts the driver from the starting brief (and sends the
continuation prompt); Stop ends the run and the session. Between attempts only the operator moves
the car. Logged to <runs>/runner/<run>/events.jsonl (the laptop config's runs folder).
"""

import asyncio
import json
from functools import partial
from pathlib import Path
from uuid import uuid4

import httpx
from drivingbench.harness import gemini, vla
from drivingbench.harness.driver import (
    BRIEF_MAX_CHARS,
    DEFAULT_BRIEF,
    DecisionPolicy,
    Driver,
    command,
    now_utc,
)
from drivingbench.harness.openrouter import (
    KEYCHAIN_ITEM,
    OpenRouter,
    keychain_has_key,
    keychain_key,
)
from drivingbench.harness.supervisor import Supervisor
from drivingbench.shared.contracts import PROTOCOL
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

NO_SUPERVISOR = NO_DRIVER = "none"
SUPERVISORS = (
    NO_SUPERVISOR,
    "openai/gpt-6-astra",
    "openai/gpt-6.1-sol",
    "openai/gpt-6.1-terra",
    "openai/gpt-6-luna",
    "anthropic/claude-fable-5.1",
    "anthropic/claude-opus-5.5",
    "anthropic/claude-sonnet-5.5",
    "x-ai/grok-4.7",
    "google/gemini-4-argon",
)
TIERS = ("normal", "fast", "ultrafast")
DRIVERS = (
    "typesafe/jev-1.14",  # pinned: ~typesafe/jev-latest moves with each release
    "openai/gpt-6-luna-decisions",
)
OUTCOMES = ("aborted", "completed", "collision")  # the first is the default
ATTEMPT_OUTCOMES = {"unknown", "completed", "intervention", "aborted", "collision"}
MISSING_KEYS = {"no_openrouter_key", "no_gemini_key"}


class Runner:
    def __init__(
        self,
        upstream: httpx.AsyncClient,
        runs: Path | None,
        decide=None,
        chat=None,
        policies: dict[str, vla.PolicySpec] | None = None,
        evaluations=None,
        online=None,
    ):
        self.upstream, self.runs = upstream, runs
        self.injected = (decide, chat)  # tests replace OpenRouter
        self.policies = policies or {}  # the operator's trained policies (vla.PolicySpec)
        if set(self.policies) & {NO_DRIVER, *DRIVERS}:
            raise ValueError(
                "a trained policy's name must differ from none and the decision models"
            )
        self.driver = Driver(upstream, self.record)
        self.supervisor = Supervisor(
            upstream, self.driver, self.record, lambda: not self.between_attempts
        )
        self.run: str | None = None
        self.log = None
        self.attempt = 0
        self.config: dict = {}
        self.policy = None  # makes the driver's policy afresh for each attempt
        self.in_attempt = False
        self.evaluations = evaluations
        self.evaluation_session = None
        self.metadata_error = None
        self.online = online
        self.preparing = None
        self.preparation_lock = asyncio.Lock()
        self.preparation_epoch = 0
        self.command_gate = asyncio.Lock()  # see the gateway proxy

    def cancel_preparation(self):
        self.preparation_epoch += 1

    async def close_policy(self, policy):
        if close := getattr(policy, "close", None):
            await asyncio.to_thread(close)

    @property
    def running(self) -> bool:
        return self.run is not None

    @property
    def between_attempts(self) -> bool:
        return self.running and not self.in_attempt

    @property
    def supervised(self) -> bool:
        return self.config.get("supervisor", NO_SUPERVISOR) != NO_SUPERVISOR

    def record(self, kind: str, **data):
        if self.log:
            self.log.write(json.dumps({"timestamp": now_utc(), "kind": kind, **data}) + "\n")
            self.log.flush()
        if self.evaluations:
            try:
                if kind == "session_start":
                    self.evaluation_session = data["result"].get("id")
                    mode = "same_context" if self.supervised else "independent"
                    self.evaluations.start(data["result"], mode=mode, source="runner")
                elif self.evaluation_session:
                    if kind == "attempt_start" and data["attempt"] > 1:
                        self.evaluations.marker(self.evaluation_session, "start")
                    elif kind == "attempt_end":
                        self.evaluations.marker(
                            self.evaluation_session,
                            "end",
                            outcome=data.get("outcome", "unknown"),
                            note=data.get("note", ""),
                        )
                    elif kind == "stop":
                        self.evaluations.end(
                            self.evaluation_session, outcome=data["outcome"], note=data["note"]
                        )
            except (OSError, ValueError) as exc:
                self.metadata_error = str(exc)

    def status(self) -> dict:
        return {
            "running": self.running,
            "preparing": self.preparing,
            "run": self.run,
            "attempt": self.attempt,
            "between_attempts": self.between_attempts,
            "metadata_error": self.metadata_error,
            "config": {
                k: v
                for k, v in self.config.items()
                if k not in ("prompt", "reflection", "continuation")
            },
            "driver": self.driver.status(),
            "supervisor": self.supervisor.status(),
        }

    def clients(self, supervisor: str, driver: str):
        """Only the credentials this run uses: OpenRouter, Google, or none for a policy alone."""
        decide, chat = self.injected
        if gemini.is_model(supervisor) and not chat:
            key = keychain_key(gemini.KEYCHAIN_ITEM)
            if not key:
                raise RuntimeError("no_gemini_key")
            chat = gemini.Gemini(key).chat
        openrouter_chat = supervisor != NO_SUPERVISOR and not chat
        if openrouter_chat or (driver in DRIVERS and not decide):
            key = keychain_key()
            if not key:
                raise RuntimeError("no_openrouter_key")
            openrouter = OpenRouter(key)
            decide, chat = decide or openrouter.decide, chat or openrouter.chat
        return decide, chat

    async def start(self, body: dict) -> dict:
        if self.preparation_lock.locked():
            raise RuntimeError("preparation_in_progress")
        async with self.preparation_lock:
            try:
                self.preparing = "Checking the comma connection"
                return await self._start(body)
            finally:
                self.preparing = None

    async def ensure_online(self):
        try:
            response = await self.upstream.get("/status", timeout=3)
            response.raise_for_status()
            status = response.json()
        except (httpx.HTTPError, ValueError):
            if not self.online:
                raise RuntimeError("producer_unavailable") from None
            self.preparing = "Bringing the comma online"
            await asyncio.to_thread(self.online)
            response = await self.upstream.get("/status", timeout=5)
            response.raise_for_status()
            status = response.json()
        if status.get("protocol") != PROTOCOL:
            raise RuntimeError("protocol_mismatch: a coordinated device deployment is required")

    async def _start(self, body: dict) -> dict:
        epoch = self.preparation_epoch
        if self.running:
            raise RuntimeError("already_running")
        keys = ("supervisor", "tier", "driver", "objective", "prompt", "reflection", "continuation")
        config = {key: str(body.get(key) or "").strip() for key in keys}
        config["supervisor"] = config["supervisor"] or NO_SUPERVISOR
        config["tier"] = config["tier"] or "normal"
        config["driver"] = config["driver"] or NO_DRIVER
        supervisor, driver = config["supervisor"], config["driver"]
        supervised, driving = supervisor != NO_SUPERVISOR, driver != NO_DRIVER
        spec = self.policies.get(driver)
        needed = ["objective"] + (["prompt", "reflection", "continuation"] if supervised else [])
        if not all(config[key] for key in needed) or config["tier"] not in TIERS:
            raise ValueError(f"{', '.join(needed)} are required")
        if not (supervised or driving):
            raise ValueError("choose a chat model, a driver, or both")
        if supervisor not in SUPERVISORS and not gemini.is_model(supervisor):
            raise ValueError(f"unknown chat model {supervisor}")
        if gemini.is_model(supervisor):
            gemini.service_tier(config["tier"])
        if driving and not spec and driver not in DRIVERS:
            raise ValueError(f"unknown driver {driver}")
        if spec and supervised and not spec.steerable:
            raise ValueError("only a steerable trained driver takes a chat model")
        if spec:  # the instruction it was trained on, or its first command
            config["brief"] = vla.COMMANDS[0] if spec.steerable else config["objective"]
        else:
            brief = str(body.get("brief") or "").strip() or DEFAULT_BRIEF.format(**config)
            config["brief"] = brief[:BRIEF_MAX_CHARS]
        decide, chat = self.clients(supervisor, driver)
        await self.ensure_online()
        if epoch != self.preparation_epoch:
            raise RuntimeError("start_cancelled_by_stop")
        if spec:
            factory = partial(vla.build, driver, spec)
        else:
            factory = partial(DecisionPolicy, decide, driver) if driving else None
        policy = factory() if factory else None
        try:
            if spec:  # a broken policy fails before the session
                self.preparing = "Checking the trained driver"
                try:
                    await asyncio.to_thread(policy.probe, config["brief"])
                except Exception as exc:
                    raise RuntimeError(f"{driver}: {type(exc).__name__}: {exc}"[:300]) from exc
            if epoch != self.preparation_epoch:
                raise RuntimeError("start_cancelled_by_stop")
            run = uuid4().hex[:8]
            self.preparing = "Starting the evaluation"
            names = [n.split("/")[-1] for n in (supervisor, driver) if n != NO_DRIVER]
            first = f"{names[0]} ({config['tier']})" if supervised else names[0]
            label = " + ".join([first, *names[1:]])
            opened = await command(  # fails before anything starts if the comma is unreachable
                self.upstream,
                "runner",
                "session/start",
                {"model": label[:80], "harness": "other", "notes": f"run {run} on the runner page"},
            )
            if "error" in opened:
                raise RuntimeError(f"session_start_failed: {opened['error']}")
            if epoch != self.preparation_epoch:
                await command(
                    self.upstream,
                    "runner",
                    "session/end",
                    {"outcome": "aborted", "note": "Start cancelled by operator Stop"},
                )
                raise RuntimeError("start_cancelled_by_stop")
        except BaseException:
            await self.close_policy(policy)
            raise
        self.metadata_error, self.evaluation_session = None, None
        self.run, self.config, self.attempt, self.policy = run, config, 1, factory
        if self.runs:
            folder = self.runs / run
            folder.mkdir(parents=True)
            self.log = (folder / "events.jsonl").open("a")
        self.record("start", **config)
        self.record("session_start", label=label, result=opened)
        self.record("attempt_start", attempt=1)
        self.in_attempt = True
        if driving:
            self.driver.start(policy, config["brief"])
        if supervised:
            await self.supervisor.start(
                chat, supervisor, config["tier"], config["prompt"], not driving
            )
        return self.status()

    async def end_attempt(self, outcome="unknown", note="") -> dict:
        if not self.running or not self.in_attempt:
            raise RuntimeError("no_attempt_running")
        if outcome not in ATTEMPT_OUTCOMES or len(note) > 500:
            raise ValueError("Invalid attempt result or note")
        async with self.command_gate:  # waits for in-flight chat commands; later ones are refused
            self.in_attempt = False
        if self.supervised:
            await self.supervisor.interrupt()  # nothing may command the car while it is reset
        await self.driver.stop(f"attempt {self.attempt} ended by the operator")  # stops the car
        self.record("attempt_end", attempt=self.attempt, outcome=outcome, note=note)
        if self.supervised:
            await self.supervisor.say(self.config["reflection"], "reflection")
        return self.status()

    async def next_attempt(self) -> dict:
        if not self.running or self.in_attempt:
            raise RuntimeError("attempt_in_progress" if self.running else "not_running")
        self.attempt += 1
        self.in_attempt = True
        self.record("attempt_start", attempt=self.attempt)
        if self.policy:
            await self.close_policy(self.driver.policy)
            self.driver.start(self.policy(), self.config["brief"])
        if self.supervised:
            await self.supervisor.say(self.config["continuation"], "continuation")
        return self.status()

    async def message(self, text: str) -> dict:
        if not self.running or not self.supervised:
            raise RuntimeError("not_running" if not self.running else "no_supervisor")
        await self.supervisor.say(text, "operator")
        return self.status()

    async def stop(
        self, outcome: str, note: str = "", attempt_outcome="unknown", attempt_note=""
    ) -> dict:
        if not self.running:
            raise RuntimeError("not_running")
        await self.supervisor.stop()
        await self.driver.stop("operator ended the run")
        if self.in_attempt:
            self.in_attempt = False
            self.record(
                "attempt_end", attempt=self.attempt, outcome=attempt_outcome, note=attempt_note
            )
        try:
            session = await command(
                self.upstream, "runner", "session/end", {"outcome": outcome, "note": note[:200]}
            )
        except (httpx.HTTPError, ValueError) as exc:  # end it in the UI once the comma is back
            session = {"error": type(exc).__name__}
        self.record(
            "stop",
            outcome=outcome,
            note=note,
            cost_usd=self.supervisor.status()["cost_usd"],
            session=session,
        )
        if self.log:
            self.log.close()
            self.log = None
        self.run, self.in_attempt = None, False
        await self.close_policy(self.driver.policy)
        return self.status()


def runner_routes(app: FastAPI, runner: Runner):
    async def act(action):
        try:
            return await action()
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except RuntimeError as exc:
            status = 503 if str(exc) in MISSING_KEYS else 409
            return JSONResponse({"error": str(exc)}, status_code=status)
        except httpx.HTTPError:
            return JSONResponse({"error": "producer_unavailable"}, status_code=503)

    @app.get("/runner/options")
    async def options():
        return {
            "supervisors": list(SUPERVISORS),
            "tiers": list(TIERS),
            "drivers": [NO_DRIVER, *DRIVERS, *runner.policies],
            "decision_drivers": list(DRIVERS),
            "trained": {  # the instruction each was trained on, prefilled as the objective
                name: {"steerable": spec.steerable, "objective": spec.objective}
                for name, spec in runner.policies.items()
            },
            "outcomes": list(OUTCOMES),
            "key_present": bool(runner.injected[1]) or keychain_has_key(),
            "keychain_item": KEYCHAIN_ITEM,
            "gemini_models": list(gemini.DEFAULT_MODELS),
            "gemini_key_present": keychain_has_key(gemini.KEYCHAIN_ITEM),
            "gemini_keychain_item": gemini.KEYCHAIN_ITEM,
        }

    async def google(action):
        key = keychain_key(gemini.KEYCHAIN_ITEM)
        if not key:
            return JSONResponse({"error": "no_gemini_key"}, status_code=503)
        api = gemini.Gemini(key)
        try:
            return await action(api)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except (RuntimeError, httpx.HTTPError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)
        finally:
            await api.close()

    @app.get("/runner/gemini/models")
    async def google_models():
        async def discover(api):
            return {"models": await api.models()}

        return await google(discover)

    @app.post("/runner/gemini/probe")
    async def google_probe(request: Request):
        body = await request.json()
        model, tier = str(body.get("model", "")), str(body.get("tier", "normal"))
        try:
            gemini.model_id(model)
            gemini.service_tier(tier)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return await google(lambda api: api.probe(model, tier))

    @app.get("/runner/status")
    async def status():
        return runner.status()

    @app.post("/runner/start")
    async def start(request: Request):
        body = await request.json()
        return await act(lambda: runner.start(body))

    @app.post("/runner/end-attempt")
    async def end_attempt(request: Request):
        body = await request.json() if await request.body() else {}
        return await act(
            lambda: runner.end_attempt(body.get("outcome", "unknown"), str(body.get("note", "")))
        )

    @app.post("/runner/next-attempt")
    async def next_attempt():
        return await act(runner.next_attempt)

    @app.post("/runner/message")
    async def message(request: Request):
        text = str((await request.json()).get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "text_required"}, status_code=422)
        return await act(lambda: runner.message(text))

    @app.post("/runner/resume")
    async def resume():
        if not runner.driver.running:
            return JSONResponse({"error": "driver_not_running"}, status_code=409)
        runner.driver.resume()
        return runner.status()

    @app.post("/runner/stop")
    async def stop(request: Request):
        body = await request.json()
        outcome = body.get("outcome")
        if outcome not in OUTCOMES:
            return JSONResponse({"error": "outcome_required"}, status_code=422)
        attempt_outcome = body.get("attempt_outcome", "unknown")
        attempt_note = str(body.get("attempt_note", ""))
        if attempt_outcome not in ATTEMPT_OUTCOMES or len(attempt_note) > 500:
            return JSONResponse({"error": "invalid_attempt_result"}, status_code=422)
        return await act(
            lambda: runner.stop(outcome, str(body.get("note") or ""), attempt_outcome, attempt_note)
        )
