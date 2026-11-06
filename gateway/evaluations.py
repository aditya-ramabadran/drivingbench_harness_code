"""Trial and attempt markers with the operator's labels, kept on the laptop.

Never part of motion or a published device trace: <runs>/evaluations/events.jsonl (the laptop
config's runs folder).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from drivingbench.gateway.install import atomic_write
from drivingbench.gateway.traces import read_events
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class EvaluationBook:
    def __init__(self, runs: Path):
        self.folder = runs / "evaluations"
        self.log = self.folder / "events.jsonl"

    def context(self) -> dict:
        path = self.folder / "context.json"
        return json.loads(path.read_text()) if path.exists() else {"course": "", "operator": ""}

    def save_context(self, body: dict) -> dict:
        if set(body) != {"course", "operator"} or not all(
            isinstance(v, str) and len(v) <= 120 for v in body.values()
        ):
            raise ValueError("course and operator must be strings under 120 characters")
        body = {k: v.strip() for k, v in body.items()}
        atomic_write(self.folder / "context.json", json.dumps(body).encode())
        return body

    def events(self) -> list[dict]:
        return read_events(self.log) if self.log.exists() else []

    def append(self, kind: str, session: str, **fields):
        self.folder.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "session": session,
            **fields,
        }
        with self.log.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def state(self, session: str) -> dict:
        events = [e for e in self.events() if e["session"] == session]
        starts = [e for e in events if e["kind"] == "attempt_start"]
        last = next(
            (
                e
                for e in reversed(events)
                if e["kind"] in {"attempt_start", "attempt_end", "trial_end"}
            ),
            {},
        )
        return {
            "session": session,
            "attempt": starts[-1]["attempt"] if starts else 0,
            "in_attempt": last.get("kind") == "attempt_start",
            "events": events,
        }

    def start(self, session: dict, mode="same_context", source="chat_app"):
        if not session.get("id") or self.state(session["id"])["events"]:
            return
        self.append(
            "trial_start",
            session["id"],
            context=self.context(),
            context_mode=mode,
            source=source,
            model=session.get("model"),
            harness=session.get("harness"),
        )
        self.marker(session["id"], "start")

    def marker(self, session: str, action: str, *, outcome="unknown", note="", request_id=None):
        events = self.state(session)
        if request_id and any(e.get("request_id") == request_id for e in events["events"]):
            return events
        if not events["events"] or events["events"][-1]["kind"] == "trial_end":
            raise ValueError("No active evaluation trial on this laptop")
        if (
            action not in {"start", "end"}
            or outcome
            not in {
                "unknown",
                "completed",
                "intervention",
                "aborted",
                "collision",
            }
            or not isinstance(note, str)
            or len(note) > 500
        ):
            raise ValueError("Invalid attempt action, outcome or note")
        if events["in_attempt"] == (action == "start"):
            raise ValueError(
                "End the current attempt before starting another"
                if action == "start"
                else "This attempt has already ended"
            )
        self.append(
            "attempt_" + action,
            session,
            attempt=events["attempt"] + (action == "start"),
            outcome=outcome,
            note=note,
            request_id=request_id,
        )
        return self.state(session)

    def end(self, session: str, *, outcome="unknown", note=""):
        state = self.state(session)
        if not state["events"] or state["events"][-1]["kind"] == "trial_end":
            return
        if state["in_attempt"]:
            # A trial result does not establish the last attempt's result.
            self.marker(session, "end", note=note)
        self.append("trial_end", session, outcome=outcome, note=note)


def evaluation_routes(app: FastAPI, book: EvaluationBook, upstream):
    @app.get("/api/evaluations/context")
    async def context():
        return book.context()

    @app.patch("/api/evaluations/context")
    async def save_context(request: Request):
        try:
            return book.save_context(await request.json())
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)

    @app.post("/api/evaluations/attempt")
    async def marker(request: Request):
        try:
            body = await request.json()
            response = await upstream.get("/status")
            response.raise_for_status()
            session = response.json().get("session") or {}
            if not session.get("id"):
                raise ValueError("Start a trial first")
            return book.marker(
                session["id"],
                body.get("action"),
                outcome=body.get("outcome", "unknown"),
                note=body.get("note", ""),
                request_id=request.headers.get("X-Request-ID"),
            )
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
