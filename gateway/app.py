"""Laptop HTTP proxy and static UI. The comma owns all shared state and motion."""

import inspect
import json
import re
import subprocess
from contextlib import asynccontextmanager, nullcontext
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from drivingbench.gateway.evaluations import EvaluationBook, evaluation_routes
from drivingbench.gateway.traces import SEGMENT_ID
from drivingbench.harness.runner import Runner, runner_routes
from drivingbench.shared.contracts import PROTOCOL
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware


def trace_routes(app: FastAPI, repo: Path | None):
    """Read-only access to published traces and fetched artifacts for the trace viewer."""
    traces = repo / "traces" if repo else None
    frames = repo / "runs/artifacts" if repo else None
    image_name = re.compile(r"[0-9a-f]{32}\.jpg")

    @app.get("/traces")
    async def viewer():
        return FileResponse(str(files("drivingbench").joinpath("ui", "traces.html")))

    @app.get("/traces/api/index")
    async def index():
        if not traces or not traces.is_dir():
            return {"segments": [], "sessions": []}
        sessions = [
            json.loads(path.read_text()) for path in sorted((traces / "sessions").glob("*.json"))
        ]
        segments = []
        for folder in sorted((traces / "segments").glob("*")):
            if not SEGMENT_ID.fullmatch(folder.name) or not (folder / "events.jsonl").is_file():
                continue
            lines = (folder / "events.jsonl").read_text().splitlines()
            first, last = json.loads(lines[0]), json.loads(lines[-1])
            segments.append(
                {
                    "id": folder.name,
                    "started_at": first["timestamp"],
                    "ended_at": last["timestamp"],
                    "events": len(lines),
                    "tool_calls": sum(
                        '"kind": "tool"' in line and '"tool": "chunk"' not in line for line in lines
                    ),
                    "plans": sum('"tool": "chunk"' in line for line in lines),  # a policy's chunks
                    "session": next(
                        (s["id"] for s in sessions if folder.name in s.get("segments", [])), None
                    ),
                    "chat": bool(list((folder / "chat").glob("*"))),
                }
            )
        segments.sort(key=lambda s: s["started_at"], reverse=True)
        return {"segments": segments, "sessions": sessions[::-1]}

    @app.get("/traces/api/segments/{segment}/events")
    async def events(segment: str):
        path = traces / "segments" / segment / "events.jsonl" if traces else None
        if not SEGMENT_ID.fullmatch(segment) or not path or not path.is_file():
            return JSONResponse({"error": "unknown_segment"}, status_code=404)
        return FileResponse(str(path), media_type="application/x-ndjson")

    @app.get("/traces/api/segments/{segment}/images/{name}")
    async def image(segment: str, name: str):
        """A fetched full-resolution frame when present locally, else the committed thumbnail."""
        if not traces or not SEGMENT_ID.fullmatch(segment) or not image_name.fullmatch(name):
            return JSONResponse({"error": "unknown_image"}, status_code=404)
        candidates = [
            *(frames.glob(f"**/frames/{name}") if frames and frames.is_dir() else []),
            traces / "segments" / segment / "thumbs" / name,
        ]
        for path in candidates:
            if path.is_file():
                return FileResponse(str(path), media_type="image/jpeg")
        return JSONResponse({"error": "unknown_image"}, status_code=404)


def create_app(
    producer_url="http://127.0.0.1:8877",
    *,
    client=None,
    online=None,
    install_status=None,
    repo: Path | None = None,
    runs: Path | None = None,
    decider=None,
    chat=None,
    policies=None,
    diagnostics=None,
):
    owns_client = client is None
    upstream = client or httpx.AsyncClient(base_url=producer_url, timeout=60, trust_env=False)
    book = EvaluationBook(runs) if runs else None
    runner = Runner(
        upstream, runs and runs / "runner", decider, chat, policies, evaluations=book, online=online
    )

    @asynccontextmanager
    async def lifespan(app):
        yield
        if runner.running:
            await runner.stop("aborted", "gateway shutting down")
        if owns_client:
            await upstream.aclose()

    app = FastAPI(title="DrivingBench Harness", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc != request.headers.get(
                "host"
            ):
                return JSONResponse({"error": "foreign_origin"}, status_code=403)
        response = await call_next(request)
        # The UI is tiny and updated with the install; a stale cached script is worse than a fetch.
        response.headers["Cache-Control"] = "no-store"
        return response

    async def local_action(callback, fallback):
        if callback is None:
            return fallback
        try:
            result = await run_in_threadpool(callback)
            return await result if inspect.isawaitable(result) else result
        except (OSError, RuntimeError, httpx.HTTPError, subprocess.SubprocessError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)

    @app.post("/api/online")
    async def bring_online():
        if online:
            return await local_action(online, {})
        try:
            response = await upstream.get("/status")
            response.raise_for_status()
            status = response.json()
            return {
                "status": "online",
                "release": status.get("release"),
                "protocol": status.get("protocol"),
            }
        except (httpx.HTTPError, ValueError):
            return JSONResponse({"error": "producer_unavailable"}, status_code=503)

    @app.get("/api/install/status")
    async def installed():
        return await local_action(install_status, {"clients": {}})

    @app.get("/api/doctor")
    async def diagnose():
        return await local_action(diagnostics, {"checks": [], "errors": 0})

    if book:
        evaluation_routes(app, book, upstream)

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PATCH"])
    async def proxy(path: str, request: Request):
        allowed = {
            "GET": {"status", "observe", "history", "camera/narrow", "camera/wide"},
            "POST": {"motion", "chunk", "stop", "session/start", "session/end"},
            "PATCH": {"settings"},
        }
        recorded_image = request.method == "GET" and re.fullmatch(
            r"recordings/[a-f0-9]{32}\.jpg", path
        )
        if path not in allowed[request.method] and not recorded_image:
            return JSONResponse({"error": "unknown_endpoint"}, status_code=404)
        headers = {
            "X-Client": request.headers.get("X-Client", "UI")[:128],
            "X-Drivingbench-Protocol": str(PROTOCOL),
        }
        # In-flight motion holds the gate from this check to its reply, so End attempt's
        # stop always lands after it and nothing passes the check afterwards.
        commanding = path in {"motion", "chunk"}
        async with runner.command_gate if commanding else nullcontext():
            if commanding and runner.between_attempts and headers["X-Client"] != "UI":
                return JSONResponse({"error": "between_attempts"}, status_code=409)
            try:
                body = None
                if request.method != "GET":
                    body = await request.json()
                    if not isinstance(body, dict):
                        raise ValueError("expected object")
                    if request.method == "POST":
                        headers["X-Request-ID"] = request.headers.get("X-Request-ID") or uuid4().hex
                        body["request_id"] = headers["X-Request-ID"]
                # A mismatched producer stays inspectable and stoppable, but cannot start motion.
                if path in {"motion", "chunk"}:
                    status = await upstream.get("/status")
                    status.raise_for_status()
                    if status.json().get("protocol") != PROTOCOL:
                        return JSONResponse({"error": "protocol_mismatch"}, status_code=409)
                if path == "stop":  # pause first, so no driver command lands after the stop
                    runner.cancel_preparation()
                    runner.driver.yield_to(headers["X-Client"], path, body)
                response = await upstream.request(
                    request.method,
                    "/" + path,
                    json=body,
                    headers=headers,
                    params=request.query_params,
                )
                if book and response.is_success and path in {"session/start", "session/end"}:
                    session = response.json()
                    if path.endswith("start"):
                        mode = (
                            "independent"
                            if session.get("model", "").startswith("Human demonstration")
                            else "same_context"
                        )
                        book.start(session, mode=mode)
                    elif session.get("id"):
                        book.end(
                            session["id"],
                            outcome=body.get("outcome", "unknown"),
                            note=body.get("note", ""),
                        )
                if response.is_success and path in {"motion", "chunk"}:
                    runner.driver.yield_to(headers["X-Client"], path, body)
                return Response(
                    response.content,
                    status_code=response.status_code,
                    headers={
                        key: value
                        for key, value in response.headers.items()
                        if key.lower()
                        in {
                            "content-type",
                            "x-frame-id",
                            "x-image-age-s",
                            "x-captured-at",
                        }
                    }
                    | {"Cache-Control": "no-store"},
                )
            except ValueError:
                return JSONResponse({"error": "invalid_request_or_response"}, status_code=400)
            except httpx.HTTPError:
                reason = "producer_unavailable" if request.method == "GET" else "outcome_unknown"
                return JSONResponse({"error": reason}, status_code=503)

    assets = str(files("drivingbench").joinpath("ui"))
    app.mount("/assets", StaticFiles(directory=assets), name="assets")
    trace_routes(app, repo)
    runner_routes(app, runner)

    @app.get("/")
    async def index():
        return FileResponse(str(files("drivingbench").joinpath("ui", "index.html")))

    return app
