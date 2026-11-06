"""Bulk artifacts for labeled sessions: full-resolution frames and openpilot's own
camera/CAN route segments, uploaded to a Hugging Face dataset and referenced by hash
from the session manifest in traces/. Only labeled sessions ever upload.
"""

import hashlib
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

from drivingbench.gateway.setup import Config, ssh_args
from drivingbench.gateway.traces import read_events

ROUTES = "/data/media/0/realdata"
SEGMENT_SECONDS = 60  # openpilot writes one route segment per minute
ROUTE_FILES = ("fcamera.hevc", "ecamera.hevc", "rlog.zst")  # road and wide video, full log
SETTLED_S = 2 * SEGMENT_SECONDS  # a route minute this old is no longer being written


def epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def frame_ids(traces: Path, session: dict) -> list[str]:
    names = []
    for segment in session["segments"]:
        for event in read_events(traces / "segments" / segment / "events.jsonl"):
            for image in event.get("images") or []:
                names.append(image["url"].rsplit("/", 1)[-1])
    return sorted(set(names))


def route_times(listing: str) -> dict[str, float]:
    """From `find -printf '%T@ %f\\n'` output, each route minute's last write time."""
    times = {}
    for line in listing.splitlines():
        try:
            finished, name = line.split(maxsplit=1)
            times[name.strip()] = float(finished)
        except ValueError:
            continue
    return times


def overlapping_routes(listing: str, started: float, ended: float, margin=SEGMENT_SECONDS) -> list:
    """The route minutes that overlap the session."""
    return sorted(
        name
        for name, finished in route_times(listing).items()
        if finished - SEGMENT_SECONDS - margin <= ended and finished + margin >= started
    )


def list_routes(config: Config) -> str:
    command = (
        f"find {ROUTES} -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\\n' 2>/dev/null || true"
    )
    return subprocess.run(
        ssh_args(config) + [command], capture_output=True, text=True, check=True
    ).stdout


def rsync(config: Config, remote: str, local: Path):
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--partial",
            *(f"--include={name}" for name in ROUTE_FILES),
            "--exclude=*",
            "-e",
            shlex.join(ssh_args(config)[:-1]),
            "--",
            f"{config.ssh_host}:{remote}",
            str(local) + "/",
        ],
        check=True,
    )


def stage(
    config: Config, traces: Path, mirror: Path, session: dict, staging: Path, listing: str, now=None
) -> Path:
    """Collect a session's bulk artifacts locally: frames from the mirror, routes from the comma
    (`listing`: list_routes, read once per sync).

    The only step that needs the comma. Each route minute is pulled once into `staging/routes/`,
    marked complete once it is settled, and hard-linked into every session that overlaps it, so
    shared minutes cost one transfer and later syncs none.
    """
    now = time.time() if now is None else now
    folder = staging / session["id"]
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    for name in frame_ids(traces, session):
        source = mirror / "images" / name
        if source.is_file() and not (folder / "frames" / name).exists():
            (folder / "frames" / name).write_bytes(source.read_bytes())
    times = route_times(listing)
    for route in overlapping_routes(
        listing, epoch(session["started_at"]), epoch(session["ended_at"])
    ):
        cache = staging / "routes" / route
        if not (cache / ".complete").exists():
            rsync(config, f"{ROUTES}/{shlex.quote(route)}/", cache)
            if now - times[route] >= SETTLED_S:
                (cache / ".complete").touch()
        link_tree(cache, folder / "routes" / route)
    return folder


def pending(traces: Path, session_ids) -> list[dict]:
    """Labeled sessions whose artifacts have not been uploaded yet."""
    sessions = [
        json.loads((traces / "sessions" / f"{sid}.json").read_text()) for sid in session_ids
    ]
    return [s for s in sessions if not s.get("artifacts")]


def link_tree(source: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.is_file() and not file.name.startswith(".") and not (target / file.name).exists():
            (target / file.name).hardlink_to(file)


def inventory(folder: Path) -> list[dict]:
    return [
        {
            "path": path.relative_to(folder).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    ]


def upload(config: Config, traces: Path, staging: Path) -> list[dict]:
    """Upload every staged labeled session not yet on the dataset; needs no comma.

    Each session folder lands at `sessions/<id>/` on the dataset and its manifest in
    traces/ records the location and every file's hash.
    """
    from huggingface_hub import HfApi

    if not config.dataset:
        return [{"status": "no_dataset_configured"}]
    api = HfApi()
    report = []
    for manifest in sorted((traces / "sessions").glob("*.json")):
        session = json.loads(manifest.read_text())
        folder = staging / session["id"]
        if session.get("artifacts") or not folder.is_dir():
            continue
        path = f"sessions/{session['id']}"
        files = inventory(folder)
        api.upload_folder(
            folder_path=str(folder), path_in_repo=path, repo_id=config.dataset, repo_type="dataset"
        )
        session["artifacts"] = {"repo": config.dataset, "path": path, "files": files}
        manifest.write_text(json.dumps(session, indent=2) + "\n")
        report.append(
            {
                "status": "uploaded",
                "path": path,
                "files": len(files),
                "bytes": sum(f["bytes"] for f in files),
            }
        )
    return report


def fetch(session_id: str, traces: Path, destination: Path) -> dict:
    """Download a session's bulk artifacts from where its manifest says they were uploaded."""
    from huggingface_hub import snapshot_download

    session = json.loads((traces / "sessions" / f"{session_id}.json").read_text())
    artifacts = session.get("artifacts") or {}
    if not artifacts.get("path"):
        raise ValueError(f"{session_id} has no uploaded artifacts")
    snapshot_download(
        repo_id=artifacts["repo"],
        repo_type="dataset",
        allow_patterns=[f"{artifacts['path']}/*"],
        local_dir=str(destination),
    )
    return {"status": "fetched", "path": str(destination / artifacts["path"])}
