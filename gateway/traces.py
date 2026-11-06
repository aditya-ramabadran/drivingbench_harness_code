"""Publish the comma's recordings into the repository's traces/ folder.

Mirror layout (device recordings, rsynced into the ignored runs/recordings):
    segments/<segment>/events.jsonl, thumbs/    sessions/<session>.json    images/<id>.jpg

Published layout (committed):
    traces/segments/<segment>/{events.jsonl, thumbs/, chat/}
    traces/sessions/<session>.json      fields, timing, outcome, segments, artifacts

Nothing here rewrites or deletes what was published before.
"""

import base64
import glob
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

SEGMENT_ID = re.compile(r"\d{4}-\d{2}-\d{2}-[0-9a-f]{6}")
TRANSCRIPT_ROOTS = {
    "codex": "~/.codex/sessions/*/*/*/rollout-*.jsonl",
    "claude": "~/.claude/projects/*/*.jsonl",
    "cursor": "~/.cursor/projects/*/agent-transcripts/*/*.jsonl",
}


def transcript_roots(config=None) -> dict[str, str]:
    """Default client locations, $CODEX_HOME, Runner logs and the laptop config's extra roots."""
    roots = dict(TRANSCRIPT_ROOTS)
    if home := os.environ.get("CODEX_HOME"):
        roots["codex-home"] = f"{home}/sessions/*/*/*/rollout-*.jsonl"
    if runs := getattr(config, "runs", None):
        roots["runner"] = f"{runs}/runner/*/events.jsonl"
    roots.update(getattr(config, "extra_transcripts", None) or {})
    return roots


def read_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass  # A partial final line means the device is still writing.
    return events


def publish_segments(mirror: Path, traces: Path) -> dict:
    """Copy each finished segment (has segment_end); report open ones and conflicts."""
    published, running, conflicts = [], [], []
    for folder in sorted((mirror / "segments").glob("*")) if (mirror / "segments").is_dir() else []:
        events = folder / "events.jsonl"
        if not SEGMENT_ID.fullmatch(folder.name) or not events.is_file():
            continue
        if not any(event.get("kind") == "segment_end" for event in read_events(events)):
            running.append(folder.name)
            continue
        destination = traces / "segments" / folder.name
        target = destination / "events.jsonl"
        if target.exists() and target.read_bytes() != events.read_bytes():
            conflicts.append(folder.name)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(events.read_bytes())
        for thumb in sorted((folder / "thumbs").glob("*.jpg")):
            (destination / "thumbs").mkdir(exist_ok=True)
            if not (destination / "thumbs" / thumb.name).exists():
                (destination / "thumbs" / thumb.name).write_bytes(thumb.read_bytes())
        published.append(folder.name)
    return {"published": published, "running": running, "conflicts": conflicts}


def publish_sessions(mirror: Path, traces: Path) -> dict:
    """Copy ended labeled sessions; the comma's fields win, published artifacts are kept."""
    published, running = [], []
    for path in (
        sorted((mirror / "sessions").glob("*.json")) if (mirror / "sessions").is_dir() else []
    ):
        session = json.loads(path.read_text())
        if not session.get("ended_at"):
            running.append(session["id"])
            continue
        target = traces / "sessions" / f"{session['id']}.json"
        existing = json.loads(target.read_text()) if target.exists() else {}
        merged = {**session, "artifacts": existing.get("artifacts")}
        if existing != merged:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(merged, indent=2) + "\n")
        published.append(session["id"])
    return {"published": published, "running": running}


def label_session(
    traces: Path, segments: list[str], model, harness, notes, outcome, note="", attempts=""
):
    """Post-hoc labeling from already published segments, for the forgotten-to-start case."""
    from drivingbench.device.recording import slug
    from drivingbench.shared.contracts import SessionEnd, SessionLabel

    label, end = (
        SessionLabel(model=model, harness=harness, notes=notes),
        SessionEnd(outcome=outcome, note=note, attempts=attempts),
    )
    times = []
    for segment in segments:
        events = traces / "segments" / segment / "events.jsonl"
        if not events.is_file():
            raise ValueError(f"{segment} is not a published segment")
        times += [event["timestamp"] for event in read_events(events)]
    started, ended = min(times), max(times)
    session = {
        "id": f"{started[:10]}-{slug(label.model)}-{slug(label.harness)}-{hashlib.sha256(''.join(segments).encode()).hexdigest()[:6]}",
        **label.model_dump(),
        "started_at": started,
        "ended_at": ended,
        **end.model_dump(),
        "segments": list(segments),
        "labeled_after": True,
        "artifacts": None,
    }
    target = traces / "sessions" / f"{session['id']}.json"
    if target.exists():
        raise ValueError(f"{target} already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(session, indent=2) + "\n")
    return session


def chat_keys(segment_folder: Path) -> dict[str, set[str]]:
    """Strings a driving transcript must quote: observe response timestamps (producer-made,
    microsecond, ASCII, unique) and the reasons the model sent (optional, may repeat)."""
    keys = {"timestamps": set(), "reasons": set()}
    for event in read_events(segment_folder / "events.jsonl"):
        if event.get("kind") != "tool":
            continue
        if event.get("tool") == "observe" and (event.get("outcome") or {}).get("timestamp"):
            keys["timestamps"].add(event["outcome"]["timestamp"])
        reason = (event.get("arguments") or {}).get("reason", "")
        if event.get("tool") in {"set_motion", "stop_now"} and len(reason) >= 12:
            keys["reasons"].add(reason)
    return keys


def quoted(text: str, key: str) -> bool:
    """Transcripts store tool arguments as JSON strings, so match the escaped form too."""
    return key in text or json.dumps(key)[1:-1] in text


def find_chats_many(
    folders: dict[str, Path], roots: dict[str, str] | None = None
) -> dict[str, list]:
    """Scan each candidate transcript once for the whole batch, with recorded match evidence."""
    facts = {}
    for name, folder in folders.items():
        events = read_events(folder / "events.jsonl")
        facts[name] = {
            "keys": chat_keys(folder),
            "started": datetime.fromisoformat(events[0]["timestamp"]).timestamp() if events else 0,
            "sessions": {e["session"] for e in events if e.get("session")},
        }
    found = {name: [] for name in folders}
    for client, pattern in (roots or transcript_roots()).items():
        for name in sorted(glob.glob(os.path.expanduser(pattern))):
            path = Path(name)
            try:
                modified = path.stat().st_mtime
                candidates = {
                    name: fact
                    for name, fact in facts.items()
                    if modified >= fact["started"]
                    and (any(fact["keys"].values()) or (client == "runner" and fact["sessions"]))
                }
                if not candidates:
                    continue
                evidence = {name: {"timestamps": set(), "reasons": set()} for name in candidates}
                # App histories can be gigabytes. Read bounded JSONL records, and stop once
                # this file establishes every candidate match; do not load the whole history.
                with path.open(errors="replace") as stream:
                    for line in stream:
                        session_id = None
                        if client == "runner":
                            try:
                                event = json.loads(line)
                                if event.get("kind") == "session_start":
                                    session_id = (event.get("result") or {}).get("id")
                            except ValueError:
                                pass
                        for segment, fact in list(candidates.items()):
                            seen = evidence[segment]
                            for kind, values in fact["keys"].items():
                                seen[kind].update(
                                    key for key in values - seen[kind] if quoted(line, key)
                                )
                            matches = {kind: len(values) for kind, values in seen.items()}
                            if session_id and session_id in fact["sessions"]:
                                matches["sessions"] = 1
                            if (
                                matches["timestamps"] >= 1
                                or matches["reasons"] >= 2
                                or matches.get("sessions")
                            ):
                                found[segment].append((client, path, matches))
                                del candidates[segment]
                        if not candidates:
                            break
            except OSError:
                continue
    return found


def find_chats(segment_folder: Path, roots: dict[str, str] | None = None) -> list:
    """One exact observe timestamp, two reasons, or a Runner's recorded session ID."""
    return find_chats_many({segment_folder.name: segment_folder}, roots)[segment_folder.name]


BASE64_IMAGE = re.compile(r"(?:data:image/[a-z]+;base64,)?([A-Za-z0-9+/]{800,}={0,2})")


def strip_images(text: str, known: dict[str, str]) -> str:
    """Replace inline base64 images with references to recorded frames (by content hash)."""

    def replace(match):
        try:
            digest = hashlib.sha256(base64.b64decode(match.group(1), validate=True)).hexdigest()
        except ValueError:
            return match.group(0)
        return f"[image {known.get(digest, 'unrecorded-' + digest[:12])}]"

    return BASE64_IMAGE.sub(replace, text)


def recorded_image_hashes(mirror: Path) -> dict[str, str]:
    images = mirror / "images"
    return (
        {hashlib.sha256(p.read_bytes()).hexdigest(): p.name for p in images.glob("*.jpg")}
        if images.is_dir()
        else {}
    )


def attach_chat(traces: Path, segment: str, transcript: Path, client: str, known=None) -> dict:
    """File the driving chat's transcript, images stripped to references; never overwrite."""
    folder = traces / "segments" / segment
    if not (folder / "events.jsonl").is_file():
        raise ValueError(f"{segment} is not a published segment under {traces}")
    transcript = transcript.expanduser().resolve()
    # Bound the snapshot at the initial file length; a live conversation may append later.
    digest = hashlib.sha256()
    with tempfile.TemporaryFile() as content:
        with transcript.open("rb") as source:
            remaining = transcript.stat().st_size
            while remaining:
                line = source.readline(remaining)
                if not line:
                    raise ValueError(f"Transcript truncated during snapshot: {transcript}")
                remaining -= len(line)
                encoded = strip_images(line.decode(errors="replace"), known or {}).encode()
                digest.update(encoded)
                content.write(encoded)
        name = (
            f"{transcript.parent.name}-{transcript.name}" if client == "runner" else transcript.name
        )
        destination = folder / "chat" / f"{client}-{name}"
        if destination.exists():
            with destination.open("rb") as existing:
                if hashlib.file_digest(existing, "sha256").digest() == digest.digest():
                    return {"status": "unchanged", "chat": str(destination)}
            destination = destination.with_name(
                f"{destination.stem}-{digest.hexdigest()[:16]}{destination.suffix}"
            )
            if destination.exists():
                with destination.open("rb") as existing:
                    if hashlib.file_digest(existing, "sha256").digest() != digest.digest():
                        raise ValueError(f"Transcript snapshot hash collision: {destination}")
                return {"status": "unchanged", "chat": str(destination)}
        destination.parent.mkdir(exist_ok=True)
        content.seek(0)
        with destination.open("xb") as target:
            shutil.copyfileobj(content, target)
    return {"status": "attached", "chat": str(destination)}
