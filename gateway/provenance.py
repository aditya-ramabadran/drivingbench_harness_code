"""Source receipts: what install copied, compared by doctor with the checkout."""

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SOURCE_DIRS = ("controller", "device", "gateway", "harness", "mcp", "shared", "ui")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_state(repo: Path) -> dict:
    def git(*args):
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    paths = [repo / name for name in ("__init__.py", "pyproject.toml", "uv.lock")]
    paths += [
        path
        for name in SOURCE_DIRS
        for path in (repo / name).rglob("*")
        if path.suffix in {".py", ".js", ".html", ".css"} and "__pycache__" not in path.parts
    ]
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "revision": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "files": {str(p.relative_to(repo)): sha256(p) for p in sorted(paths) if p.is_file()},
    }
