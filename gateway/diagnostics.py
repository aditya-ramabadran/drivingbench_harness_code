"""Read-only laptop diagnostics. Network and device probes are explicit."""

import glob
import json
import platform
import shutil
import subprocess
from pathlib import Path

import httpx
from drivingbench.gateway.install import RUNTIME, installation_status
from drivingbench.gateway.provenance import source_state
from drivingbench.gateway.setup import Config, ssh_args
from drivingbench.gateway.traces import transcript_roots
from drivingbench.harness import gemini, openrouter
from drivingbench.shared.contracts import PROTOCOL

KEYS = {
    openrouter.KEYCHAIN_ITEM: "Runner chat models and decision drivers via OpenRouter",
    gemini.KEYCHAIN_ITEM: "Runner chat models via the Google Gemini API",
}


def doctor(config: Config, repo: Path, *, network=False, device=False, runtime=RUNTIME) -> dict:
    checks = []

    def add(name, status, detail, action=""):
        checks.append({"name": name, "status": status, "detail": detail, "action": action})

    add(
        "platform",
        "ok" if platform.system() == "Darwin" else "warning",
        platform.system(),
        "The supported installer uses macOS Keychain and launchd.",
    )
    for name in ("uv", "git", "ssh", "rsync"):
        path = shutil.which(name)
        add(name, "ok" if path else "error", path or "missing", f"Install {name}.")
    current = source_state(repo)
    receipt = runtime / "install.json"
    try:
        installed = json.loads(receipt.read_text())
        changed = sorted(
            k
            for k in current["files"] | installed["files"]
            if current["files"].get(k) != installed["files"].get(k)
        )
        add(
            "runtime",
            "warning" if changed else "ok",
            {"installed_revision": installed.get("revision"), "changed_files": changed},
            "Re-run drivingbench install, then restart the registered chat apps.",
        )
    except (OSError, ValueError, KeyError, TypeError):
        add(
            "runtime",
            "warning",
            "No valid install receipt; running code is unverified.",
            "Run drivingbench install --client <your-client>.",
        )
    add("chat_apps", "ok", installation_status(paths=config.client_paths))
    for key, purpose in KEYS.items():
        found = platform.system() == "Darwin" and openrouter.keychain_has_key(key)
        add(
            key,
            "ok" if found else "warning",
            {"present": found, "used_for": purpose},
            f"If needed, add Keychain item {key}; never paste its value into a chat.",
        )
    add(
        "transcripts",
        "ok",
        {
            label: {"glob": pattern, "files_found": len(glob.glob(str(Path(pattern).expanduser())))}
            for label, pattern in transcript_roots(config).items()
        },
        "Add nonstandard app homes to extra_transcripts in the laptop config.",
    )
    key = Path(config.ssh_key or "~/.ssh/id_ed25519").expanduser()
    add(
        "ssh_identity",
        "ok" if key.is_file() else "warning",
        str(key),
        "Generate your own key; authorize its public key on the comma and refresh SSH keys.",
    )
    if not config.dataset:
        add(
            "hugging_face",
            "ok",
            "No dataset configured; labeled sessions stay local.",
            "To publish them, drivingbench install --dataset <user>/<repo>.",
        )
    elif network:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            account = api.whoami().get("name")
            info = api.repo_info(config.dataset, repo_type="dataset")
            add(
                "hugging_face",
                "ok",
                {"account": account, "dataset": config.dataset, "revision": info.sha},
            )
        except Exception as exc:
            add(
                "hugging_face",
                "error",
                type(exc).__name__,
                f"Run hf auth login with write access to {config.dataset}. Check the network.",
            )
    else:
        add("hugging_face", "unchecked", "Use --network to check login and dataset access.")
    if device:
        args = ["ConnectTimeout=5" if a == "ConnectTimeout=30" else a for a in ssh_args(config)]
        try:
            result = subprocess.run(args + ["true"], capture_output=True, timeout=10)
            add(
                "comma_ssh",
                "ok" if result.returncode == 0 else "error",
                {"host": config.ssh_host, "exit_code": result.returncode},
                "Same Wi-Fi, check device IP, authorize your public key, ssh comma true once.",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            add("comma_ssh", "error", type(exc).__name__, "Check Wi-Fi, IP and SSH authorization.")
        try:
            response = httpx.get(config.producer_url + "/status", timeout=3, trust_env=False)
            response.raise_for_status()
            status = response.json()
            add(
                "producer",
                "ok" if status.get("protocol") == PROTOCOL else "error",
                {
                    k: status.get(k)
                    for k in ("release", "protocol", "state", "reason", "recording_error")
                },
                "Bring online in Setup. Device deployment is a separate supervised car task.",
            )
        except (httpx.HTTPError, ValueError) as exc:
            add("producer", "warning", type(exc).__name__, "Use Bring online, then re-run doctor.")
    else:
        add("comma", "unchecked", "Use --device with the comma nearby; no device was contacted.")
    return {
        "source": {k: v for k, v in current.items() if k != "files"},
        "checks": checks,
        "errors": sum(c["status"] == "error" for c in checks),
    }
