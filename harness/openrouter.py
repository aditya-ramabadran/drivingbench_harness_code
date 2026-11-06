"""OpenRouter for the runner: decisions for the driver, chat for the supervisor.

The key comes only from the Keychain item `drivingbench-openrouter`, never from the environment,
so no other OpenRouter key on the laptop can be picked up by accident.
"""

import subprocess

import httpx

BASE_URL = "https://openrouter.ai/api"
KEYCHAIN_ITEM = "drivingbench-openrouter"


def keychain_key(item: str = KEYCHAIN_ITEM) -> str | None:
    try:
        found = subprocess.run(
            ["security", "find-generic-password", "-s", item, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return found.stdout.strip() or None


def keychain_has_key(item: str = KEYCHAIN_ITEM) -> bool:
    """Whether the Keychain item exists, without reading its value."""
    try:
        found = subprocess.run(
            ["security", "find-generic-password", "-s", item],
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return found.returncode == 0


class OpenRouter:
    def __init__(self, key: str, client: httpx.AsyncClient | None = None):
        self.key = key
        self.client = client or httpx.AsyncClient(trust_env=False)

    async def post(self, path: str, body: dict, timeout: float) -> dict:
        response = await self.client.post(
            BASE_URL + path,
            json=body,
            timeout=timeout,
            headers={"Authorization": f"Bearer {self.key}", "X-Title": "DrivingBench"},
        )
        response.raise_for_status()
        return response.json()

    async def decide(self, model: str, state: list, questions: dict) -> dict:
        body = {"model": model, "state": state, "questions": questions}
        return (await self.post("/alpha/decisions", body, timeout=5.0))["answers"]

    async def chat(self, body: dict) -> dict:
        return await self.post("/v1/chat/completions", body, timeout=180.0)
