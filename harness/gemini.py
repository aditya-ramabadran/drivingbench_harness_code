"""Google's direct Gemini API, using its documented chat-completions endpoint.

Runner IDs use gemini/<exact Google model ID>, separate from OpenRouter's google/... IDs.
Tool calls (including Google's thought signatures) and image history pass through unchanged.
An AI Studio key identifies the Cloud project; this path does not use Vertex AI or OpenRouter.
"""

import base64
import io
import json
import re
import time

import httpx
from drivingbench.harness.supervisor import assistant_message, sandbox_tools
from PIL import Image

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
KEYCHAIN_ITEM = "drivingbench-gemini"
PREFIX = "gemini/"
DEFAULT_MODELS = ("gemini/gemini-4-argon", "gemini/gemini-3.8-flash")
TIERS = {"normal": "default", "fast": "priority"}


def is_model(model: str) -> bool:
    return bool(re.fullmatch(r"gemini/gemini-[a-zA-Z0-9][a-zA-Z0-9._-]*", model))


def model_id(model: str) -> str:
    if not is_model(model):
        raise ValueError("Use gemini/<exact Google model ID>, without a URL or models/ prefix")
    return model.removeprefix(PREFIX)


def service_tier(tier: str) -> str:
    if tier not in TIERS:
        raise ValueError(
            "Direct Gemini supports normal (standard) or fast (priority), not ultrafast"
        )
    return TIERS[tier]


class Gemini:
    def __init__(self, key: str, client: httpx.AsyncClient | None = None):
        self.key, self.client = key, client

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None

    async def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        if self.client is None:
            self.client = httpx.AsyncClient(trust_env=False)
        headers = (
            {"Authorization": f"Bearer {self.key}"}
            if path.startswith("/openai/")
            else {"x-goog-api-key": self.key}
        )
        response = await self.client.request(method, BASE_URL + path, headers=headers, **kwargs)
        if not response.is_success:
            try:
                payload = response.json()
                errors = payload if isinstance(payload, list) else [payload]
                messages = []
                for error in errors:
                    if isinstance(error, dict):
                        error = error.get("error", error)
                    messages.append(
                        error.get("message", str(error)) if isinstance(error, dict) else str(error)
                    )
                detail = "; ".join(messages)
            except ValueError:
                detail = "Invalid response from Google"
            detail = detail.replace(self.key, "[redacted]")
            raise RuntimeError(f"Google Gemini API HTTP {response.status_code}: {detail[:500]}")
        return response

    async def models(self) -> list[str]:
        """Catalog discovery is not proof of inference entitlement or quota."""
        models, params = [], {"pageSize": 1000}
        while True:
            response = await self.request("GET", "/models", params=params, timeout=30)
            data = response.json()
            for spec in data.get("models", []):
                name = spec["name"].removeprefix("models/")
                if name.startswith("gemini-") and "generateContent" in spec.get(
                    "supportedGenerationMethods", []
                ):
                    models.append(PREFIX + name)
            if not data.get("nextPageToken"):
                return sorted(set(models))
            params["pageToken"] = data["nextPageToken"]

    async def chat(self, body: dict) -> dict:
        request = {k: v for k, v in body.items() if k not in ("reasoning", "service_tier")}
        request["model"] = model_id(body["model"])
        request["reasoning_effort"] = body.get("reasoning", {}).get("effort", "medium")
        tier = service_tier(body.get("service_tier", "normal"))
        request["service_tier"] = tier
        response = await self.request("POST", "/openai/chat/completions", json=request, timeout=180)
        data = response.json()
        if not data.get("choices"):
            raise RuntimeError("Google Gemini API returned no completion choices")
        data["provider"] = "Google Gemini API"
        data["requested_service_tier"] = tier
        # Never report the requested tier as the tier Google actually served.
        data["service_tier"] = response.headers.get("x-gemini-service-tier") or data.get(
            "service_tier"
        )
        return data

    async def probe(self, model: str, tier: str) -> dict:
        """Two real inference calls with synthetic imagery; never contacts a car or executes tools."""
        tools = await sandbox_tools(solo=True)
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), "red").save(buffer, "JPEG")
        messages = [{"role": "user", "content": "Offline API test, no vehicle. Call observe once."}]
        body = {"model": model, "messages": messages, "tools": tools, "service_tier": tier}
        started = time.monotonic()
        first = await self.chat(
            body | {"tool_choice": {"type": "function", "function": {"name": "observe"}}}
        )
        message = first["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls or any(c["function"]["name"] != "observe" for c in calls):
            raise RuntimeError("Google probe did not return the required observe call")
        messages.append(assistant_message(message))
        messages.extend(
            {"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"state": "held"})}
            for c in calls
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Synthetic observe image. Name its color; no tools."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(buffer.getvalue()).decode()
                        },
                    },
                ],
            }
        )
        second = await self.chat(body | {"tool_choice": "none"})
        text = second["choices"][0]["message"].get("content") or ""
        if "red" not in text.lower():
            raise RuntimeError("Google probe did not identify the synthetic red image")
        return {
            "model": model,
            "response_model": second.get("model"),
            "provider": second["provider"],
            "requested_service_tier": second["requested_service_tier"],
            "service_tier": second["service_tier"],
            "latency_s": round(time.monotonic() - started, 2),
            "checks": ["tool_call", "tool_result", "image"],
            "thought_signature_returned": any(
                c.get("extra_content", {}).get("google", {}).get("thought_signature") for c in calls
            ),
            "usage": [first.get("usage"), second.get("usage")],
        }
