import copy
import json

import httpx
import pytest
from drivingbench.harness.gemini import Gemini, is_model, model_id, service_tier


def test_direct_model_ids_cannot_change_the_endpoint_or_claim_ultrafast():
    assert is_model("gemini/gemini-5-preview-10-30")
    assert model_id("gemini/gemini-4-argon") == "gemini-4-argon"
    for invalid in (
        "google/gemini-4-argon",
        "gemini/../models",
        "gemini/https://elsewhere",
        "gemini/custom",
    ):
        assert not is_model(invalid)
        with pytest.raises(ValueError):
            model_id(invalid)
    assert service_tier("normal") == "default"
    assert service_tier("fast") == "priority"
    with pytest.raises(ValueError, match="not ultrafast"):
        service_tier("ultrafast")


async def test_direct_chat_keeps_signatures_images_usage_and_actual_tier():
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "observe", "arguments": "{}"},
        "extra_content": {"google": {"thought_signature": "opaque-signature"}},
    }
    body = {
        "model": "gemini/gemini-4-argon",
        "messages": [
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}
                ],
            },
        ],
        "tools": [],
        "reasoning": {"effort": "medium"},
        "service_tier": "fast",
    }
    original = copy.deepcopy(body)

    def handle(request):
        assert (
            str(request.url)
            == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )
        assert request.headers["Authorization"] == "Bearer test-key"
        sent = json.loads(request.content)
        assert sent["messages"] == original["messages"]
        assert sent["model"] == "gemini-4-argon"
        assert sent["reasoning_effort"] == "medium" and "reasoning" not in sent
        assert sent["service_tier"] == "priority"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 7},
            },
            headers={"x-gemini-service-tier": "standard"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        response = await Gemini("test-key", client).chat(body)
    assert body == original
    assert response["provider"] == "Google Gemini API"
    assert response["service_tier"] == "standard"
    assert response["requested_service_tier"] == "priority"
    assert response["usage"] == {"prompt_tokens": 15, "completion_tokens": 7}


async def test_google_model_discovery_paginates_and_keeps_only_chat_models():
    calls = []

    def handle(request):
        calls.append(request)
        if not request.url.params.get("pageToken"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-4-argon",
                            "supportedGenerationMethods": ["generateContent"],
                        }
                    ],
                    "nextPageToken": "page2",
                },
            )
        assert request.url.params["pageToken"] == "page2"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-embedding-2",
                        "supportedGenerationMethods": ["embedContent"],
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        assert await Gemini("test-key", client).models() == ["gemini/gemini-4-argon"]
    assert len(calls) == 2


async def test_google_errors_distinguish_quota_and_do_not_expose_the_key():
    def handle(request):
        return httpx.Response(429, json={"error": {"message": "Quota exceeded for test-key"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(
            RuntimeError, match="Google Gemini API HTTP 429: Quota exceeded"
        ) as error:
            await Gemini("test-key", client).models()
    assert "test-key" not in str(error.value)


async def test_google_accepts_error_arrays_from_the_compatibility_endpoint():
    def handle(request):
        return httpx.Response(
            400, json=[{"error": {"message": "Missing or invalid Authorization header."}}]
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(
            RuntimeError,
            match="Google Gemini API HTTP 400: Missing or invalid Authorization header",
        ):
            await Gemini("test-key", client).chat(
                {"model": "gemini/gemini-3.8-flash", "messages": []}
            )


async def test_probe_only_round_trips_synthetic_observations_and_never_executes_motion():
    calls = []
    signature = {"google": {"thought_signature": "opaque"}}

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        assert {t["function"]["name"] for t in body["tools"]} == {
            "observe",
            "set_motion",
            "stop_now",
        }
        if len(calls) == 1:
            assert body["tool_choice"]["function"]["name"] == "observe"
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "observe", "arguments": "{}"},
                        "extra_content": signature,
                    }
                ],
            }
        else:
            assert body["messages"][1]["tool_calls"][0]["extra_content"] == signature
            assert body["messages"][2]["role"] == "tool"
            assert body["messages"][3]["content"][1]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
            assert body["tool_choice"] == "none"
            message = {"role": "assistant", "content": "Red"}
        return httpx.Response(
            200, json={"model": "gemini-3.8-flash", "choices": [{"message": message}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await Gemini("test-key", client).probe("gemini/gemini-3.8-flash", "normal")
    assert result["checks"] == ["tool_call", "tool_result", "image"]
    assert result["thought_signature_returned"]
    assert result["service_tier"] is None  # Google did not report it
    assert len(calls) == 2
