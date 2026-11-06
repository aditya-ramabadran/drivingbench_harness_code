import json
from types import SimpleNamespace

import pytest
from drivingbench.gateway import diagnostics
from drivingbench.gateway.setup import Config


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """doctor with every probe recorded: Keychain lookups, source receipts, no device."""
    calls = []
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        diagnostics.subprocess,
        "run",
        lambda args, **kw: calls.append(args) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        diagnostics,
        "source_state",
        lambda repo: {"files": {"gateway/app.py": "new"}, "revision": "new"},
    )
    monkeypatch.setattr(diagnostics, "installation_status", lambda **kw: {"clients": {}})
    monkeypatch.setattr(diagnostics, "transcript_roots", lambda config: {})
    monkeypatch.setattr(
        diagnostics.httpx, "get", lambda *a, **kw: pytest.fail("No implicit device network request")
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "install.json").write_text(
        json.dumps({"revision": "old", "files": {"gateway/app.py": "old"}})
    )

    def doctor(config, **kw):
        calls.clear()
        report = diagnostics.doctor(config, tmp_path, runtime=runtime, **kw)
        return {row["name"]: row for row in report["checks"]}, calls

    return doctor


def test_doctor_does_not_contact_device_or_expose_key_values(offline):
    checks, calls = offline(Config(dataset="me/traces"))
    assert calls == [
        ["security", "find-generic-password", "-s", "drivingbench-openrouter"],
        ["security", "find-generic-password", "-s", "drivingbench-gemini"],
    ]
    assert checks["runtime"]["status"] == "warning"
    assert checks["runtime"]["detail"]["changed_files"] == ["gateway/app.py"]
    assert checks["comma"]["status"] == "unchecked"
    assert checks["hugging_face"]["status"] == "unchecked"


def test_hugging_face_is_checked_only_for_a_configured_dataset(offline, monkeypatch):
    import huggingface_hub

    asked = []

    class Api:
        def whoami(self):
            return {"name": "me"}

        def repo_info(self, repo, repo_type):
            asked.append((repo, repo_type))
            if repo == "me/missing":
                raise RuntimeError("404")
            return SimpleNamespace(sha="abc")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    checks, _ = offline(Config(), network=True)
    assert checks["hugging_face"]["status"] == "ok" and asked == []  # uploads off: nothing to ask
    checks, _ = offline(Config(dataset="me/traces"), network=True)
    assert checks["hugging_face"]["detail"] == {
        "account": "me",
        "dataset": "me/traces",
        "revision": "abc",
    }
    checks, _ = offline(Config(dataset="me/missing"), network=True)
    assert checks["hugging_face"]["status"] == "error"
    assert "me/missing" in checks["hugging_face"]["action"]
