"""Contract tests for the pipeline HTTP server.

These verify the exact contract expected by plf-tauri's pipeline.rs:
  GET  /status  → JSON with component availability
  POST /generate {video_path, context} → plain-text PLF body

run_pipeline is mocked — no video or heavy deps needed.
"""

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient

from pipeline.schema import (
    UserContext, ShotBoundary, PipelineResult,
)
from pipeline import server
from pipeline.server import app


client = TestClient(app)


def _patch_run(monkeypatch, result):
    def fake_run(video_path, context, output_path=None, parallel=True):
        return result

    # server.py does `from .pipeline import run_pipeline`, so patch the
    # name bound in the server module.
    monkeypatch.setattr(server, "run_pipeline", fake_run)


def _make_result(plf_text="PLF1.0 proj:Test  fps:24  ar:16:9  dur:4.00  src:bruto"):
    r = PipelineResult()
    r.raw_plf_text = plf_text
    r.shots = [ShotBoundary(i=0.0, o=4.0)]
    return r


def test_status_reports_ok():
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "components" in data
    assert set(data["components"].keys()) == {
        "scenedetect", "whisper", "ollama", "pyannote", "panns",
    }


def test_generate_returns_plain_text_plf(monkeypatch):
    _patch_run(monkeypatch, _make_result())

    resp = client.post(
        "/generate",
        json={"video_path": "/tmp/fake.mp4", "context": {"language": "es"}},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text.startswith("PLF1.0")


def test_generate_maps_context(monkeypatch):
    captured = {}

    def fake_run(video_path, context, output_path=None, parallel=True):
        captured["context"] = context
        return _make_result()

    monkeypatch.setattr(server, "run_pipeline", fake_run)

    resp = client.post(
        "/generate",
        json={
            "video_path": "/tmp/fake.mp4",
            "context": {
                "description": "A walk on the beach",
                "language": "es",
                "genre": "drama",
                "characters": [{"name": "Carlos", "desc": "man"}],
            },
        },
    )
    assert resp.status_code == 200
    ctx = captured["context"]
    assert isinstance(ctx, UserContext)
    assert ctx.language == "es"
    assert ctx.genre == "drama"
    assert ctx.characters == [{"name": "Carlos", "desc": "man"}]


def test_generate_ignores_unknown_context_keys(monkeypatch):
    _patch_run(monkeypatch, _make_result())

    resp = client.post(
        "/generate",
        json={
            "video_path": "/tmp/fake.mp4",
            "context": {"language": "es", "bogus_key": 123},
        },
    )
    assert resp.status_code == 200


def test_generate_errors_return_500(monkeypatch):
    def fake_run(video_path, context, output_path=None, parallel=True):
        r = PipelineResult()
        r.errors.append("Video track error: ollama down")
        return r

    monkeypatch.setattr(server, "run_pipeline", fake_run)

    resp = client.post(
        "/generate",
        json={"video_path": "/tmp/fake.mp4", "context": {}},
    )
    assert resp.status_code == 500
    assert "ollama down" in resp.text


def test_generate_exception_returns_500(monkeypatch):
    def fake_run(video_path, context, output_path=None, parallel=True):
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "run_pipeline", fake_run)

    resp = client.post(
        "/generate",
        json={"video_path": "/tmp/fake.mp4", "context": {}},
    )
    assert resp.status_code == 500
    assert "PIPELINE ERROR" in resp.text
    assert "boom" in resp.text
