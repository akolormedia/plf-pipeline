"""Tests for the pipeline orchestrator with all heavy stages mocked.

No scenedetect / whisper / ollama / panns needed: every stage function
imported into pipeline.pipeline is monkeypatched.
"""

from pipeline.pipeline import run_pipeline
from pipeline.schema import (
    UserContext, ShotBoundary, Transcript, Word, AudioClassification,
)


def _fake_shots():
    return [
        ShotBoundary(i=0.0, o=4.0),
        ShotBoundary(i=4.0, o=10.5),
    ]


def _fake_transcript():
    return Transcript(
        words=[
            Word(text="Hola", start=5.0, end=5.3),
            Word(text="mundo", start=5.3, end=5.8),
        ],
        language="es",
    )


def _patch_stages(monkeypatch):
    import pipeline.pipeline as mod

    monkeypatch.setattr(mod, "detect_shots", lambda video_path: _fake_shots())
    monkeypatch.setattr(mod, "sample_frames", lambda video_path, shots: [])
    monkeypatch.setattr(mod, "analyze_shots", lambda frames, context: [])
    monkeypatch.setattr(
        mod, "extract_audio", lambda video_path: "/tmp/fake.wav"
    )
    monkeypatch.setattr(
        mod, "transcribe", lambda path, language="es": _fake_transcript()
    )
    monkeypatch.setattr(
        mod, "classify_audio", lambda path: AudioClassification()
    )


def test_run_pipeline_success(monkeypatch, tmp_path):
    _patch_stages(monkeypatch)

    ctx = UserContext(description="Test video. A man walks.", language="es")
    out = tmp_path / "test.raw.plf"
    result = run_pipeline("fake.mp4", ctx, output_path=str(out))

    assert result.errors == []
    assert result.success
    assert result.raw_plf_text.startswith("PLF1.0")
    assert "T:SH" in result.raw_plf_text
    assert result.duration == 10.5
    assert out.exists()
    assert out.read_text(encoding="utf-8") == result.raw_plf_text


def test_run_pipeline_scene_error(monkeypatch):
    _patch_stages(monkeypatch)
    import pipeline.pipeline as mod

    def boom(video_path):
        raise RuntimeError("no video")

    monkeypatch.setattr(mod, "detect_shots", boom)

    result = run_pipeline("fake.mp4", UserContext())
    assert not result.success
    assert any("Scene detection" in e or "scene" in e.lower() for e in result.errors)


def test_run_pipeline_video_track_error(monkeypatch):
    _patch_stages(monkeypatch)
    import pipeline.pipeline as mod

    def boom(frames, context):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(mod, "analyze_shots", boom)

    result = run_pipeline("fake.mp4", UserContext())
    assert any("Video track" in e for e in result.errors)
    assert result.raw_plf_text == ""


def test_run_pipeline_audio_track_error(monkeypatch):
    _patch_stages(monkeypatch)
    import pipeline.pipeline as mod

    def boom(video_path, language="es"):
        raise RuntimeError("whisper down")

    monkeypatch.setattr(mod, "transcribe", boom)

    result = run_pipeline("fake.mp4", UserContext())
    assert any("Audio track" in e for e in result.errors)
