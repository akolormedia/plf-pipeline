"""
Audio transcription using Whisper large-v3.

Extracts dialogue with word-level timestamps for T:DLG and T:W entries.
"""

import subprocess
import tempfile
from pathlib import Path

from .schema import Word, Transcript


def transcribe(
    video_path: str,
    language: str = "es",
    model_size: str = "large-v3",
) -> Transcript:
    try:
        import whisper
        return _transcribe_local(video_path, language, model_size)
    except ImportError:
        return _transcribe_fallback(video_path, language)


def _transcribe_local(
    video_path: str,
    language: str,
    model_size: str,
) -> Transcript:
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(
        video_path,
        language=language,
        word_timestamps=True,
    )

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append(Word(
                text=w["word"].strip(),
                start=round(w["start"], 2),
                end=round(w["end"], 2),
                confidence=w.get("confidence", 1.0),
            ))

    return Transcript(
        words=words,
        language=result.get("language", language),
    )


def _transcribe_fallback(video_path: str, language: str) -> Transcript:
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper is required. Install with:\n"
            "  pip install openai-whisper"
        )


def extract_audio(video_path: str, output_path: str | None = None) -> str:
    if output_path is None:
        output_path = str(
            Path(tempfile.gettempdir()) /
            f"plf_audio_{Path(video_path).stem}.wav"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_path,
    ]

    subprocess.run(cmd, capture_output=True, timeout=60)
    return output_path


def transcribe_audio_file(
    audio_path: str,
    language: str = "es",
    model_size: str = "large-v3",
) -> Transcript:
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
    )

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append(Word(
                text=w["word"].strip(),
                start=round(w["start"], 2),
                end=round(w["end"], 2),
                confidence=w.get("confidence", 1.0),
            ))

    return Transcript(
        words=words,
        language=result.get("language", language),
    )
