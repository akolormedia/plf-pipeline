"""
Audio transcription using Whisper.

Extracts dialogue with word-level timestamps for T:DLG and T:W entries.

The default large-v3 model can segfault on some CPU/Mac setups (a native crash
in the forward pass, not catchable in-process). To keep a dead model from
taking the whole pipeline down, transcription is attempted in a *subprocess*
per model size, degrading to the next smaller model if one dies.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from .schema import Word, Transcript

# Attempted in order; the first that runs wins. large-v3 is preferred but
# crashes on some CPUs, so smaller fallbacks follow.
_MODEL_LADDER = ("large-v3", "medium", "small", "base", "tiny")


def transcribe(
    video_path: str,
    language: str = "es",
    model_size: str = "large-v3",
) -> Transcript:
    try:
        import whisper  # noqa: F401
    except ImportError:
        raise ImportError(
            "openai-whisper is required. Install with:\n"
            "  pip install openai-whisper"
        )

    # Build the ladder starting at the requested model, then smaller ones.
    start = _MODEL_LADDER.index(model_size) if model_size in _MODEL_LADDER else 0
    ladder = _MODEL_LADDER[start:]

    for size in ladder:
        transcript = _try_transcribe_in_subprocess(video_path, language, size)
        if transcript is not None:
            return transcript

    raise RuntimeError(
        "Every Whisper model size crashed (segfault). The CPU/torch build may "
        "be incompatible with Whisper's forward pass. Tried: "
        + ", ".join(ladder)
    )


def _try_transcribe_in_subprocess(
    video_path: str, language: str, model_size: str
) -> Transcript | None:
    """Run one model in a subprocess; return None if it crashes.

    A native segfault kills the child process (not this one), so we can safely
    degrade to a smaller model instead of losing the whole pipeline.
    """
    import json

    # numba's default threading layer (omp) segfaults in whisper's DTW on this
    # platform; workqueue runs correctly. Set it before the child imports numba.
    import os

    env = dict(os.environ, NUMBA_THREADING_LAYER="workqueue")

    # The child prints a single JSON line on success; anything else (crash,
    # traceback, signal) means this model size failed.
    proc = subprocess.run(
        [
            sys.executable, "-c",
            (
                "import json,sys;"
                "from pipeline.transcriber import _transcribe_local;"
                "t=_transcribe_local(sys.argv[1],sys.argv[2],sys.argv[3]);"
                "print(json.dumps({'words':[w.__dict__ for w in t.words],"
                "'language':t.language}))"
            ),
            video_path, language, model_size,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if proc.returncode != 0:
        # Crash (segfault -> negative returncode in some shells, else 139) or a
        # clean error; either way this model size is unusable here.
        print(
            f"[transcriber] model '{model_size}' failed "
            f"(exit {proc.returncode}); trying a smaller one...",
            file=sys.stderr,
        )
        return None

    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None

    return Transcript(
        words=[Word(**w) for w in data["words"]],
        language=data.get("language", language),
    )


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
