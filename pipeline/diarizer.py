"""
Speaker diarization using pyannote-audio 3.x.

Identifies who speaks when, producing speaker_id labels
that map to CAST entries in the PLF header.

Requires HuggingFace token with access to pyannote models.
"""

from .schema import DiarizationSegment


def diarize(
    audio_path: str,
    hf_token: str | None = None,
    num_speakers: int | None = None,
) -> list[DiarizationSegment]:
    try:
        from pyannote.audio import Pipeline
    except Exception:
        # Catch broadly: a broken transitive dep (e.g. torchcodec ABI mismatch)
        # raises OSError on import, not ImportError.
        raise ImportError(
            "pyannote-audio is required. Install with:\n"
            "  pip install pyannote-audio\n"
            "Also requires huggingface-cli login for model access."
        )

    import os
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN env var "
            "or pass hf_token parameter."
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    if num_speakers:
        diarization = pipeline(audio_path, num_speakers=num_speakers)
    else:
        diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(DiarizationSegment(
            start=round(turn.start, 2),
            end=round(turn.end, 2),
            speaker_id=speaker,
            speaker_label=speaker,
        ))

    return segments


def assign_cast_ids(
    segments: list[DiarizationSegment],
    speaker_map: dict[str, str] | None = None,
) -> list[DiarizationSegment]:
    if speaker_map:
        for seg in segments:
            if seg.speaker_label in speaker_map:
                seg.speaker_id = speaker_map[seg.speaker_label]

    return segments


def merge_with_transcript(
    segments: list[DiarizationSegment],
    words: list,
) -> list:
    for word in words:
        for seg in segments:
            if seg.start <= word.start < seg.end:
                word.speaker_id = seg.speaker_id
                break

    return words
