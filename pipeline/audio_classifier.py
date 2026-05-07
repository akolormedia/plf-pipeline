"""
Audio classification for ambient sound and music detection.

Classifies background audio into T:AMB and T:MUS entries
with source descriptions, diegetic status, and mood.
"""

from .schema import AudioEvent, AudioClassification


DEFAULT_AMBIENT_SOURCES = [
    "silence",
    "room-tone",
    "wind-light",
    "wind-strong",
    "rain-light",
    "rain-heavy",
    "traffic-distant",
    "traffic-close",
    "crowd-murmur",
    "nature-birds",
    "nature-insects",
    "ocean-waves",
    "city-ambient",
    "interior-hum",
    "machinery",
]

DEFAULT_MUSIC_SOURCES = [
    "orchestral-score",
    "piano-solo",
    "guitar-acoustic",
    "electronic-ambient",
    "strings-section",
    "percussion",
    "vocal",
    "band",
    "radio-music",
    "source-music",
]

MOOD_PATTERNS = {
    "minor-key slow tempo": "mel",
    "major-key fast tempo": "ale",
    "dissonant irregular": "ten",
    "warm intimate slow": "rom",
    "minor-key sparse": "sad",
    "low-frequency build": "mie",
    "aggressive distorted": "ang",
    "neutral steady": "neu",
}


def classify_audio(
    audio_path: str,
) -> AudioClassification:
    return AudioClassification(
        music_events=[],
        ambient_events=[],
    )


def classify_with_panns(
    audio_path: str,
) -> AudioClassification:
    try:
        import panns_inference
    except ImportError:
        raise ImportError(
            "panns-inference is required. Install with:\n"
            "  pip install panns-inference"
        )

    result = AudioClassification()

    return result


def detect_diegetic(
    event_type: str,
    has_visual_source: bool = False,
) -> str:
    if event_type == "music" and not has_visual_source:
        return "ND"
    return "D"


def describe_source(
    label: str,
    labels_list: list[str],
    fallback: str = "unknown",
) -> str:
    label_lower = label.lower().replace(" ", "-")
    for src in labels_list:
        if label_lower in src or src in label_lower:
            return src
    return f"{fallback}-{label_lower}"


def infer_mood(characteristics: str) -> str:
    for pattern, mood in MOOD_PATTERNS.items():
        if pattern in characteristics.lower():
            return mood
    return "neu"
