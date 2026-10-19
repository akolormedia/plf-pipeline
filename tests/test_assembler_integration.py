"""Integration test: assembler output must parse and validate with plf-python.

This is the contract between plf-pipeline (producer) and plf-python
(consumer). If it breaks, the generated .raw.plf files are unusable.
"""

import pytest

from pipeline.assembler import assemble
from pipeline.schema import (
    ShotBoundary, VisionOutput, Transcript, Word,
    AudioClassification, AudioEvent, UserContext,
)

plf = pytest.importorskip("plf", reason="plf library not installed")


def _make_context():
    return UserContext(
        description="A man walks on the beach at sunset and speaks.",
        characters=[{"name": "Carlos", "desc": "man middle-aged tired"}],
        locations=[{"name": "ext beach sunset", "desc": "empty shore golden light"}],
        language="es",
        genre="drama",
    )


def _make_inputs():
    shots = [
        ShotBoundary(i=0.0, o=4.12),
        ShotBoundary(i=4.12, o=12.00),
        ShotBoundary(i=12.00, o=30.00),
    ]
    vision = [
        VisionOutput(
            shot_index=0, shot_type="ELS", cam="sta", mood="mel",
            act="man-walks-toward-shoreline", arc="approach",
            location="ext-beach-sunset",
            elements=[{"type": "person", "desc": "man silhouette far"}],
        ),
        VisionOutput(
            shot_index=1, shot_type="MS", cam="push", mood="mel",
            act="man-stops-and-looks", arc="hesitation",
            location="ext-beach-sunset",
        ),
    ]
    transcript = Transcript(
        words=[
            Word(text="No", start=5.20, end=5.40, speaker_id="A1"),
            Word(text="puedo", start=5.40, end=5.70, speaker_id="A1"),
            Word(text="más", start=5.70, end=6.10, speaker_id="A1"),
        ],
        language="es",
    )
    audio = AudioClassification(
        ambient_events=[AudioEvent(
            start=0.0, end=30.0, source="ocean-waves-medium",
            diegetic="D", level=3, mood="mel",
        )],
    )
    return shots, vision, transcript, audio


def test_assembler_output_is_valid_plf():
    shots, vision, transcript, audio = _make_inputs()
    text = assemble(
        shots=shots, vision=vision, transcript=transcript, audio=audio,
        context=_make_context(), video_filename="test.mp4",
    )

    parsed = plf.parse(text)
    assert parsed.src == "bruto"
    assert parsed.dur == 30.0
    assert parsed.cast[0].id == "A1"
    assert parsed.cast[0].nombre == "Carlos"

    result = plf.validate(parsed)
    assert result.is_valid, f"assembler output invalid: {result.errors}"


def test_assembler_output_cross_track_valid():
    shots, vision, transcript, audio = _make_inputs()
    text = assemble(
        shots=shots, vision=vision, transcript=transcript, audio=audio,
        context=_make_context(), video_filename="test.mp4",
    )

    parsed = plf.parse(text)
    result = plf.validate_cross_track(parsed)
    assert result.is_valid, f"cross-track errors: {result.errors}"


def test_assembler_empty_inputs():
    """Pipeline with no shots or transcript must still emit a valid header."""
    text = assemble(
        shots=[], vision=[], transcript=Transcript(), audio=AudioClassification(),
        context=UserContext(description="Empty test"), video_filename="x.mp4",
    )
    assert text.startswith("PLF1.0")
    assert "src:bruto" in text


def test_assembler_no_cast_uses_unidentified_figure():
    """With an empty CAST, T:ACT/T:EL subjects must fall back to X1.

    Regression: the assembler used to emit subj:A1 unconditionally, which
    failed plf.validate() with 'references non-existent CAST member' when
    the user context had no characters (found by the Step 3 e2e run).
    """
    shots, vision, transcript, audio = _make_inputs()
    context = UserContext(
        description="No characters provided.",
        characters=[],  # empty cast
        locations=[{"name": "ext field", "desc": "open field"}],
    )
    text = assemble(
        shots=shots, vision=vision, transcript=Transcript(), audio=audio,
        context=context, video_filename="test.mp4",
    )

    parsed = plf.parse(text)
    result = plf.validate(parsed)
    assert result.is_valid, f"validation errors with empty cast: {result.errors}"

    acts = [e for e in parsed.timeline if e.track == "ACT"]
    assert acts, "expected T:ACT entries"
    assert all(e.fields["subj"] == "X1" for e in acts)

    cast_ids = {c.id for c in parsed.cast}
    assert "X1" in cast_ids
    assert "A1" not in cast_ids


def test_assembler_words_stay_inside_dialogue():
    shots, vision, transcript, audio = _make_inputs()
    text = assemble(
        shots=shots, vision=vision, transcript=transcript, audio=audio,
        context=_make_context(), video_filename="test.mp4",
    )
    parsed = plf.parse(text)

    dlg = [e for e in parsed.timeline if e.track == "DLG"]
    words = [e for e in parsed.timeline if e.track == "W"]
    assert len(dlg) == 1
    assert len(words) == 3

    parent = dlg[0]
    for w in words:
        assert w.i >= parent.i and w.o <= parent.o
        assert w.fields["spk"] == parent.fields["spk"]
