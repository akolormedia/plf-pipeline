"""
PLF Assembler — merges pipeline outputs into valid PLF text.

Takes vision analysis, transcript, diarization, and audio classification
outputs and assembles them into a complete .raw.plf file via Model A.

For now, uses a template-based approach. Model A (Qwen2.5-3B) integration
comes in Step 7 after fine-tuning.
"""

from .schema import (
    ShotBoundary, VisionOutput, Transcript, AudioClassification,
    AudioEvent, UserContext, PipelineResult,
)


def assemble(
    shots: list[ShotBoundary],
    vision: list[VisionOutput],
    transcript: Transcript,
    audio: AudioClassification,
    context: UserContext,
    video_filename: str = "video.mp4",
) -> str:
    proj = context.description.split(".")[0][:30] if context.description else "Untitled"

    lines = []
    _write_header(lines, proj, shots, context, video_filename)
    _write_scenes(lines, vision, audio, context)
    _write_timeline(lines, shots, vision, transcript, audio, context)

    return "\n".join(lines)


def _write_header(
    lines: list[str],
    proj: str,
    shots: list[ShotBoundary],
    context: UserContext,
    video_filename: str,
):
    proj_slug = proj.replace(" ", "").replace(":", "-")
    duration = shots[-1].o if shots else 0.0

    lines.append(
        f"PLF1.0 proj:{proj_slug}  fps:{context.fps}  "
        f"ar:{context.aspect_ratio}  dur:{duration:.2f}  src:bruto"
    )
    lines.append(f'ctx:"{context.description}"')
    lines.append("")
    lines.append("CAST:")

    for i, char in enumerate(context.characters):
        char_id = f"A{i + 1}"
        nombre = char.get("name", f"Speaker_{i + 1}")
        desc = char.get("desc", "unknown")
        lines.append(f"  {char_id}  nombre:{nombre}  desc:{desc}")

    if not context.characters:
        # No cast provided: register an unidentified figure so T:ACT entries
        # have a valid subj: (X1 = unidentified per the CAST ID rules).
        lines.append("  X1  nombre:unknown  desc:unidentified-figure")

    lines.append("")
    lines.append("SCENES:")

    for i, loc in enumerate(context.locations):
        scene_id = f"SC{i + 1:02d}"
        loc_slug = loc.get("name", f"loc-{i + 1}").lower().replace(" ", "-")
        time_val = loc.get("time", "tarde")
        lines.append(
            f'  {scene_id}  loc:{loc_slug}  time:{time_val}  act:1  '
            f'desc:"{loc.get("desc", "")}"'
        )

    lines.append("")
    lines.append("CLIPS:")
    lines.append(f"  CLP001  file:{video_filename}  tc:00:00:00:00")
    lines.append("")


def _write_scenes(
    lines: list[str],
    vision: list[VisionOutput],
    audio: AudioClassification,
    context: UserContext,
):
    pass


def _write_timeline(
    lines: list[str],
    shots: list[ShotBoundary],
    vision: list[VisionOutput],
    transcript: Transcript,
    audio: AudioClassification,
    context: UserContext,
):
    vision_by_index = {v.shot_index: v for v in vision}
    # Actions and person-elements need a CAST-valid subject: identified
    # characters get A1, otherwise fall back to the unidentified figure X1.
    subj = "A1" if context.characters else "X1"

    for i, shot in enumerate(shots):
        v = vision_by_index.get(i, VisionOutput(shot_index=i))

        shot_id = f"s{i + 1:02d}"
        shot_type = v.shot_type or "MS"
        cam = v.cam or "sta"
        mood = v.mood or "neu"
        loc = v.location or "unknown"

        lines.append(
            f"i:{shot.i:.2f}    o:{shot.o:.2f}    T:SH   id:{shot_id}  "
            f"shot:{shot_type}  loc:{loc}  trans_i:CT  trans_o:CT  "
            f"cam:{cam}  mood:{mood}  raw:CLP001@{shot.i:.2f}-{shot.o:.2f}"
        )

        if v.act:
            lines.append(
                f"i:{shot.i:.2f}    o:{shot.o:.2f}    T:ACT  "
                f"subj:{subj}  act:{v.act}  arc:{v.arc or 'neutral'}"
            )

        for elem in v.elements:
            elem_type = elem.get("type", "object")
            elem_desc = elem.get("desc", "")
            if elem_type == "person":
                lines.append(
                    f"i:{shot.i:.2f}    o:{shot.o:.2f}    T:EL   "
                    f"sh:{shot_id}  subj:{subj}  attr:{elem_desc}"
                )
            else:
                obj_name = elem_desc.replace(" ", "-")
                lines.append(
                    f"i:{shot.i:.2f}    o:{shot.o:.2f}    T:EL   "
                    f"sh:{shot_id}  obj:{obj_name}  attr:{elem_desc}"
                )

    for event in audio.ambient_events:
        lines.append(
            f"i:{event.start:.2f}    o:{event.end:.2f}    T:AMB  "
            f"src:{event.source}  dieg:{event.diegetic}  "
            f"lvl:{event.level}  mood:{event.mood}"
        )

    for event in audio.music_events:
        lines.append(
            f"i:{event.start:.2f}    o:{event.end:.2f}    T:MUS  "
            f"src:{event.source}  dieg:{event.diegetic}  "
            f"lvl:{event.level}  mood:{event.mood}"
        )

    dialogue_groups = _group_words_by_speaker(transcript.words)
    for speaker, words in dialogue_groups.items():
        if len(words) < 2:
            continue

        dlg_start = words[0].start
        dlg_end = words[-1].end
        lines.append(
            f"i:{dlg_start:.2f}    o:{dlg_end:.2f}    T:DLG  "
            f"spk:{speaker}  tone:neutral"
        )

        for w in words:
            lines.append(
                f"i:{w.start:.2f}    o:{w.end:.2f}    T:W    "
                f'spk:{speaker}  "{w.text}"'
            )

    lines.sort(key=lambda l: _extract_i(l))


def _group_words_by_speaker(words):
    groups = {}
    for w in words:
        spk = w.speaker_id or "A1"
        if spk not in groups:
            groups[spk] = []
        groups[spk].append(w)
    return groups


def _extract_i(line: str) -> float:
    try:
        return float(line.split()[0].split(":")[1])
    except (IndexError, ValueError):
        return 0.0
