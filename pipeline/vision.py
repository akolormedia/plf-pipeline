"""
Vision analysis using Qwen2-VL via Ollama.

Analyzes video frames to determine shot type, camera movement,
mood, action, and elements in frame.
"""

import base64
from .schema import ShotFrame, VisionOutput, UserContext

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


VISION_MODEL = "qwen2-vl:7b"

VISION_PROMPT = """Analyze this video frame and describe:
1. Shot type: one of ELS, LS, MLS, MS, MCU, CU, ECU, OTS, POV, AER
2. Camera movement: one of sta, pan, tilt, push, pull, track, crane, hh, dly
3. Mood: one of mel, ale, ten, rom, sad, mie, ang, neu
4. Action: what is happening (hyphenated, lowercase, English)
5. Location: describe the setting (hyphenated slug format)
6. Elements: list objects and people visible with attributes

Respond ONLY with a JSON object:
{
  "shot": "MS",
  "cam": "sta",
  "mood": "mel",
  "act": "person-walks-toward-door",
  "loc": "int-room-dim",
  "elements": [
    {"type": "person", "desc": "woman standing tall"},
    {"type": "object", "desc": "wooden-table dusty"}
  ],
  "description": "A woman stands in a dimly lit room near a wooden table."
}
"""


def analyze_shots(
    frames: list[ShotFrame],
    context: UserContext | None = None,
    model: str = VISION_MODEL,
) -> list[VisionOutput]:
    if not HAS_OLLAMA:
        raise ImportError(
            "Ollama Python client is required. Install with:\n"
            "  pip install ollama"
        )

    results = []
    for shot_frame in frames:
        if not shot_frame.frame_paths:
            results.append(VisionOutput(shot_index=shot_frame.shot_index))
            continue

        primary_frame = shot_frame.frame_paths[0]
        try:
            response = ollama.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": VISION_PROMPT,
                    "images": [_encode_image(primary_frame)],
                }],
            )
            output = _parse_vision_response(response["message"]["content"])
            output.shot_index = shot_frame.shot_index
            results.append(output)
        except Exception as e:
            results.append(VisionOutput(
                shot_index=shot_frame.shot_index,
                description=f"ERROR: {e}",
            ))

    return results


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_vision_response(text: str) -> VisionOutput:
    import json
    import re

    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return VisionOutput(description=text.strip())

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return VisionOutput(description=text.strip())

    return VisionOutput(
        shot_index=0,
        shot_type=data.get("shot", ""),
        cam=data.get("cam", ""),
        mood=data.get("mood", ""),
        act=data.get("act", ""),
        location=data.get("loc", ""),
        elements=data.get("elements", []),
        description=data.get("description", ""),
    )


def analyze_single_frame(
    frame_path: str,
    model: str = VISION_MODEL,
) -> VisionOutput:
    frames = [ShotFrame(
        shot_index=0,
        i=0.0,
        o=0.0,
        frame_paths=[frame_path],
    )]
    results = analyze_shots(frames, model=model)
    return results[0] if results else VisionOutput()
