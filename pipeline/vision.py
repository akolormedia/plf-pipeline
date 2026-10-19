"""
Vision analysis using Qwen2.5-VL via Ollama.

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


VISION_MODEL = "qwen2.5vl:7b"

VISION_PROMPT = """You are analyzing a single video frame that is attached to this message. Look at the attached image carefully before answering.

Answer these in plain words:
1. What is the shot type? (ELS, LS, MLS, MS, MCU, CU, ECU, OTS, POV, AER)
2. What is the camera movement? (sta, pan, tilt, push, pull, track, crane, hh, dly)
3. What is the mood? (mel, ale, ten, rom, sad, mie, ang, neu)
4. What action is happening? Answer in hyphenated lowercase English.
5. Where is this? Describe the setting as a hyphenated slug.
6. What objects or people are visible? List each with a short attribute description.

Then, after your analysis, output ONLY a JSON object with these keys and no extra text:
{"shot":"","cam":"","mood":"","act":"","loc":"","elements":[{"type":"person|object","desc":""}],"description":""}
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
            # NOTE: use generate(), not chat(). In this ollama client version,
            # chat() silently ignores the `images` field, so the model never
            # sees the frame and parrots the worked example in VISION_PROMPT.
            # generate() does pass images through. Verified 2026-10-19.
            response = ollama.generate(
                model=model,
                prompt=VISION_PROMPT,
                images=[_encode_image(primary_frame)],
                options={"temperature": 0.2},
            )
            output = _parse_vision_response(response["response"])
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
