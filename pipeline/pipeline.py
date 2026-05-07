"""
Pipeline orchestrator — coordinates all analysis stages.

Runs video/audio analysis in parallel where possible and
assembles the final .raw.plf output.
"""

import concurrent.futures
from pathlib import Path

from .schema import UserContext, PipelineResult, ShotBoundary
from .scene_detector import detect_shots
from .frame_sampler import sample_frames
from .vision import analyze_shots
from .transcriber import transcribe, extract_audio
from .diarizer import diarize, merge_with_transcript
from .audio_classifier import classify_audio
from .assembler import assemble


def run_pipeline(
    video_path: str,
    context: UserContext,
    output_path: str | None = None,
    parallel: bool = True,
) -> PipelineResult:
    result = PipelineResult()

    shots = detect_shots(video_path)
    result.shots = shots

    if parallel and len(shots) > 0:
        result = _run_parallel(video_path, shots, context, result)
    else:
        result = _run_sequential(video_path, shots, context, result)

    if not result.errors:
        result.raw_plf_text = assemble(
            shots=result.shots,
            vision=result.vision,
            transcript=result.transcript,
            audio=result.audio,
            context=context,
            video_filename=Path(video_path).name,
        )

    if output_path and result.raw_plf_text:
        Path(output_path).write_text(result.raw_plf_text, encoding="utf-8")

    return result


def _run_parallel(
    video_path: str,
    shots: list[ShotBoundary],
    context: UserContext,
    result: PipelineResult,
) -> PipelineResult:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        video_future = executor.submit(_process_video_track, video_path, shots, context)
        audio_future = executor.submit(_process_audio_track, video_path, context)

        try:
            vision_outputs = video_future.result(timeout=600)
            result.vision = vision_outputs
        except Exception as e:
            result.errors.append(f"Video track error: {e}")

        try:
            transcript, audio_class = audio_future.result(timeout=600)
            result.transcript = transcript
            result.audio = audio_class
        except Exception as e:
            result.errors.append(f"Audio track error: {e}")

    return result


def _run_sequential(
    video_path: str,
    shots: list[ShotBoundary],
    context: UserContext,
    result: PipelineResult,
) -> PipelineResult:
    try:
        result.vision = _process_video_track(video_path, shots, context)
    except Exception as e:
        result.errors.append(f"Video track error: {e}")

    try:
        result.transcript, result.audio = _process_audio_track(video_path, context)
    except Exception as e:
        result.errors.append(f"Audio track error: {e}")

    return result


def _process_video_track(
    video_path: str,
    shots: list[ShotBoundary],
    context: UserContext,
) -> list:
    frames = sample_frames(video_path, shots)
    vision_outputs = analyze_shots(frames, context)
    return vision_outputs


def _process_audio_track(
    video_path: str,
    context: UserContext,
) -> tuple:
    audio_path = extract_audio(video_path)
    transcript = transcribe(audio_path, language=context.language)
    audio_class = classify_audio(audio_path)
    return transcript, audio_class


def run_pipeline_minimal(
    video_path: str,
    context: UserContext | None = None,
) -> PipelineResult:
    if context is None:
        context = UserContext(description="Video analysis")

    result = PipelineResult()

    try:
        shots = detect_shots(video_path)
        result.shots = shots
    except Exception as e:
        result.errors.append(f"Scene detection error: {e}")
        return result

    try:
        result.raw_plf_text = assemble(
            shots=result.shots,
            vision=[],
            transcript=result.transcript,
            audio=result.audio,
            context=context,
            video_filename=Path(video_path).name,
        )
    except Exception as e:
        result.errors.append(f"Assembly error: {e}")

    return result


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.pipeline <video_path> [context.json]")
        sys.exit(1)

    video = sys.argv[1]
    ctx = UserContext()

    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            ctx_data = json.load(f)
            ctx = UserContext(**ctx_data)

    output = str(Path(video).with_suffix(".raw.plf"))
    result = run_pipeline(video, ctx, output_path=output)

    print(f"Pipeline complete: {output}")
    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Shots: {len(result.shots)}")
    print(f"  Errors: {len(result.errors)}")
    if result.errors:
        for err in result.errors:
            print(f"  - {err}")
