"""
Shot boundary detection using PySceneDetect.

Detects cuts and transitions in video to produce i:/o: boundaries
for all T:SH entries in the generated PLF.
"""

from .schema import ShotBoundary

try:
    from scenedetect import SceneManager, open_video, AdaptiveDetector
    HAS_SCENEDETECT = True
except ImportError:
    HAS_SCENEDETECT = False


def detect_shots(video_path: str, min_duration: float = 0.5) -> list[ShotBoundary]:
    if not HAS_SCENEDETECT:
        raise ImportError(
            "PySceneDetect is required. Install with:\n"
            "  pip install scenedetect[opencv]"
        )

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    shots = []
    for i, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        duration = end_sec - start_sec

        if duration < min_duration and i > 0 and shots:
            shots[-1].o = end_sec
            continue

        shots.append(ShotBoundary(
            i=round(start_sec, 2),
            o=round(end_sec, 2),
            confidence=1.0,
        ))

    return shots


def detect_shots_content_aware(
    video_path: str,
    threshold: float = 27.0,
    min_duration: float = 1.0,
) -> list[ShotBoundary]:
    if not HAS_SCENEDETECT:
        raise ImportError(
            "PySceneDetect is required. Install with:\n"
            "  pip install scenedetect[opencv]"
        )

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        AdaptiveDetector(adaptive_threshold=threshold)
    )

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    shots = []
    for start, end in scene_list:
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()

        if end_sec - start_sec < min_duration:
            continue

        shots.append(ShotBoundary(
            i=round(start_sec, 2),
            o=round(end_sec, 2),
            confidence=1.0,
        ))

    return shots
