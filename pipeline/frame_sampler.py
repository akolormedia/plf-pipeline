"""
Frame sampling strategy for video shots.

Extracts representative frames from each shot at strategic positions
for vision model analysis.
"""

import subprocess
import tempfile
from pathlib import Path

from .schema import ShotBoundary, ShotFrame

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def sample_frames(
    video_path: str,
    shots: list[ShotBoundary],
    output_dir: str | None = None,
) -> list[ShotFrame]:
    if not HAS_CV2:
        raise ImportError(
            "OpenCV is required. Install with:\n"
            "  pip install opencv-python"
        )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    frames = []
    for i, shot in enumerate(shots):
        duration = shot.o - shot.i
        positions = _pick_positions(duration)

        frame_paths = []
        for pos in positions:
            time_sec = shot.i + pos
            path = _extract_frame(video_path, time_sec, i, pos, output_dir)
            if path:
                frame_paths.append(path)

        frames.append(ShotFrame(
            shot_index=i,
            i=shot.i,
            o=shot.o,
            frame_paths=frame_paths,
        ))

    return frames


def _pick_positions(duration: float) -> list[float]:
    if duration < 3:
        return [0.0, duration * 0.9]
    elif duration < 10:
        return [0.0, duration * 0.5, duration * 0.9]
    else:
        return [0.0, duration * 0.25, duration * 0.5, duration * 0.75, duration * 0.9]


def _extract_frame(
    video_path: str,
    time_sec: float,
    shot_index: int,
    position: float,
    output_dir: str | None,
) -> str | None:
    if output_dir:
        out_path = Path(output_dir) / f"shot_{shot_index:03d}_{position:.1f}.jpg"
    else:
        out_path = Path(tempfile.gettempdir()) / f"plf_shot_{shot_index:03d}_{position:.1f}.jpg"

    cmd = [
        "ffmpeg", "-y", "-ss", str(time_sec),
        "-i", video_path,
        "-vframes", "1", "-q:v", "2",
        str(out_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=10)
        if out_path.exists():
            return str(out_path)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None
