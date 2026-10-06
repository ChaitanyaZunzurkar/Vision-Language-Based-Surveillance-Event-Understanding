"""Video utility functions for surveillance processing."""

from pathlib import Path
from typing import Dict, Any, Optional
import cv2
from server.app.utils.logger import logger


def get_video_info(video_path: Path | str) -> Dict[str, Any]:
    """Extract metadata from video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys: duration_sec, fps, frame_count, width, height.
    """
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        fps = 30.0  # safe default

    duration_sec = frame_count / fps if fps > 0 else 0.0

    return {
        "duration_sec": round(duration_sec, 2),
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
    }


def extract_frame_at_sec(video_path: Path | str, timestamp_sec: float) -> Optional[Any]:
    """Extract a single frame (RGB) at given timestamp in seconds."""
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = int(round(timestamp_sec * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        return None

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

