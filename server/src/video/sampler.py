"""Video frame sampler for temporal surveillance processing."""

from pathlib import Path
from typing import Generator, Tuple, List, Optional
import cv2
import numpy as np
from server.src.utils.logger import logger
from server.src.utils.video_utils import get_video_info


class VideoSampler:
    """Samples video frames uniformly at a target frames-per-second rate."""

    def __init__(self, sample_fps: float = 4.0, max_dimension: Optional[int] = 1280):
        self.sample_fps = float(sample_fps)
        self.max_dimension = max_dimension

    def sample_frames(
        self, video_path: Path | str
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield (frame_index, timestamp_seconds, frame_rgb) at sample_fps.

        Args:
            video_path: Path to video file.

        Yields:
            Tuple of (frame_idx, timestamp_sec, rgb_frame_array).
        """
        path_str = str(video_path)
        cap = cv2.VideoCapture(path_str)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0.0

        step_sec = 1.0 / self.sample_fps
        current_sec = 0.0

        logger.info(
            f"Sampling {video_path} (native {video_fps:.1f} fps, {duration:.1f}s) at {self.sample_fps} fps"
        )

        while current_sec < duration:
            frame_idx = int(round(current_sec * video_fps))
            if frame_idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success or frame is None:
                current_sec += step_sec
                continue

            # Resize if exceeding max_dimension
            if self.max_dimension is not None:
                h, w = frame.shape[:2]
                if max(h, w) > self.max_dimension:
                    scale = self.max_dimension / float(max(h, w))
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_idx, round(current_sec, 3), frame_rgb

            current_sec += step_sec

        cap.release()

    def sample_all_frames(
        self, video_path: Path | str
    ) -> List[Tuple[int, float, np.ndarray]]:
        """Collect all sampled frames into a list."""
        return list(self.sample_frames(video_path))
