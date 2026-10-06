"""Evidence video clip extraction for event verification."""

from pathlib import Path
from typing import Optional
import cv2
from server.app.utils.paths import paths
from server.app.utils.logger import logger
from server.app.utils.video_utils import get_video_info


class ClipGenerator:
    """Cuts and exports video evidence clips corresponding to extracted events."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        padding_before_sec: float = 1.5,
        padding_after_sec: float = 1.5,
    ):
        self.output_dir = Path(output_dir) if output_dir else paths.clips_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.padding_before_sec = padding_before_sec
        self.padding_after_sec = padding_after_sec

    def cut_evidence_clip(
        self,
        video_path: Path | str,
        start_sec: float,
        end_sec: float,
        clip_name: str,
        video_duration_sec: Optional[float] = None,
    ) -> Optional[str]:
        """Cut and save an evidence clip for an event.

        Args:
            video_path: Path to raw source video.
            start_sec: Event start time in seconds.
            end_sec: Event end time in seconds.
            clip_name: Unique identifier for the output clip file (e.g., event_id).
            video_duration_sec: Total duration of video if known.

        Returns:
            Relative or absolute path of generated evidence clip, or None on failure.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Cannot cut clip: video not found at {video_path}")
            return None

        # Determine boundaries with padding
        clip_start = max(0.0, start_sec - self.padding_before_sec)
        clip_end = end_sec + self.padding_after_sec
        if video_duration_sec:
            clip_end = min(clip_end, video_duration_sec)

        output_filename = f"{clip_name}.mp4"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / output_filename

        # If clip already exists and is non-empty, return it
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path.resolve())

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video {video_path} for clip extraction")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(round(clip_start * fps))
        end_frame = int(round(clip_end * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Standard H264 or MP4V codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )

        curr_frame = start_frame
        frames_written = 0

        while curr_frame <= end_frame:
            success, frame = cap.read()
            if not success or frame is None:
                break
            writer.write(frame)
            frames_written += 1
            curr_frame += 1

        writer.release()
        cap.release()

        if frames_written == 0:
            logger.warning(f"No frames written for evidence clip {output_path}")
            if output_path.exists():
                output_path.unlink()
            return None

        logger.info(
            f"Generated evidence clip {output_filename} ({clip_start:.1f}s - {clip_end:.1f}s, {frames_written} frames)"
        )
        return str(output_path.resolve())

