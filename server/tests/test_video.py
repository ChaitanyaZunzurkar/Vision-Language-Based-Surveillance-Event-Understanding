"""Tests for video utilities and frame sampling."""

import tempfile
from pathlib import Path
import cv2
import numpy as np
from server.src.utils.video_utils import get_video_info
from server.src.detection.sampler import VideoSampler
from server.src.evidence.clip_generator import ClipGenerator


def create_synthetic_video(file_path: Path, num_frames: int = 30, fps: float = 10.0):
    """Generate a test MP4 video file with simulated motion."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw moving rectangle simulating a person walking
        x = int(20 + i * 5)
        cv2.rectangle(frame, (x, 50), (x + 30, 150), (255, 255, 255), -1)
        out.write(frame)

    out.release()


def test_video_metadata_and_sampling():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video = Path(tmpdir) / "test_surveillance.mp4"
        create_synthetic_video(tmp_video, num_frames=30, fps=10.0)

        info = get_video_info(tmp_video)
        assert info["frame_count"] == 30
        assert info["width"] == 320
        assert info["height"] == 240
        assert abs(info["duration_sec"] - 3.0) < 0.1

        sampler = VideoSampler(sample_fps=4.0)
        sampled = sampler.sample_all_frames(tmp_video)
        assert len(sampled) >= 10
        assert sampled[0][2].shape == (240, 320, 3)


def test_clip_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video = Path(tmpdir) / "source.mp4"
        create_synthetic_video(tmp_video, num_frames=40, fps=10.0)

        clips_dir = Path(tmpdir) / "clips"
        generator = ClipGenerator(output_dir=clips_dir, padding_before_sec=0.5, padding_after_sec=0.5)

        clip_path = generator.cut_evidence_clip(
            video_path=tmp_video,
            start_sec=1.0,
            end_sec=2.0,
            clip_name="ev_test_01",
            video_duration_sec=4.0,
        )

        assert clip_path is not None
        assert Path(clip_path).exists()
        clip_info = get_video_info(clip_path)
        assert clip_info["frame_count"] > 0

