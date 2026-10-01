"""Surveillance entity detector using YOLO or robust OpenCV background subtraction fallback."""

from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from src.detection.schemas import BoundingBox, Detection, DetectionResult
from src.video.sampler import VideoSampler
from src.utils.logger import logger
from src.config.loader import config_loader


class SurveillanceDetector:
    """Detects surveillance entities (persons, vehicles, bags) in video frames."""

    def __init__(
        self,
        weights_path: Optional[Path | str] = None,
        conf_threshold: float = 0.20,
        device: str = "auto",
    ):
        self.conf_threshold = conf_threshold
        self.device = device
        self.weights_path = Path(weights_path) if weights_path else None
        self.model = None
        self.use_yolo = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempt to load Ultralytics YOLO model, fallback to CV2 vision detector if unavailable."""
        if self.weights_path and self.weights_path.exists():
            try:
                from ultralytics import YOLO

                logger.info(f"Loading YOLO model from {self.weights_path}")
                self.model = YOLO(str(self.weights_path))
                self.use_yolo = True
                logger.info("YOLO detector initialized successfully.")
                return
            except Exception as e:
                logger.warning(f"Could not load YOLO from {self.weights_path}: {e}")

        logger.info(
            "Using OpenCV motion & contour heuristic detector (fallback for CPU / lightweight environment)."
        )
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=32, detectShadows=True
        )

    def detect_frame(
        self, frame_rgb: np.ndarray, frame_idx: int, timestamp_sec: float
    ) -> List[Detection]:
        """Detect objects in a single frame.

        Args:
            frame_rgb: RGB image numpy array.
            frame_idx: Index of current frame.
            timestamp_sec: Video timestamp in seconds.

        Returns:
            List of Detection objects.
        """
        detections: List[Detection] = []

        if self.use_yolo and self.model is not None:
            results = self.model.predict(
                frame_rgb,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device if self.device != "auto" else None,
            )
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = r.names.get(cls_id, f"class_{cls_id}")

                    detections.append(
                        Detection(
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp_sec,
                            class_id=cls_id,
                            class_name=cls_name.capitalize(),
                            confidence=round(conf, 3),
                            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        )
                    )
            return detections

        # Heuristic fallback using MOG2 + contours for surveillance entities
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        fg_mask = self.bg_subtractor.apply(frame_bgr)
        # Filter shadow pixels (gray = 127)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        h, w = frame_rgb.shape[:2]
        min_area = (h * w) * 0.0015  # min 0.15% of frame area

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bh / float(bw)

            # Heuristic category based on aspect ratio
            if aspect_ratio > 1.4:
                class_name = "Person"
                class_id = 0
            elif 0.5 <= aspect_ratio <= 1.4 and area > min_area * 3:
                class_name = "Car"
                class_id = 1
            else:
                class_name = "Backpack"
                class_id = 6

            # Confidence based on contour compactness
            conf = min(0.95, round(0.50 + min(0.45, area / (h * w * 0.1)), 2))
            if conf >= self.conf_threshold:
                detections.append(
                    Detection(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=BoundingBox(
                            x1=float(x),
                            y1=float(y),
                            x2=float(x + bw),
                            y2=float(y + bh),
                        ),
                    )
                )

        return detections

    def detect_video(
        self,
        video_path: Path | str,
        video_id: str,
        sample_fps: Optional[float] = None,
    ) -> DetectionResult:
        """Run detection across all sampled frames of a video."""
        fps = sample_fps or config_loader.get("video.sample_fps", 4.0)
        sampler = VideoSampler(sample_fps=fps)

        all_detections: List[Detection] = []
        for frame_idx, timestamp_sec, frame_rgb in sampler.sample_frames(video_path):
            dets = self.detect_frame(frame_rgb, frame_idx, timestamp_sec)
            all_detections.extend(dets)

        logger.info(
            f"Detected {len(all_detections)} instances across {video_id} at {fps} fps"
        )
        return DetectionResult(
            video_id=video_id,
            total_detections=len(all_detections),
            detections=all_detections,
        )
