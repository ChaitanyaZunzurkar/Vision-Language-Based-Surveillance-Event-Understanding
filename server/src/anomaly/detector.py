"""Video anomaly detector implementing UCF-Crime 14-category classification."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import pickle
import numpy as np
import cv2
from server.src.anomaly.schemas import ClipAnomalyScore, VideoAnomalySummary
from server.src.tracking.schemas import TrackingResult
from server.src.utils.logger import logger
from server.src.config.loader import config_loader

CATEGORIES = [
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]
CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}
IDX2CAT = {i: c for c, i in CAT2IDX.items()}


class AnomalyDetector:
    """Detects unusual or suspicious surveillance segments and classifies into UCF-Crime categories."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        scaler_path: Optional[Path | str] = None,
        clip_duration_sec: float = 4.0,
        clip_stride_sec: float = 2.0,
        anomaly_threshold: float = 0.50,
        feature_extractor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.clip_duration_sec = clip_duration_sec
        self.clip_stride_sec = clip_stride_sec
        self.anomaly_threshold = anomaly_threshold
        self.feature_extractor = feature_extractor

        self.classifier = None
        self.scaler = None
        self.feature_extractor_name = "AnomalyCLIP"
        self._load_models()

    def _load_models(self) -> None:
        """Attempt to load trained classifier and scaler if available."""
        if self.model_path and self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.classifier = pickle.load(f)
                logger.info(f"Loaded anomaly classifier from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load anomaly classifier: {e}")

        if self.scaler_path and self.scaler_path.exists():
            try:
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded anomaly scaler from {self.scaler_path}")
            except Exception as e:
                logger.warning(f"Failed to load anomaly scaler: {e}")

    @property
    def model_available(self) -> bool:
        return self.classifier is not None and self.feature_extractor is not None

    def _sample_frames(self, video_path: Path, start_sec: float, end_sec: float) -> np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for anomaly extraction: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
        frame_count = max(1, int(round(max(end_sec - start_sec, 0.1) * fps)))
        indices = np.linspace(
            int(start_sec * fps),
            max(int(end_sec * fps) - 1, int(start_sec * fps)),
            num=min(frame_count, 16),
            dtype=int,
        )
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        if not frames:
            raise RuntimeError(f"No frames available for anomaly clip {start_sec}-{end_sec}")
        return np.stack(frames)

    def _predict_clip(
        self, video_path: Path, start_sec: float, end_sec: float
    ) -> tuple[str, float, bool, Dict[str, Any]]:
        if self.classifier is None or self.feature_extractor is None:
            return "Unassessed", 0.0, False, {
                "model_available": False,
                "feature_extractor": self.feature_extractor_name,
            }
        frames = self._sample_frames(video_path, start_sec, end_sec)
        features = np.asarray(self.feature_extractor(frames))
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        probabilities = self.classifier.predict_proba(features)[0]
        classes = list(getattr(self.classifier, "classes_", range(len(probabilities))))
        best = int(np.argmax(probabilities))
        raw_category = classes[best]
        category = str(raw_category)
        if isinstance(raw_category, (int, np.integer)):
            category = IDX2CAT.get(int(raw_category), str(raw_category))
        confidence = float(probabilities[best])
        is_anomaly = category != "Normal" and confidence >= self.anomaly_threshold
        return category, confidence, is_anomaly, {
            "model_available": True,
            "feature_extractor": self.feature_extractor_name,
            "feature_dimension": int(features.shape[-1]),
        }

    def detect_anomalies(
        self,
        video_path: Path | str,
        video_id: str,
        video_duration_sec: float,
        tracking_result: Optional[TrackingResult] = None,
    ) -> VideoAnomalySummary:
        """Evaluate video in temporal clips and produce anomaly scores and category predictions."""
        clips: List[ClipAnomalyScore] = []
        c_start = 0.0
        clip_idx = 1

        duration = max(video_duration_sec, 4.0)

        # Map tracks to temporal intervals
        tracks = tracking_result.tracks if tracking_result else []

        while c_start < duration:
            c_end = min(c_start + self.clip_duration_sec, duration)
            clip_id = f"{video_id}_c{clip_idx:03d}"

            # Calculate activity density in this clip window
            active_tracks = [
                t for t in tracks if not (t.end_sec < c_start or t.start_sec > c_end)
            ]

            category, confidence, is_anomaly, metadata = self._predict_clip(
                Path(video_path), c_start, c_end
            )

            clips.append(
                ClipAnomalyScore(
                    clip_id=clip_id,
                    start_sec=round(c_start, 2),
                    end_sec=round(c_end, 2),
                    category=category,
                    confidence=round(confidence, 3),
                    is_anomaly=is_anomaly,
                    metadata=metadata,
                )
            )

            c_start += self.clip_stride_sec
            clip_idx += 1

        # Calculate overall video-level anomaly
        anom_clips = [c for c in clips if c.is_anomaly]
        if anom_clips:
            # Pick highest confidence anomaly
            top_anom = max(anom_clips, key=lambda c: c.confidence)
            overall_cat = top_anom.category
            overall_conf = top_anom.confidence
            is_anom = True
        else:
            overall_cat = "Unassessed"
            overall_conf = 0.0
            is_anom = False

        logger.info(
            f"Anomaly detection for {video_id}: category={overall_cat} (conf={overall_conf:.2f}), anomalous_clips={len(anom_clips)}/{len(clips)}"
        )

        return VideoAnomalySummary(
            video_id=video_id,
            overall_category=overall_cat,
            overall_confidence=round(overall_conf, 3),
            is_anomalous=is_anom,
            model_available=self.model_available,
            model_name=self.feature_extractor_name,
            anomaly_segments=clips,
        )
