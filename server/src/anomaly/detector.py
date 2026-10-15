"""Anomaly inference using the project's saved AnomalyCLIP temporal ensembles."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import codecs
import os

import cv2
import numpy as np

from server.src.anomaly.schemas import ClipAnomalyScore, VideoAnomalySummary
from server.src.tracking.schemas import TrackingResult
from server.src.utils.logger import logger
from server.src.config.loader import config_loader

CATEGORIES = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing",
    "Vandalism",
]
ANOMALY_CATEGORIES = CATEGORIES[1:]
NUM_SEGMENTS = 64


def _segment_model(torch, params: dict[str, Any], task: str, input_dim: int):
    """Recreate the temporal CNN defined in notebooks/ucf-anomalyclip-binary-and-13class.ipynb."""
    nn = torch.nn
    hidden = int(params["hidden"])
    dropout = float(params["dropout"])
    in_dropout = float(params["in_dropout"])
    layers = int(params["n_layers"])

    class SegmentNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = max(1, int(round(float(params.get("topk_frac", 0.1)) * NUM_SEGMENTS)))
            self.inp = nn.Sequential(
                nn.LayerNorm(input_dim), nn.Dropout(in_dropout), nn.Linear(input_dim, hidden),
                nn.GELU(), nn.Dropout(dropout),
            )
            blocks = []
            for _ in range(layers):
                blocks.extend([nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Dropout(dropout)])
            self.enc = nn.Sequential(*blocks)
            self.norm = nn.LayerNorm(hidden)
            if task == "binary":
                self.head = nn.Linear(hidden, 1)
            else:
                self.att_v = nn.Linear(hidden, hidden // 2)
                self.att_u = nn.Linear(hidden, hidden // 2)
                self.att_w = nn.Linear(hidden // 2, 1)
                self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, len(ANOMALY_CATEGORIES)))

        def forward(self, values):
            encoded = self.inp(values)
            encoded = encoded + self.enc(encoded.transpose(1, 2)).transpose(1, 2)
            encoded = self.norm(encoded)
            if task == "binary":
                segment_logits = self.head(encoded).squeeze(-1)
                video_logit = segment_logits.topk(self.k, dim=1).values.mean(1)
                return video_logit, segment_logits
            attention = self.att_w(
                torch.tanh(self.att_v(encoded)) * torch.sigmoid(self.att_u(encoded))
            ).squeeze(-1)
            weights = torch.softmax(attention, dim=1)
            return self.cls((weights.unsqueeze(-1) * encoded).sum(1)), weights

    return SegmentNet()


class AnomalyDetector:
    """Run the saved binary and 13-class temporal ensembles over a surveillance video."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        scaler_path: Optional[Path | str] = None,
        clip_duration_sec: float = 4.0,
        clip_stride_sec: float = 2.0,
        anomaly_threshold: float = 0.50,
        feature_extractor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        num_frames: int = 256,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.clip_duration_sec = clip_duration_sec
        self.clip_stride_sec = clip_stride_sec
        self.anomaly_threshold = anomaly_threshold
        self.feature_extractor = feature_extractor
        self.num_frames = max(int(num_frames), NUM_SEGMENTS)
        self.model_name = "AnomalyCLIP temporal ensemble (CLIP ViT-B/16 features)"
        self.torch = None
        self.device = None
        self.model_banks: dict[str, list[dict[str, Any]]] = {"binary": [], "multi": []}
        self.clip_model = None
        self.clip_preprocess = None
        self.model_load_error: Optional[str] = None
        self._load_models()

    def _load_models(self) -> None:
        """Load the tensor-only checkpoint without permitting arbitrary pickle globals."""
        if not self.model_path or not self.model_path.is_file():
            self.model_load_error = f"Anomaly checkpoint is missing: {self.model_path}"
            logger.error(self.model_load_error)
            return
        try:
            import torch
            import numpy as np

            numpy_core = getattr(np, "_core", np.core)
            safe_globals = [
                numpy_core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                type(np.dtype("float32")),
                codecs.encode,
            ]
            with torch.serialization.safe_globals(safe_globals):
                checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)

            if not isinstance(checkpoint, dict) or not {"binary", "multi"}.issubset(checkpoint):
                raise ValueError("Expected binary and multi model ensembles in the checkpoint")
            runtime_device = str(os.environ.get("SURVEILLANCE_DEVICE", config_loader.get("runtime.device", "auto"))).lower()
            cuda_enabled = bool(torch.cuda.is_available() and config_loader.get("runtime.use_cuda", True))
            if runtime_device == "cpu" or not cuda_enabled:
                self.device = torch.device("cpu")
            else:
                gpu_id = int(config_loader.get("runtime.gpu_id", 0))
                if runtime_device == "cuda" and not cuda_enabled:
                    raise RuntimeError("CUDA was requested but PyTorch cannot access CUDA")
                self.device = torch.device(f"cuda:{gpu_id}")
            self.torch = torch

            for task in ("binary", "multi"):
                entries = checkpoint[task]
                if not isinstance(entries, list) or not entries:
                    raise ValueError(f"The checkpoint's {task} ensemble is empty")
                for entry in entries:
                    params = entry["params"]
                    if params.get("arch") != "conv":
                        raise ValueError(f"Unsupported {task} model architecture: {params.get('arch')}")
                    center = np.asarray(entry["center"], dtype=np.float32).reshape(1, 1, -1)
                    scale = np.asarray(entry["scale"], dtype=np.float32).reshape(1, 1, -1)
                    if center.shape != scale.shape or center.shape[-1] != 512:
                        raise ValueError(f"Unexpected {task} feature normalization dimensions")
                    model = _segment_model(torch, params, task, center.shape[-1])
                    model.load_state_dict(entry["state"], strict=True)
                    model.to(self.device).eval()
                    self.model_banks[task].append({
                        "model": model,
                        "center": torch.as_tensor(center, device=self.device),
                        "scale": torch.as_tensor(np.maximum(scale, 1e-6), device=self.device),
                    })
            logger.info(
                "Loaded AnomalyCLIP checkpoint from %s (%d binary + %d 13-class models)",
                self.model_path, len(self.model_banks["binary"]), len(self.model_banks["multi"]),
            )
        except Exception as exc:
            self.model_banks = {"binary": [], "multi": []}
            self.model_load_error = f"Could not load anomaly checkpoint {self.model_path}: {exc}"
            logger.exception(self.model_load_error)

    @property
    def model_available(self) -> bool:
        return bool(self.model_banks["binary"] and self.model_banks["multi"])

    def _ensure_clip_encoder(self) -> None:
        if self.clip_model is not None:
            return
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai", device=self.device
        )
        self.clip_model = model.eval()
        self.clip_preprocess = preprocess
        logger.info("Loaded OpenAI CLIP ViT-B/16 feature extractor on %s", self.device)

    def _video_frame_indices(self, capture: cv2.VideoCapture) -> np.ndarray:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
            frame_count = max(1, int(round(fps * self.num_frames)))
        return np.unique(np.linspace(0, max(frame_count - 1, 0), min(self.num_frames, frame_count), dtype=np.int64))

    def _extract_features(self, video_path: Path) -> np.ndarray:
        """Sample the full video, encode RGB frames as 512-D CLIP vectors, and return [T,512]."""
        if self.feature_extractor is None:
            self._ensure_clip_encoder()
        from PIL import Image

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for anomaly extraction: {video_path}")
        indices = self._video_frame_indices(capture)
        all_features: list[np.ndarray] = []
        images = []
        batch_size = int(config_loader.get("anomaly.batch_size", 8))

        def encode_batch(batch):
            if not batch:
                return
            if self.feature_extractor is not None:
                frame_array = np.stack(batch)
                encoded = np.asarray(self.feature_extractor(frame_array), dtype=np.float32)
            else:
                tensors = self.torch.stack(batch).to(self.device)
                with self.torch.inference_mode():
                    if self.device.type == "cuda":
                        with self.torch.autocast("cuda", dtype=self.torch.float16):
                            encoded = self.clip_model.encode_image(tensors)
                    else:
                        encoded = self.clip_model.encode_image(tensors)
                encoded = encoded.float()
                encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                encoded = encoded.cpu().numpy()
            if encoded.ndim != 2 or encoded.shape[1] != 512:
                raise ValueError(f"CLIP feature extractor must return [frames, 512], got {encoded.shape}")
            all_features.extend(encoded)
            batch.clear()

        try:
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.feature_extractor is not None:
                    images.append(rgb)
                else:
                    images.append(self.clip_preprocess(Image.fromarray(rgb)))
                if len(images) >= batch_size:
                    encode_batch(images)
            encode_batch(images)
        finally:
            capture.release()
        if not all_features:
            raise RuntimeError(f"No frames available for anomaly extraction: {video_path}")
        return np.asarray(all_features, dtype=np.float32)

    @staticmethod
    def _pool_features(features: np.ndarray) -> np.ndarray:
        """Mirror the notebook's L2-normalized, equal-duration 64-segment pooling."""
        features = np.asarray(features, dtype=np.float32)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-6)
        pooled = np.empty((NUM_SEGMENTS, features.shape[1]), dtype=np.float32)
        if len(features) >= NUM_SEGMENTS:
            boundaries = np.linspace(0, len(features), NUM_SEGMENTS + 1).astype(np.int64)
            for index in range(NUM_SEGMENTS):
                start = int(boundaries[index])
                end = max(int(boundaries[index + 1]), start + 1)
                pooled[index] = features[start:end].mean(axis=0)
        else:
            for index in range(NUM_SEGMENTS):
                pooled[index] = features[min(index * len(features) // NUM_SEGMENTS, len(features) - 1)]
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-6)
        return pooled

    def _predict_video(self, features: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        torch = self.torch
        inputs = torch.as_tensor(self._pool_features(features)[None], device=self.device)
        binary_probs, segment_probs, type_probs = [], [], []
        with torch.inference_mode():
            for entry in self.model_banks["binary"]:
                normalized = (inputs - entry["center"]) / entry["scale"]
                video_logits, segment_logits = entry["model"](normalized)
                binary_probs.append(torch.sigmoid(video_logits)[0].item())
                segment_probs.append(torch.sigmoid(segment_logits)[0].cpu().numpy())
            for entry in self.model_banks["multi"]:
                normalized = (inputs - entry["center"]) / entry["scale"]
                logits, _ = entry["model"](normalized)
                type_probs.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
        return (
            float(np.mean(binary_probs)),
            np.mean(np.stack(segment_probs), axis=0),
            np.mean(np.stack(type_probs), axis=0),
        )

    def detect_anomalies(
        self,
        video_path: Path | str,
        video_id: str,
        video_duration_sec: float,
        tracking_result: Optional[TrackingResult] = None,
    ) -> VideoAnomalySummary:
        """Score the full video, then map the model's 64 temporal scores back to timestamps."""
        del tracking_result  # The trained model consumes visual embeddings, not track features.
        duration = max(float(video_duration_sec), 0.0)
        if not self.model_available:
            reason = self.model_load_error or "The anomaly classifier checkpoint contains no usable models."
            raise RuntimeError(reason)
        try:
            features = self._extract_features(Path(video_path))
            video_probability, segment_probabilities, type_probabilities = self._predict_video(features)
        except Exception as exc:
            logger.exception("Anomaly inference failed for %s", video_id)
            raise RuntimeError(
                "Anomaly classification failed. Ensure open_clip_torch and its pretrained "
                f"ViT-B/16 weights are available. Details: {exc}"
            ) from exc

        top_type_index = int(np.argmax(type_probabilities))
        top_type = ANOMALY_CATEGORIES[top_type_index]
        top_type_probability = float(type_probabilities[top_type_index])
        segment_duration = duration / NUM_SEGMENTS if duration > 0 else 0.0
        segments: List[ClipAnomalyScore] = []
        for index, probability in enumerate(segment_probabilities):
            is_anomaly = float(probability) >= self.anomaly_threshold
            start = index * segment_duration
            end = min((index + 1) * segment_duration, duration)
            segments.append(ClipAnomalyScore(
                clip_id=f"{video_id}_s{index + 1:03d}",
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                category=top_type if is_anomaly else "Normal",
                confidence=round(float(probability) if is_anomaly else 1.0 - float(probability), 4),
                is_anomaly=is_anomaly,
                metadata={
                    "model_available": True,
                    "model_name": self.model_name,
                    "anomaly_probability": round(float(probability), 4),
                    "video_anomaly_probability": round(video_probability, 4),
                    "type_confidence": round(top_type_probability, 4),
                    "segment_index": index,
                    "num_segments": NUM_SEGMENTS,
                },
            ))

        is_anomalous = video_probability >= self.anomaly_threshold
        summary = VideoAnomalySummary(
            video_id=video_id,
            overall_category=top_type if is_anomalous else "Normal",
            overall_confidence=round(video_probability if is_anomalous else 1.0 - video_probability, 4),
            is_anomalous=is_anomalous,
            model_available=True,
            model_name=self.model_name,
            anomaly_segments=segments,
        )
        logger.info(
            "Anomaly detection for %s: category=%s (prob=%.2f), anomalous_segments=%d/%d",
            video_id, summary.overall_category, video_probability,
            sum(segment.is_anomaly for segment in segments), len(segments),
        )
        return summary
