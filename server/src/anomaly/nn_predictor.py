"""Inference for the notebook's temporal binary and 13-class anomaly networks."""

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from server.src.anomaly.schemas import ClipAnomalyScore, VideoAnomalySummary
from server.src.utils.logger import logger


ANOMALY_CLASSES = [
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
SEGMENT_COUNT = 64


class TemporalAnomalyPredictor:
    """Run the fixed notebook ensemble over sampled CLIP ViT-B/16 video features."""

    def __init__(
        self,
        model_path: Path | str,
        feature_model: str = "openai/clip-vit-base-patch16",
        sample_fps: float = 2.0,
        batch_size: int = 16,
        anomaly_threshold: float = 0.5,
        device: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.feature_model_name = feature_model
        self.sample_fps = sample_fps
        self.batch_size = batch_size
        self.anomaly_threshold = anomaly_threshold
        self.device_name = device
        self._device = None
        self._torch = None
        self._encoder = None
        self._processor = None
        self._ensembles = None
        self._loaded = False

    @property
    def model_available(self) -> bool:
        return self._loaded and self._ensembles is not None and self._encoder is not None

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Anomaly network checkpoint not found: {self.model_path}")
        if self.model_path.stat().st_size < 1024:
            raise FileNotFoundError(
                f"Anomaly checkpoint is not hydrated from Git LFS: {self.model_path}"
            )
        if self.sample_fps <= 0 or self.batch_size <= 0:
            raise ValueError("Anomaly sample_fps and batch_size must be greater than zero")

        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self._torch = torch
        if self.device_name == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.device_name)

        checkpoint = torch.load(
            self.model_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict) or not {
            "binary", "multi"
        }.issubset(checkpoint):
            raise ValueError(
                "Anomaly checkpoint must contain the notebook's 'binary' and 'multi' ensembles"
            )
        self._ensembles = {
            task: [self._build_model(item, task) for item in checkpoint[task]]
            for task in ("binary", "multi")
        }
        if not self._ensembles["binary"] or not self._ensembles["multi"]:
            raise ValueError("Anomaly checkpoint is missing a binary or 13-class ensemble")

        self._processor = CLIPImageProcessor.from_pretrained(self.feature_model_name)
        self._encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.feature_model_name
        ).to(self._device)
        self._encoder.eval()
        feature_dim = self._ensembles["binary"][0]["feature_dim"]
        projected_dim = int(self._encoder.config.projection_dim)
        if projected_dim != feature_dim:
            raise ValueError(
                f"CLIP feature dimension {projected_dim} does not match anomaly checkpoint "
                f"dimension {feature_dim}"
            )
        self._loaded = True
        logger.info(
            "Loaded temporal anomaly checkpoint and %s image encoder on %s",
            self.model_path,
            self._device,
        )

    def _build_model(self, item: Dict[str, Any], task: str) -> Dict[str, Any]:
        torch = self._torch
        params = item["params"]
        state = item["state"]
        feature_dim = int(state["inp.2.weight"].shape[1])
        model = _build_segment_network(
            torch,
            feature_dim,
            params,
            task,
            len(ANOMALY_CLASSES),
        )
        model.load_state_dict(item["state"])
        model.to(self._device)
        model.eval()
        return {
            "model": model,
            "center": np.asarray(item["center"], dtype=np.float32),
            "scale": np.asarray(item["scale"], dtype=np.float32),
            "feature_dim": feature_dim,
        }

    def _extract_video_features(self, video_path: Path, duration_sec: float):
        from PIL import Image

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for anomaly prediction: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or not np.isfinite(fps) or fps <= 0 or total_frames <= 0:
            capture.release()
            raise RuntimeError(f"Video has invalid frame metadata: {video_path}")

        stride = max(1, int(round(fps / self.sample_fps)))
        images = []
        timestamps = []
        feature_batches = []
        frame_indices = range(0, total_frames, stride)
        try:
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    continue
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                timestamps.append(frame_index / fps)
                if len(images) >= self.batch_size:
                    feature_batches.append(self._encode_images(images))
                    images.clear()
            if images:
                feature_batches.append(self._encode_images(images))
        finally:
            capture.release()

        if not feature_batches:
            raise RuntimeError(f"No frames could be sampled for anomaly prediction: {video_path}")
        features = np.concatenate(feature_batches, axis=0)
        sampled_times = np.asarray(timestamps[:len(features)], dtype=np.float32)
        return _pool_features(features, sampled_times, duration_sec)

    def _encode_images(self, images) -> np.ndarray:
        torch = self._torch
        inputs = self._processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            embeddings = self._encoder(
                pixel_values=inputs["pixel_values"].to(self._device)
            ).image_embeds
        return embeddings.float().cpu().numpy()

    def predict_video(
        self,
        video_path: Path | str,
        video_id: str,
        video_duration_sec: float,
    ) -> VideoAnomalySummary:
        self._load()
        pooled = self._extract_video_features(Path(video_path), video_duration_sec)
        normalized = _l2_normalize(pooled)[None, ...]
        binary_probability, segment_probabilities = self._predict_ensemble(
            self._ensembles["binary"], normalized, "binary"
        )
        class_probabilities, _ = self._predict_ensemble(
            self._ensembles["multi"], normalized, "multi"
        )

        anomaly_probability = float(binary_probability[0])
        predicted_class_index = int(np.argmax(class_probabilities[0]))
        predicted_class = (
            ANOMALY_CLASSES[predicted_class_index]
            if anomaly_probability >= self.anomaly_threshold
            else "Normal"
        )
        type_confidence = float(class_probabilities[0][predicted_class_index])
        class_probabilities_by_name = {
            name: float(class_probabilities[0][index])
            for index, name in enumerate(ANOMALY_CLASSES)
        }
        segment_scores = segment_probabilities[0]
        segments = []
        for index, score in enumerate(segment_scores):
            start = video_duration_sec * index / SEGMENT_COUNT
            end = video_duration_sec * (index + 1) / SEGMENT_COUNT
            score = float(score)
            anomalous = (
                anomaly_probability >= self.anomaly_threshold
                and score >= self.anomaly_threshold
            )
            segments.append(
                ClipAnomalyScore(
                    clip_id=f"{video_id}_s{index + 1:03d}",
                    start_sec=round(start, 2),
                    end_sec=round(end, 2),
                    category=predicted_class if anomalous else "Normal",
                    confidence=round(score if anomalous else 1.0 - score, 4),
                    is_anomaly=anomalous,
                    metadata={
                        "anomaly_probability": round(score, 5),
                        "video_anomaly_probability": round(anomaly_probability, 5),
                        "type_probabilities": class_probabilities_by_name,
                        "feature_model": self.feature_model_name,
                    },
                )
            )

        logger.info(
            "Anomaly prediction for %s: %s (video probability %.3f)",
            video_id,
            predicted_class,
            anomaly_probability,
        )
        return VideoAnomalySummary(
            video_id=video_id,
            overall_category=predicted_class,
            overall_confidence=round(
                1.0 - anomaly_probability
                if predicted_class == "Normal"
                else anomaly_probability * type_confidence,
                4,
            ),
            is_anomalous=predicted_class != "Normal",
            model_available=True,
            model_name=f"Notebook temporal NN + {self.feature_model_name}",
            anomaly_segments=segments,
        )

    def _predict_ensemble(self, ensemble, pooled: np.ndarray, task: str):
        torch = self._torch
        video_probabilities = []
        segment_outputs = []
        for member in ensemble:
            member_input = (
                _l2_normalize(pooled) - member["center"]
            ) / np.maximum(member["scale"], 1e-6)
            tensor = torch.as_tensor(
                member_input.astype(np.float32), device=self._device
            )
            with torch.inference_mode():
                video_output, segment_output = member["model"](tensor)
                if task == "binary":
                    video_output = torch.sigmoid(video_output)
                    segment_output = torch.sigmoid(segment_output)
                else:
                    video_output = torch.softmax(video_output, dim=1)
                video_probabilities.append(video_output.cpu().numpy())
                segment_outputs.append(segment_output.cpu().numpy())
        return (
            np.mean(video_probabilities, axis=0),
            np.mean(segment_outputs, axis=0),
        )


def _pool_features(
    features: np.ndarray, timestamps: np.ndarray, duration_sec: float
) -> np.ndarray:
    """L2-normalize sampled frame embeddings and mean-pool 64 temporal segments."""
    if features.ndim != 2 or features.shape[0] != timestamps.shape[0]:
        raise ValueError("CLIP features and frame timestamps have inconsistent shapes")
    features = _l2_normalize(features)
    boundaries = np.linspace(0.0, duration_sec, SEGMENT_COUNT + 1)
    pooled = np.empty((SEGMENT_COUNT, features.shape[1]), dtype=np.float32)
    for index in range(SEGMENT_COUNT):
        if index == SEGMENT_COUNT - 1:
            selected = (
                (timestamps >= boundaries[index])
                & (timestamps <= boundaries[index + 1])
            )
        else:
            selected = (timestamps >= boundaries[index]) & (timestamps < boundaries[index + 1])
        if np.any(selected):
            pooled[index] = features[selected].mean(axis=0)
        else:
            midpoint = (boundaries[index] + boundaries[index + 1]) / 2
            nearest = int(np.argmin(np.abs(timestamps - midpoint)))
            pooled[index] = features[nearest]
    return np.nan_to_num(pooled)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-6)


def _build_segment_network(torch, feature_dim, params, task, class_count):
    import torch.nn as nn

    hidden = int(params["hidden"])
    n_layers = int(params["n_layers"])
    dropout = float(params["dropout"])
    input_dropout = float(params["in_dropout"])
    architecture = params["arch"]
    n_heads = int(params["n_heads"])
    segment_count = SEGMENT_COUNT

    class SegmentNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.task = task
            self.arch = architecture
            self.k = max(1, int(round(float(params.get("topk_frac", 0.1)) * segment_count)))
            self.inp = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(input_dropout),
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if architecture == "transformer":
                self.pos = nn.Parameter(torch.zeros(1, segment_count, hidden))
                nn.init.trunc_normal_(self.pos, std=0.02)
                layer = nn.TransformerEncoderLayer(
                    hidden,
                    n_heads,
                    dim_feedforward=hidden * 2,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                self.enc = nn.TransformerEncoder(
                    layer, n_layers, enable_nested_tensor=False
                )
            elif architecture == "bigru":
                self.enc = nn.GRU(
                    hidden,
                    hidden // 2,
                    num_layers=n_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if n_layers > 1 else 0.0,
                )
            else:
                blocks = []
                for _ in range(n_layers):
                    blocks.extend([
                        nn.Conv1d(hidden, hidden, 3, padding=1),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ])
                self.enc = nn.Sequential(*blocks)
            self.norm = nn.LayerNorm(hidden)
            if task == "binary":
                self.head = nn.Linear(hidden, 1)
            else:
                self.att_v = nn.Linear(hidden, hidden // 2)
                self.att_u = nn.Linear(hidden, hidden // 2)
                self.att_w = nn.Linear(hidden // 2, 1)
                self.cls = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hidden, class_count)
                )

        def forward(self, values):
            hidden_values = self.inp(values)
            if self.arch == "transformer":
                hidden_values = self.enc(hidden_values + self.pos)
            elif self.arch == "bigru":
                hidden_values = self.enc(hidden_values)[0]
            else:
                hidden_values = hidden_values + self.enc(
                    hidden_values.transpose(1, 2)
                ).transpose(1, 2)
            hidden_values = self.norm(hidden_values)
            if self.task == "binary":
                segment_logits = self.head(hidden_values).squeeze(-1)
                video_logit = segment_logits.topk(self.k, dim=1).values.mean(1)
                return video_logit, segment_logits
            attention = self.att_w(
                torch.tanh(self.att_v(hidden_values))
                * torch.sigmoid(self.att_u(hidden_values))
            ).squeeze(-1)
            weights = torch.softmax(attention, dim=1)
            return self.cls((weights.unsqueeze(-1) * hidden_values).sum(1)), weights

    return SegmentNetwork()
