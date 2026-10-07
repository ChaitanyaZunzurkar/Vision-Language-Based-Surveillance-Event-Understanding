"""AnomalyCLIP feature extraction and checkpoint-backed anomaly inference."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from server.src.anomaly.schemas import ClipAnomalyScore, VideoAnomalySummary
from server.src.tracking.schemas import TrackingResult
from server.src.utils.logger import logger


ANOMALY_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
    "Stealing", "Vandalism",
]
CATEGORIES = ["Normal", *ANOMALY_CLASSES]


class _SegNetModule:
    def __init__(self, nn, torch, in_dim, n_seg, hidden, n_layers, task, n_classes,
                 dropout, in_dropout, topk_frac):
        self.torch = torch
        self.task = task
        self.n_seg = n_seg
        self.k = max(1, int(round(topk_frac * n_seg)))
        self.module = nn.Module()
        self.module.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(in_dropout),
                                        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.module.enc = nn.Sequential(*sum(
            ([nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Dropout(dropout)]
             for _ in range(n_layers)), []))
        self.module.norm = nn.LayerNorm(hidden)
        if task == "binary":
            self.module.head = nn.Linear(hidden, 1)
        else:
            self.module.att_v = nn.Linear(hidden, hidden // 2)
            self.module.att_u = nn.Linear(hidden, hidden // 2)
            self.module.att_w = nn.Linear(hidden // 2, 1)
            self.module.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def __getattr__(self, name):
        return getattr(self.module, name)

    def load_state_dict(self, state):
        return self.module.load_state_dict(state, strict=True)

    def eval(self):
        self.module.eval()
        return self

    def to(self, device):
        self.module.to(device)
        return self

    def __call__(self, x):
        h = self.module.inp(x)
        h = h + self.module.enc(h.transpose(1, 2)).transpose(1, 2)
        h = self.module.norm(h)
        if self.task == "binary":
            seg = self.module.head(h).squeeze(-1)
            return seg.topk(self.k, dim=1).values.mean(1), seg
        a = self.module.att_w(self.torch.tanh(self.module.att_v(h)) * self.torch.sigmoid(self.module.att_u(h))).squeeze(-1)
        weights = self.torch.softmax(a, dim=1)
        return self.module.cls((weights.unsqueeze(-1) * h).sum(1)), weights


class AnomalyDetector:
    """Run the exact 64-segment, 512-D preprocessing used by the training notebook."""

    def __init__(self, checkpoint_path: Path | str, feature_model: str,
                 feature_device: str = "auto", clip_duration_sec: float = 4.0,
                 clip_stride_sec: float = 2.0, num_frames: int = 16,
                 num_segments: int = 64, anomaly_threshold: float = 0.5):
        self.checkpoint_path = Path(checkpoint_path)
        self.feature_model_name = feature_model
        self.device_name = feature_device
        self.clip_duration_sec = float(clip_duration_sec)
        self.clip_stride_sec = float(clip_stride_sec)
        self.num_frames = int(num_frames)
        self.num_segments = int(num_segments)
        self.anomaly_threshold = float(anomaly_threshold)
        self.device = None
        self.torch = None
        self.processor = None
        self.feature_model = None
        self.entries: Dict[str, List[Dict[str, Any]]] = {"binary": [], "multi": []}
        self.model_metadata: Dict[str, Any] = {}
        self.load_error: Optional[str] = None
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        try:
            import torch
            self.torch = torch
            self.device = torch.device(
                "cuda" if self.device_name == "auto" and torch.cuda.is_available()
                else self.device_name if self.device_name != "auto" else "cpu"
            )
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Anomaly checkpoint not found: {self.checkpoint_path}")
            bank = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            if not isinstance(bank, dict) or not {"binary", "multi"}.issubset(bank):
                raise ValueError("Anomaly checkpoint must contain binary and multi model banks")
            self.entries = bank
            self.model_metadata = {
                "checkpoint": str(self.checkpoint_path),
                "feature_dim": 512,
                "num_segments": self.num_segments,
                "device": str(self.device),
                "binary_models": len(bank["binary"]),
                "multiclass_models": len(bank["multi"]),
                "class_mapping": {str(i): c for i, c in enumerate(ANOMALY_CLASSES)},
            }
            logger.info("Loaded anomaly checkpoint: %s", self.model_metadata)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("Anomaly model loading failed: %s", self.load_error)

    def _load_feature_model(self) -> None:
        if self.feature_model is not None:
            return
        if self.torch is None:
            raise RuntimeError(self.load_error or "PyTorch is unavailable")
        from transformers import AutoProcessor, CLIPModel
        self.processor = AutoProcessor.from_pretrained(self.feature_model_name)
        self.feature_model = CLIPModel.from_pretrained(self.feature_model_name).to(self.device).eval()

    def _read_frames(self, video_path: Path, start: float, end: float) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        count = max(1, int(round((end - start) * fps)))
        indices = np.linspace(max(0, int(start * fps)), max(0, int(end * fps) - 1),
                              max(self.num_frames, count), dtype=np.int64)
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames read from {video_path} in [{start}, {end}]")
        return frames

    def _features(self, frames: List[np.ndarray]) -> np.ndarray:
        self._load_feature_model()
        inputs = self.processor(images=frames, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            features = self.feature_model.get_image_features(**inputs)
        values = features.detach().cpu().numpy().astype(np.float32)
        if values.shape[-1] != 512:
            raise ValueError(f"AnomalyCLIP feature dimension is {values.shape[-1]}, expected 512")
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)
        return values

    def _segment_features(self, video_path: Path, start: float, end: float) -> np.ndarray:
        frames = self._read_frames(video_path, start, end)
        frame_features = self._features(frames)
        bounds = np.linspace(0, len(frame_features), self.num_segments + 1).astype(np.int64)
        segments = []
        for i in range(self.num_segments):
            left = int(bounds[i])
            right = max(left + 1, int(bounds[i + 1]))
            right = min(right, len(frame_features))
            segments.append(frame_features[min(left, len(frame_features) - 1):right].mean(axis=0))
        return np.asarray(segments, dtype=np.float32)

    def _predict(self, features: np.ndarray) -> tuple[float, str, float, Dict[str, Any]]:
        if self.load_error:
            raise RuntimeError(self.load_error)
        probabilities = []
        class_probabilities = []
        x = features[None, ...]
        for task, entries in self.entries.items():
            for entry in entries:
                params = dict(entry["params"])
                center = np.asarray(entry["center"], dtype=np.float32)
                scale = np.asarray(entry["scale"], dtype=np.float32)
                if center.shape != (512,) or scale.shape != (512,):
                    raise ValueError("Checkpoint center/scale must each have shape [512]")
                transformed = ((features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-6)
                                - center) / np.maximum(scale, 1e-6)).astype(np.float32)
                model = _SegNetModule(__import__("torch.nn", fromlist=["nn"]), self.torch,
                                      512, self.num_segments, int(params["hidden"]),
                                      int(params["n_layers"]), task,
                                      1 if task == "binary" else 13,
                                      float(params.get("dropout", 0.0)),
                                      float(params.get("in_dropout", 0.0)),
                                      float(params.get("topk_frac", 0.1))).to(self.device)
                model.load_state_dict({k: v.to(self.device) for k, v in entry["state"].items()})
                tensor = self.torch.as_tensor(transformed[None, ...], device=self.device)
                with self.torch.no_grad():
                    output, _ = model(tensor)
                if task == "binary":
                    probabilities.append(float(self.torch.sigmoid(output).item()))
                else:
                    class_probabilities.append(self.torch.softmax(output, dim=1).cpu().numpy()[0])
        anomaly_probability = float(np.mean(probabilities))
        if not class_probabilities:
            raise RuntimeError("Checkpoint contains no multiclass models")
        class_probs = np.mean(class_probabilities, axis=0)
        category_index = int(np.argmax(class_probs))
        category = ANOMALY_CLASSES[category_index] if anomaly_probability >= self.anomaly_threshold else "Normal"
        confidence = anomaly_probability if category == "Normal" else float(class_probs[category_index])
        return anomaly_probability, category, confidence, {
            "binary_probability": anomaly_probability,
            "multiclass_probabilities": {name: float(class_probs[i]) for i, name in enumerate(ANOMALY_CLASSES)},
        }

    def detect_anomalies(self, video_path: Path | str, video_id: str,
                         video_duration_sec: float,
                         tracking_result: Optional[TrackingResult] = None) -> VideoAnomalySummary:
        del tracking_result
        if self.load_error:
            raise RuntimeError(self.load_error)
        path = Path(video_path)
        clips: List[ClipAnomalyScore] = []
        start = 0.0
        index = 1
        while start < video_duration_sec:
            end = min(start + self.clip_duration_sec, video_duration_sec)
            features = self._segment_features(path, start, end)
            binary_probability, category, confidence, details = self._predict(features)
            clips.append(ClipAnomalyScore(
                clip_id=f"{video_id}_c{index:03d}", start_sec=round(start, 3),
                end_sec=round(end, 3), category=category, confidence=round(confidence, 6),
                is_anomaly=binary_probability >= self.anomaly_threshold,
                model_metadata={**self.model_metadata, **details},
            ))
            start += self.clip_stride_sec
            index += 1
        top = max(clips, key=lambda item: item.confidence)
        return VideoAnomalySummary(
            video_id=video_id, overall_category=top.category,
            overall_confidence=top.confidence, is_anomalous=any(c.is_anomaly for c in clips),
            anomaly_segments=clips, model_metadata=self.model_metadata,
        )
