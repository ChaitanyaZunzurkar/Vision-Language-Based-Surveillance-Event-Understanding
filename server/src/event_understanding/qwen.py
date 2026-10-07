"""Grounded Qwen2.5-VL event understanding."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import cv2
import numpy as np
import torch
from PIL import Image

from server.src.event_understanding.schemas import TemporalWindow, SemanticEvent, EventUnderstandingResult
from server.src.tracking.schemas import TrackingResult
from server.src.anomaly.schemas import VideoAnomalySummary
from server.src.utils.logger import logger


SYSTEM_PROMPT = """You describe only observable evidence in the supplied surveillance frames and structured detections.
Do not invent objects, actions, intent, identity, location, cause, crime, timestamps, or events.
Do not treat an anomaly model label as proof of criminal or suspicious behavior.
Use the supplied timestamps and track IDs only. If the evidence is insufficient, say so explicitly.
Return one concise factual description and no JSON, markdown, or recommendations."""


class VLMEventUnderstanding:
    def __init__(self, model_name: str, device: str = "auto", temperature: float = 0.2,
                 max_tokens: int = 1024, local_files_only: bool = False):
        self.model_name = model_name
        self.device_name = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local_files_only = local_files_only
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else (
            device if device != "auto" else "cpu")
        self.model = None
        self.processor = None
        self.load_error: Optional[str] = None

    def _initialize(self) -> None:
        if self.model is not None:
            return
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, local_files_only=self.local_files_only)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype="auto", device_map=self.device,
                local_files_only=self.local_files_only).eval()
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("Qwen2.5-VL loading failed: %s", self.load_error)
            raise RuntimeError(self.load_error) from exc

    @staticmethod
    def _frames(video_path: Path, start: float, end: float, count: int = 8) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open evidence video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        indices = np.linspace(max(0, int(start * fps)), max(0, int(end * fps) - 1),
                              max(1, count), dtype=np.int64)
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"No evidence frames in [{start}, {end}]")
        return frames

    def _describe(self, video_path: Path, start: float, end: float, evidence: Dict[str, Any]) -> str:
        self._initialize()
        frames = self._frames(video_path, start, end)
        images = [Image.fromarray(frame) for frame in frames]
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Evidence interval: {start:.3f}s to {end:.3f}s.\n"
            f"Structured evidence:\n{json.dumps(evidence, sort_keys=True, default=str)}\n"
            "Describe only what is visible and supported by this evidence."
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image} for image in images
        ] + [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        generation_args = {"max_new_tokens": self.max_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            generation_args["temperature"] = self.temperature
        with torch.no_grad():
            generated = self.model.generate(**inputs, **generation_args)
        output = self.processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True)[0].strip()
        if not output:
            raise RuntimeError("Qwen2.5-VL returned an empty description")
        return output

    def enrich_and_verify_events(self, video_id: str, windows: List[TemporalWindow],
                                  tracking_result: Optional[TrackingResult] = None,
                                  video_path: Optional[Path | str] = None,
                                  anomaly_summary: Optional[VideoAnomalySummary] = None
                                  ) -> EventUnderstandingResult:
        if video_path is None:
            raise ValueError("video_path is required for Qwen2.5-VL inference")
        track_map = {int(track.track_id): track for track in (tracking_result.tracks if tracking_result else [])}
        anomaly_segments = anomaly_summary.anomaly_segments if anomaly_summary else []
        events = []
        for window in windows:
            tracks = []
            for track_id in window.track_ids:
                track = track_map.get(int(track_id))
                if track:
                    tracks.append({
                        "track_id": int(track.track_id), "class": track.class_name,
                        "start_sec": track.start_sec, "end_sec": track.end_sec,
                    })
            for candidate in window.candidate_events:
                relevant_anomaly = next(
                    (segment for segment in anomaly_segments
                     if segment.end_sec >= candidate.start_sec and segment.start_sec <= candidate.end_sec), None)
                evidence = {
                    "candidate_event": candidate.model_dump(),
                    "tracks": tracks,
                    "anomaly": relevant_anomaly.model_dump() if relevant_anomaly else None,
                }
                description = self._describe(Path(video_path), candidate.start_sec, candidate.end_sec, evidence)
                event_id = f"ev_{video_id[:8]}_{window.window_id}_{len(events):04d}"
                events.append(SemanticEvent(
                    event_id=event_id, video_id=video_id, event_type=candidate.event_type,
                    start_sec=candidate.start_sec, end_sec=candidate.end_sec,
                    entity_ids=candidate.entity_ids, description=description,
                    confidence=candidate.confidence,
                    anomaly_category=relevant_anomaly.category if relevant_anomaly else "Normal",
                    anomaly_confidence=relevant_anomaly.confidence if relevant_anomaly else 0.0,
                    metadata={"window_id": window.window_id, "evidence": evidence,
                              "vlm_model": self.model_name},
                ))
        return EventUnderstandingResult(video_id=video_id, total_windows=len(windows),
                                        total_events=len(events), windows=windows, events=events)
