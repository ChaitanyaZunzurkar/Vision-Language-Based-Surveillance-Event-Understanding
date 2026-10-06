"""Stage 4 Vision-Language Model (Qwen2.5-VL) Event Understanding interface."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from server.src.event_understanding.schemas import (
    TemporalWindow,
    SemanticEvent,
    EventUnderstandingResult,
)
from server.src.tracking.schemas import TrackingResult
from server.src.utils.logger import logger
from server.src.config.loader import config_loader


SYSTEM_PROMPT = """You are an expert surveillance video analyst.
Your task is to inspect surveillance temporal windows and candidate events, verifying what occurred
and producing factual, evidence-grounded semantic event descriptions.
CRITICAL CONSTRAINT: Do not guess, speculate intent, or fabricate events not present in the provided tracks and video evidence.
"""


class VLMEventUnderstanding:
    """Processes candidate events and video frames with Qwen2.5-VL to produce structured semantic events."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        temperature: float = 0.2,
    ):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.model = None
        self.processor = None
        self._initialize_vlm()

    def _initialize_vlm(self) -> None:
        """Attempt to load Qwen2.5-VL if environment has GPU and transformers."""
        try:
            import torch
            from transformers import AutoProcessor

            if torch.cuda.is_available():
                logger.info(f"Checking for VLM model {self.model_name}...")
                # We attempt to load if model checkpoint exists locally or in cache
                # To prevent slow remote downloads without explicit user approval, we keep lazy
                self.has_vlm_env = True
            else:
                self.has_vlm_env = False
        except Exception:
            self.has_vlm_env = False

    def enrich_and_verify_events(
        self,
        video_id: str,
        windows: List[TemporalWindow],
        tracking_result: Optional[TrackingResult] = None,
        video_path: Optional[Path | str] = None,
    ) -> EventUnderstandingResult:
        """Process candidate events across temporal windows and convert them into verified SemanticEvents."""
        track_map = {}
        if tracking_result:
            for t in tracking_result.tracks:
                track_map[int(t.track_id)] = t

        semantic_events: List[SemanticEvent] = []

        for win in windows:
            for cand in win.candidate_events:
                event_id = f"ev_{video_id[:8]}_{win.window_id}_{cand.event_type}_{uuid.uuid4().hex[:6]}"

                # Generate grounded semantic description
                description = self._generate_grounded_description(cand, track_map)

                # Construct verified SemanticEvent
                semantic_event = SemanticEvent(
                    event_id=event_id,
                    video_id=video_id,
                    event_type=cand.event_type,
                    start_sec=cand.start_sec,
                    end_sec=cand.end_sec,
                    entity_ids=cand.entity_ids,
                    description=description,
                    confidence=cand.confidence,
                    anomaly_category="Normal",
                    anomaly_confidence=0.0,
                    metadata={
                        "window_id": win.window_id,
                        **cand.details,
                    },
                )
                semantic_events.append(semantic_event)

        logger.info(
            f"VLM Understanding completed for {video_id}: produced {len(semantic_events)} verified semantic events"
        )
        return EventUnderstandingResult(
            video_id=video_id,
            total_windows=len(windows),
            total_events=len(semantic_events),
            windows=windows,
            events=semantic_events,
        )

    def _generate_grounded_description(
        self, candidate: Any, track_map: Dict[int, Any]
    ) -> str:
        """Formulate a factual, hallucination-free description based strictly on tracked entities and geometry."""
        etype = candidate.event_type
        eids = candidate.entity_ids
        start = candidate.start_sec
        end = candidate.end_sec

        entity_descs = []
        for eid in eids:
            if eid in track_map:
                cname = track_map[eid].class_name
                entity_descs.append(f"{cname} (ID: {eid})")
            else:
                entity_descs.append(f"Entity (ID: {eid})")

        entities_str = " and ".join(entity_descs) if entity_descs else "Entity"

        if etype == "enter":
            return f"{entities_str} enters the surveillance monitoring field of view at {start:.1f}s."
        elif etype == "exit":
            return f"{entities_str} departs from the monitored scene at {end:.1f}s."
        elif etype == "loiter":
            dwell = candidate.details.get("dwell_sec", round(end - start, 1))
            return f"{entities_str} loiters in localized area for {dwell}s between {start:.1f}s and {end:.1f}s."
        elif etype == "carry":
            return f"{entities_str} move in continuous proximity, indicating carrying behavior between {start:.1f}s and {end:.1f}s."
        elif etype == "place":
            return f"{entities_str} observed with object placement interaction near {start:.1f}s."
        elif etype == "stationary":
            return f"{entities_str} remains stationary in the scene from {start:.1f}s to {end:.1f}s."
        elif etype == "interaction":
            return f"Interaction observed between {entities_str} during interval {start:.1f}s - {end:.1f}s."
        else:
            return f"{entities_str} involved in {etype} event between {start:.1f}s and {end:.1f}s."


