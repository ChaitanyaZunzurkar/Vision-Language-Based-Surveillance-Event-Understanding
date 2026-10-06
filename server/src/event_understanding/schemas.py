"""Schemas for temporal windowing and vision-language event understanding."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CandidateEvent(BaseModel):
    """Rule-based candidate event extracted from track trajectories within a temporal window."""

    event_type: str  # enter, exit, loiter, carry, place, leave, stationary, interaction
    start_sec: float
    end_sec: float
    entity_ids: List[int]
    confidence: float = 0.80
    details: Dict[str, Any] = Field(default_factory=dict)


class TemporalWindow(BaseModel):
    """Temporal window grouping tracks and candidate events."""

    window_id: int
    start_sec: float
    end_sec: float
    track_ids: List[int] = Field(default_factory=list)
    candidate_events: List[CandidateEvent] = Field(default_factory=list)


class SemanticEvent(BaseModel):
    """High-level semantic event validated and described by VLM, stored in database."""

    event_id: str
    video_id: str
    event_type: str
    start_sec: float
    end_sec: float
    entity_ids: List[int] = Field(default_factory=list)
    description: str
    confidence: float
    anomaly_category: str = "Normal"
    anomaly_confidence: float = 0.0
    vlm_verified: bool = False
    evidence_clip_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EventUnderstandingResult(BaseModel):
    """Full event understanding output for a surveillance video."""

    video_id: str
    total_windows: int
    total_events: int
    windows: List[TemporalWindow] = Field(default_factory=list)
    events: List[SemanticEvent] = Field(default_factory=list)
