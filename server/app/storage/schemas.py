"""Database models and schema representations for surveillance storage."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class VideoRecord(BaseModel):
    """Metadata record of an ingested surveillance video."""

    video_id: str
    filename: str
    file_path: str
    duration_sec: float
    fps: float
    frame_count: int
    resolution: str
    status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class EventRecord(BaseModel):
    """Database record for a structured surveillance event."""

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
    clip_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AnomalyRecord(BaseModel):
    """Database record for video anomaly segment."""

    anomaly_id: str
    video_id: str
    category: str
    confidence: float
    is_anomaly: bool
    start_sec: float
    end_sec: float


class TrackRecord(BaseModel):
    """Database record for an entity track."""

    track_id: int
    video_id: str
    class_name: str
    start_sec: float
    end_sec: float
    total_observations: int
