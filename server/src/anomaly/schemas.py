"""Schemas for video anomaly detection and categorization."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ClipAnomalyScore(BaseModel):
    """Anomaly score and category prediction for a temporal video clip segment."""

    clip_id: str
    start_sec: float
    end_sec: float
    category: str  # Normal or one of 13 UCF-Crime anomalies
    confidence: float
    is_anomaly: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VideoAnomalySummary(BaseModel):
    """Aggregate anomaly summary for an entire surveillance video."""

    video_id: str
    overall_category: str
    overall_confidence: float
    is_anomalous: bool
    model_available: bool = False
    model_name: Optional[str] = None
    anomaly_segments: List[ClipAnomalyScore] = Field(default_factory=list)
