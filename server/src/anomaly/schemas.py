"""Schemas for video anomaly detection and categorization."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ClipAnomalyScore(BaseModel):
    """Anomaly score and category prediction for a temporal video clip segment."""

    clip_id: str
    start_sec: float
    end_sec: float
    category: str  # Normal or one of 13 UCF-Crime anomalies
    confidence: float
    is_anomaly: bool


class VideoAnomalySummary(BaseModel):
    """Aggregate anomaly summary for an entire surveillance video."""

    video_id: str
    overall_category: str
    overall_confidence: float
    is_anomalous: bool
    anomaly_segments: List[ClipAnomalyScore] = Field(default_factory=list)

