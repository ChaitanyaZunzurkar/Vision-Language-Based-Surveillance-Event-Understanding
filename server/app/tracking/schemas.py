"""Tracking schemas for persistent surveillance entity trajectories."""

from typing import List, Tuple
from pydantic import BaseModel, Field
from server.app.detection.schemas import BoundingBox


class TrackObservation(BaseModel):
    """Observation of a tracked entity in a single frame."""

    frame_idx: int
    timestamp_sec: float
    bbox: BoundingBox
    confidence: float


class Track(BaseModel):
    """Persistent track representation of an entity across video frames."""

    track_id: int
    class_name: str
    start_sec: float
    end_sec: float
    observations: List[TrackObservation] = Field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return max(0.0, round(self.end_sec - self.start_sec, 2))

    @property
    def median_height(self) -> float:
        if not self.observations:
            return 1.0
        heights = [obs.bbox.height for obs in self.observations]
        heights.sort()
        mid = len(heights) // 2
        return heights[mid]

    @property
    def displacement(self) -> float:
        """Euclidean distance between first and last observation centers."""
        if len(self.observations) < 2:
            return 0.0
        first = self.observations[0].bbox
        last = self.observations[-1].bbox
        dx = last.center_x - first.center_x
        dy = last.center_y - first.center_y
        return float((dx**2 + dy**2) ** 0.5)

    def trajectory(self) -> List[Tuple[float, float, float]]:
        """List of (center_x, center_y, timestamp_sec)."""
        return [
            (obs.bbox.center_x, obs.bbox.center_y, obs.timestamp_sec)
            for obs in self.observations
        ]


class TrackingResult(BaseModel):
    """Result of tracking for a surveillance video."""

    video_id: str
    total_tracks: int
    tracks: List[Track] = Field(default_factory=list)

