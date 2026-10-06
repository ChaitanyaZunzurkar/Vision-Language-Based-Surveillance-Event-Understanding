"""Detection schemas for surveillance entity localization."""

from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates in pixels [x1, y1, x2, y2]."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height


class Detection(BaseModel):
    """Single object detection event."""

    frame_idx: int
    timestamp_sec: float
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class DetectionResult(BaseModel):
    """Result of batch detection on a surveillance video."""

    video_id: str
    total_detections: int
    detections: List[Detection] = Field(default_factory=list)
