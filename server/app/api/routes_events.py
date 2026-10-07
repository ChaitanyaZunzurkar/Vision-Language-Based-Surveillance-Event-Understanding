"""Surveillance events query and inspection API endpoints."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import uuid
import math
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form
from pydantic import BaseModel
from server.src.storage.schemas import EventRecord
from server.src.storage.metadata_store import MetadataStore
from server.app.api.deps import get_metadata_store
from server.src.storage.vector_store import vector_store
from server.src.evidence.clip_generator import ClipGenerator

router = APIRouter(prefix="/events", tags=["Events"])
clip_generator = ClipGenerator()


class EventDetailResponse(BaseModel):
    event_id: str
    video_id: str
    event_type: str
    start_sec: float
    end_sec: float
    entity_ids: List[int]
    description: str
    confidence: float
    anomaly_category: str
    anomaly_confidence: float
    vlm_verified: bool
    clip_url: Optional[str] = None
    created_at: str


def to_event_response(ev: EventRecord) -> EventDetailResponse:
    clip_url = None
    if ev.clip_path:
        clip_name = Path(ev.clip_path).name
        clip_url = f"/api/media/clip/{clip_name}"

    return EventDetailResponse(
        event_id=ev.event_id,
        video_id=ev.video_id,
        event_type=ev.event_type,
        start_sec=ev.start_sec,
        end_sec=ev.end_sec,
        entity_ids=ev.entity_ids,
        description=ev.description,
        confidence=ev.confidence,
        anomaly_category=ev.anomaly_category,
        anomaly_confidence=ev.anomaly_confidence,
        vlm_verified=ev.vlm_verified,
        clip_url=clip_url,
        created_at=ev.created_at,
    )


@router.get("", response_model=List[EventDetailResponse])
def get_events(
    video_id: Optional[str] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    anomaly_category: Optional[str] = Query(default=None),
    time_start: Optional[float] = Query(default=None),
    time_end: Optional[float] = Query(default=None),
    min_confidence: float = Query(default=0.0),
    store: MetadataStore = Depends(get_metadata_store),
):
    """Retrieve filtered surveillance events."""
    events = store.get_events(
        video_id=video_id,
        event_type=event_type,
        anomaly_category=anomaly_category,
        time_start=time_start,
        time_end=time_end,
        min_confidence=min_confidence,
    )
    return [to_event_response(e) for e in events]


@router.get("/{event_id}", response_model=EventDetailResponse)
def get_event(event_id: str, store: MetadataStore = Depends(get_metadata_store)):
    """Retrieve single event details and evidence clip link."""
    ev = store.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    return to_event_response(ev)


@router.post("/import")
async def import_stage4_events(
    video_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Reject legacy imports so database descriptions always come from Qwen2.5-VL."""
    raise HTTPException(
        status_code=410,
        detail="Legacy event imports are disabled. Run /api/pipeline/run/{video_id} for grounded Qwen2.5-VL inference.",
    )
