"""Surveillance events query and inspection API endpoints."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import uuid
import math
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form
from pydantic import BaseModel
from src.storage.schemas import EventRecord
from src.storage.metadata_store import MetadataStore
from app.api.deps import get_metadata_store
from src.storage.vector_store import vector_store
from src.evidence.clip_generator import ClipGenerator

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
    store: MetadataStore = Depends(get_metadata_store),
):
    """Import Stage 4's events.json output, including the VISTA windows/events format."""
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    if Path(file.filename or "").suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="Upload a Stage 4 .json file")

    try:
        payload = json.loads((await file.read()).decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}") from exc

    raw_events = []
    if isinstance(payload, dict) and isinstance(payload.get("windows"), list):
        for window_index, window in enumerate(payload["windows"]):
            if isinstance(window, dict):
                window_events = window.get("events", [])
                if not isinstance(window_events, list):
                    raise HTTPException(status_code=400, detail=f"Window {window_index} has an invalid events list")
                for event_index, event in enumerate(window_events):
                    raw_events.append((event, window_index, event_index, window))
    elif isinstance(payload, dict) and isinstance(payload.get("events"), list):
        raw_events = [(event, 0, i, {}) for i, event in enumerate(payload["events"])]
    else:
        raise HTTPException(status_code=400, detail="Expected a top-level 'windows' or 'events' array")

    source_video_ids = set()
    if isinstance(payload, dict):
        if isinstance(payload.get("windows"), list):
            source_video_ids = {
                str(window["video_id"]) for window in payload["windows"]
                if isinstance(window, dict) and window.get("video_id") is not None
            }
        elif isinstance(payload.get("events"), list):
            source_video_ids = {
                str(event["video_id"]) for event in payload["events"]
                if isinstance(event, dict) and event.get("video_id") is not None
            }
    allowed_ids = {video.video_id, Path(video.filename).stem}
    if source_video_ids and not source_video_ids.issubset(allowed_ids):
        raise HTTPException(
            status_code=409,
            detail=f"This file contains video ID(s) {sorted(source_video_ids)}; selected video is {video.filename}.",
        )

    records = []
    for event, window_index, event_index, window in raw_events:
        if not isinstance(event, dict):
            raise HTTPException(status_code=400, detail=f"Event {event_index} is not an object")
        event_type = event.get("event_type") or event.get("type")
        start = event.get("start_sec", event.get("start"))
        end = event.get("end_sec", event.get("end"))
        description = event.get("description")
        if not event_type or start is None or end is None or not description:
            raise HTTPException(status_code=400, detail=f"Event {event_index} is missing type, times, or description")
        try:
            start, end = float(start), float(end)
            confidence = float(event.get("confidence", 0.5))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Event {event_index} has invalid numeric fields") from exc
        window_range = window.get("window")
        if isinstance(window_range, list) and len(window_range) == 2:
            try:
                window_start, window_end = float(window_range[0]), float(window_range[1])
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"Window {window_index} has invalid timestamps") from exc
            if start < window_start and end <= window_end - window_start + 1:
                # Stage 4 may report seconds relative to the sampled window.
                start += window_start
                end += window_start
        if start < 0 or end < start or end > video.duration_sec + 1:
            raise HTTPException(status_code=400, detail=f"Event {event_index} has timestamps outside the selected video")
        if not math.isfinite(start) or not math.isfinite(end) or not math.isfinite(confidence):
            raise HTTPException(status_code=400, detail=f"Event {event_index} contains non-finite numeric fields")

        subject = event.get("subject_track")
        object_track = event.get("object_track")
        entity_ids = event.get("entity_ids")
        if entity_ids is None:
            entity_ids = [item for item in (subject, object_track) if item is not None]
        try:
            entity_ids = [int(value) for value in entity_ids]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Event {event_index} has invalid track IDs") from exc

        source_id = event.get("event_id") or f"{window_index}:{event_index}"
        safe_id = f"ev_{uuid.uuid5(uuid.NAMESPACE_URL, f'{video_id}:{source_id}').hex[:20]}"
        records.append(EventRecord(
            event_id=safe_id,
            video_id=video_id,
            event_type=str(event_type).lower(),
            start_sec=start,
            end_sec=end,
            entity_ids=entity_ids,
            description=str(description),
            confidence=max(0.0, min(1.0, confidence)),
            anomaly_category="Unassessed",
            anomaly_confidence=0.0,
            clip_path=None,
            metadata={
                "source": "stage4_import",
                "source_event_id": event.get("event_id"),
                "camera_id": event.get("camera_id", window.get("camera_id")),
                "subject_track": subject,
                "object_track": object_track,
                "zone": event.get("zone"),
                "window": window.get("window"),
            },
        ))

    for record in records:
        record.clip_path = clip_generator.cut_evidence_clip(
            video.file_path, record.start_sec, record.end_sec, record.event_id, video.duration_sec
        )
    store.insert_events(records)
    vector_store.add_events(records)
    if records:
        store.update_video_status(video_id, "completed")
    return {"video_id": video_id, "imported_events": len(records), "status": "completed" if records else video.status}
