"""Natural-language surveillance video query API endpoint."""

from pathlib import Path
import re
from typing import Dict, Any, List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from server.src.retrieval.retriever import SurveillanceRetriever, QueryResult
from server.app.api.deps import (
    get_metadata_store,
    get_pipeline,
    get_retriever,
    get_vector_store,
)
from server.src.pipeline.pipeline import SurveillancePipeline
from server.src.storage.metadata_store import MetadataStore
from server.src.storage.vector_store import VectorStore
from server.src.utils.logger import logger

router = APIRouter(prefix="/query", tags=["Query & Retrieval"])


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    video_id: Optional[str] = Field(default=None, description="Limit the search to one video")
    history: List[ConversationTurn] = Field(default_factory=list, max_length=8)


_FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:it|its|they|them|their|he|she|him|her|this|that|these|those|there|"
    r"then|next|before|after|again|more|also|time|when|where|who|how|what else|"
    r"what about|how about)\b",
    re.IGNORECASE,
)
_TIMELINE_PATTERN = re.compile(
    r"\b(?:timeline|what happened|summari[sz]e|summary|overview|"
    r"describe (?:the )?video|sequence of events)\b",
    re.IGNORECASE,
)
_RETRY_PATTERN = re.compile(
    r"^\s*(?:try again|again|repeat(?: that)?|one more time)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


@router.post("", response_model=QueryResult)
def natural_language_query(
    request: QueryRequest,
    retriever: SurveillanceRetriever = Depends(get_retriever),
    pipeline: SurveillancePipeline = Depends(get_pipeline),
    store: MetadataStore = Depends(get_metadata_store),
):
    """Execute hybrid natural-language search over surveillance archive.

    Matches events using semantic vector search combined with structured metadata filtering
    (e.g., event type, anomaly category, time interval) and generates a grounded response.
    """
    if request.video_id:
        video = store.get_video(request.video_id)
        if video and video.status in {"pending", "processing"}:
            raise HTTPException(
                status_code=409,
                detail="This video's analysis has not completed. Start or wait for analysis to finish, then ask again.",
            )
        if video and video.status == "failed":
            raise HTTPException(
                status_code=409,
                detail="This video's analysis failed. Retry its pipeline before asking questions about it.",
            )

    latest_user_turn = next(
        (
            turn.content
            for turn in reversed(request.history)
            if turn.role == "user" and not _RETRY_PATTERN.fullmatch(turn.content)
        ),
        None,
    )
    retrieval_query = request.query
    if latest_user_turn and _RETRY_PATTERN.fullmatch(request.query):
        retrieval_query = latest_user_turn
    elif latest_user_turn and _FOLLOW_UP_PATTERN.search(request.query):
        retrieval_query = f"{latest_user_turn}\nFollow-up: {request.query}"

    if request.video_id and _TIMELINE_PATTERN.search(request.query):
        parsed = retriever.parser.parse(request.query)
        events = store.get_events(video_id=request.video_id)
        matches = [
            {
                "event_id": event.event_id,
                "video_id": event.video_id,
                "event_type": event.event_type,
                "start_sec": event.start_sec,
                "end_sec": event.end_sec,
                "entity_ids": event.entity_ids,
                "description": event.description,
                "confidence": event.confidence,
                "relevance_score": 1.0,
                "anomaly_category": event.anomaly_category,
                "anomaly_confidence": event.anomaly_confidence,
                "clip_path": event.clip_path,
            }
            for event in events[:50]
        ]
        result = QueryResult(
            query=request.query,
            parsed_filters=parsed.model_dump(),
            total_matches=len(matches),
            matched_events=matches,
            grounded_summary=retriever._generate_grounded_summary(
                request.query, matches, parsed
            ),
        )
    else:
        result = retriever.query(
            query_text=retrieval_query,
            top_k=request.top_k,
            min_score=request.min_score,
            video_id=request.video_id,
        )
        result.query = request.query

    if result.matched_events:
        conversational_answer = None
        try:
            conversational_answer = pipeline.vlm.generate_grounded_answer(
                request.query,
                result.matched_events,
                [turn.model_dump() for turn in request.history],
            )
            if conversational_answer:
                result.grounded_summary = conversational_answer
        except Exception:
            logger.exception(
                "Local Qwen chat answer failed; returning the grounded event summary"
            )
        result.grounded_summary = conversational_answer or _conversational_fallback(
            request.query, result.matched_events
        )
    else:
        result.grounded_summary = (
            "I couldn't find a verified event that answers that in the selected "
            "footage. Try asking about a specific action, person, or time."
        )

    # Attach browser-accessible streaming URLs for evidence clips
    for match in result.matched_events:
        clip_path = match.get("clip_path")
        if clip_path:
            clip_name = Path(clip_path).name
            match["clip_url"] = f"/api/media/clip/{clip_name}"
        else:
            match["clip_url"] = None

    return result


def _conversational_fallback(query: str, events: List[Dict[str, Any]]) -> str:
    """Return a short, natural answer from evidence when local generation is unavailable."""
    ordered = sorted(events, key=lambda event: float(event.get("start_sec", 0.0)))
    timeline_request = _TIMELINE_PATTERN.search(query)
    selected = ordered[:6] if timeline_request else events[:3]
    if len(selected) == 1:
        event = selected[0]
        return (
            f"I found one relevant moment, from {event['start_sec']:.1f} to "
            f"{event['end_sec']:.1f} seconds: {event['description']}"
        )
    if timeline_request:
        lines = [
            f"Here’s the timeline from the processed event records "
            f"({len(ordered)} events total):"
        ]
        lines.extend(
            f"{event['start_sec']:.1f}–{event['end_sec']:.1f}s: {event['description']}"
            for event in selected
        )
        if len(ordered) > len(selected):
            lines.append(f"I’ve included the first {len(selected)} moments here.")
        return "\n".join(lines)

    lines = ["I found a few relevant moments in the footage:"]
    lines.extend(
        f"Around {event['start_sec']:.1f}–{event['end_sec']:.1f} seconds, "
        f"{event['description']}"
        for event in selected
    )
    return "\n".join(lines)


@router.post("/rebuild-index")
def rebuild_index(
    store: MetadataStore = Depends(get_metadata_store),
    vec_store: VectorStore = Depends(get_vector_store),
):
    """Rebuild FAISS exclusively from current SQLite event records."""
    events = store.get_events()
    vec_store.rebuild(events)
    return {"status": "rebuilt", "indexed_events": len(events)}
