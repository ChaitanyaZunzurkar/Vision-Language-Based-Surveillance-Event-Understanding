"""Natural-language surveillance video query API endpoint."""

from pathlib import Path
import re
from typing import Dict, Any, List, Literal, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from server.src.retrieval.retriever import SurveillanceRetriever, QueryResult
from server.app.api.deps import get_pipeline, get_retriever
from server.src.storage.metadata_store import MetadataStore
from server.src.storage.vector_store import VectorStore
from server.app.api.deps import get_metadata_store, get_vector_store
from server.src.pipeline.pipeline import SurveillancePipeline
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
    r"then|next|before|after|again|more|also|what about|how about)\b",
    re.IGNORECASE,
)


@router.post("", response_model=QueryResult)
def natural_language_query(
    request: QueryRequest,
    retriever: SurveillanceRetriever = Depends(get_retriever),
    pipeline: SurveillancePipeline = Depends(get_pipeline),
):
    """Execute hybrid natural-language search over surveillance archive.

    Matches events using semantic vector search combined with structured metadata filtering
    (e.g., event type, anomaly category, time interval) and generates a grounded response.
    """
    latest_user_turn = next(
        (turn.content for turn in reversed(request.history) if turn.role == "user"),
        None,
    )
    retrieval_query = request.query
    if latest_user_turn and _FOLLOW_UP_PATTERN.search(request.query):
        retrieval_query = f"{latest_user_turn}\nFollow-up: {request.query}"

    result = retriever.query(
        query_text=retrieval_query,
        top_k=request.top_k,
        min_score=request.min_score,
        video_id=request.video_id,
    )
    result.query = request.query

    if result.matched_events:
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

    # Attach browser-accessible streaming URLs for evidence clips
    for match in result.matched_events:
        clip_path = match.get("clip_path")
        if clip_path:
            clip_name = Path(clip_path).name
            match["clip_url"] = f"/api/media/clip/{clip_name}"
        else:
            match["clip_url"] = None

    return result


@router.post("/rebuild-index")
def rebuild_index(
    store: MetadataStore = Depends(get_metadata_store),
    vec_store: VectorStore = Depends(get_vector_store),
):
    """Rebuild FAISS exclusively from current SQLite event records."""
    events = store.get_events()
    vec_store.rebuild(events)
    return {"status": "rebuilt", "indexed_events": len(events)}
