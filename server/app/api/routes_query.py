"""Natural-language surveillance video query API endpoint."""

from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from server.src.retrieval.retriever import SurveillanceRetriever, QueryResult
from server.app.api.deps import get_retriever
from server.src.storage.metadata_store import MetadataStore
from server.src.storage.vector_store import VectorStore
from server.app.api.deps import get_metadata_store, get_vector_store

router = APIRouter(prefix="/query", tags=["Query & Retrieval"])


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    video_id: Optional[str] = Field(default=None, description="Limit the search to one video")


@router.post("", response_model=QueryResult)
def natural_language_query(
    request: QueryRequest,
    retriever: SurveillanceRetriever = Depends(get_retriever),
):
    """Execute hybrid natural-language search over surveillance archive.

    Matches events using semantic vector search combined with structured metadata filtering
    (e.g., event type, anomaly category, time interval) and generates a grounded response.
    """
    result = retriever.query(
        query_text=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        video_id=request.video_id,
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

