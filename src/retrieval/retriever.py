"""Natural-language surveillance event retriever and grounded answer generator."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.storage.schemas import EventRecord
from src.storage.metadata_store import MetadataStore, metadata_store
from src.storage.vector_store import VectorStore, vector_store
from src.retrieval.query_parser import QueryParser, query_parser, ParsedQuery
from src.retrieval.ranker import HybridRanker
from src.utils.logger import logger


class QueryResult(BaseModel):
    """Grounded query response with evidence events and playback references."""

    query: str
    parsed_filters: Dict[str, Any]
    total_matches: int
    matched_events: List[Dict[str, Any]] = Field(default_factory=list)
    grounded_summary: str


class SurveillanceRetriever:
    """Retrieves surveillance events matching natural-language queries using hybrid search."""

    def __init__(
        self,
        meta_store: Optional[MetadataStore] = None,
        vec_store: Optional[VectorStore] = None,
        parser: Optional[QueryParser] = None,
    ):
        self.meta_store = meta_store or metadata_store
        self.vec_store = vec_store or vector_store
        self.parser = parser or query_parser
        self.ranker = HybridRanker()

    def query(
        self, query_text: str, top_k: int = 5, min_score: float = 0.20,
        video_id: Optional[str] = None,
    ) -> QueryResult:
        """Execute hybrid natural-language retrieval and produce grounded report."""
        # 1. Parse natural language query
        parsed = self.parser.parse(query_text)

        # 2. Vector search over semantic corpus
        vector_results = self.vec_store.search(
            parsed.cleaned_query, top_k=top_k * 2, min_score=0.05
        )
        vector_scores = dict(vector_results)

        # 3. Retrieve candidates from SQLite
        # If metadata filters matched, retrieve filtered candidates, else get all recent events
        candidates = self.meta_store.get_events(
            video_id=video_id,
            event_type=parsed.event_type,
            anomaly_category=parsed.anomaly_category,
            time_start=parsed.time_min_sec,
            time_end=parsed.time_max_sec,
        )

        # If candidates are empty (e.g. strict filters yielded 0), fallback to all events for semantic ranking
        if not candidates and video_id is None:
            candidates = self.meta_store.get_events()

        # 4. Rank candidates using HybridRanker
        ranked_pairs = self.ranker.rank(candidates, vector_scores, parsed)
        filtered_pairs = [p for p in ranked_pairs if p[1] >= min_score][:top_k]

        matched_event_dicts: List[Dict[str, Any]] = []
        for ev, score in filtered_pairs:
            matched_event_dicts.append(
                {
                    "event_id": ev.event_id,
                    "video_id": ev.video_id,
                    "event_type": ev.event_type,
                    "start_sec": ev.start_sec,
                    "end_sec": ev.end_sec,
                    "entity_ids": ev.entity_ids,
                    "description": ev.description,
                    "confidence": ev.confidence,
                    "relevance_score": score,
                    "anomaly_category": ev.anomaly_category,
                    "clip_path": ev.clip_path,
                }
            )

        # 5. Generate Grounded Summary (RAG principle - no intent hallucination)
        grounded_summary = self._generate_grounded_summary(
            query_text, matched_event_dicts, parsed
        )

        logger.info(
            f"Query '{query_text}' returned {len(matched_event_dicts)} grounded events"
        )
        return QueryResult(
            query=query_text,
            parsed_filters=parsed.model_dump(),
            total_matches=len(matched_event_dicts),
            matched_events=matched_event_dicts,
            grounded_summary=grounded_summary,
        )

    def _generate_grounded_summary(
        self, query: str, matches: List[Dict[str, Any]], parsed: ParsedQuery
    ) -> str:
        """Formulate a strictly factual response grounded only in retrieved surveillance records."""
        if not matches:
            filters_applied = []
            if parsed.event_type:
                filters_applied.append(f"event type '{parsed.event_type}'")
            if parsed.anomaly_category:
                filters_applied.append(f"anomaly '{parsed.anomaly_category}'")
            filter_str = (
                f" with {', '.join(filters_applied)}" if filters_applied else ""
            )
            return f"No verified surveillance events found matching '{query}'{filter_str}."

        lines = [
            f"Retrieved {len(matches)} evidence-grounded event(s) relevant to '{query}':\n"
        ]
        for i, m in enumerate(matches, 1):
            clip_note = (
                f" [Evidence clip available: {Path(m['clip_path']).name}]"
                if m.get("clip_path")
                else ""
            )
            lines.append(
                f"{i}. Video '{m['video_id']}' ({m['start_sec']:.1f}s - {m['end_sec']:.1f}s): "
                f"{m['description']} (Event: {m['event_type']}, Confidence: {m['confidence']:.2f}, Anomaly: {m['anomaly_category']}){clip_note}"
            )

        lines.append(
            "\nNote: All summaries are grounded directly in extracted trajectories and verified video timestamps without inferring intent."
        )
        return "\n".join(lines)


# Global retriever instance
retriever = SurveillanceRetriever()
