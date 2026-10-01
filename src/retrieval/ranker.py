"""Hybrid ranker combining vector similarity, metadata filtering, and event confidence."""

from typing import List, Dict, Any, Tuple
from src.storage.schemas import EventRecord
from src.retrieval.query_parser import ParsedQuery


class HybridRanker:
    """Ranks candidate surveillance events using hybrid metadata matching and semantic scores."""

    def __init__(
        self,
        vector_weight: float = 0.55,
        metadata_weight: float = 0.30,
        confidence_weight: float = 0.15,
    ):
        self.vector_weight = vector_weight
        self.metadata_weight = metadata_weight
        self.confidence_weight = confidence_weight

    def rank(
        self,
        events: List[EventRecord],
        vector_scores: Dict[str, float],
        parsed_query: ParsedQuery,
    ) -> List[Tuple[EventRecord, float]]:
        """Compute hybrid relevance score for each event.

        Args:
            events: List of EventRecord candidates.
            vector_scores: Dict mapping event_id -> cosine similarity score.
            parsed_query: Structured filters from query parser.

        Returns:
            List of (EventRecord, total_score) sorted descending by total_score.
        """
        scored_events: List[Tuple[EventRecord, float]] = []

        for ev in events:
            # 1. Vector score (0.0 to 1.0)
            v_score = vector_scores.get(ev.event_id, 0.0)

            # 2. Metadata matching score (0.0 to 1.0)
            meta_score = 0.0
            checks = 0

            # Match event type
            if parsed_query.event_type:
                checks += 1
                if ev.event_type == parsed_query.event_type:
                    meta_score += 1.0

            # Match anomaly category
            if parsed_query.anomaly_category:
                checks += 1
                if ev.anomaly_category.lower() == parsed_query.anomaly_category.lower():
                    meta_score += 1.0

            # Match time constraints
            if parsed_query.time_min_sec is not None:
                checks += 1
                if ev.end_sec >= parsed_query.time_min_sec:
                    meta_score += 0.5
            if parsed_query.time_max_sec is not None:
                checks += 1
                if ev.start_sec <= parsed_query.time_max_sec:
                    meta_score += 0.5

            norm_meta_score = (meta_score / checks) if checks > 0 else 0.5

            # 3. Underlying detection confidence
            conf_score = min(1.0, max(0.0, ev.confidence))

            # Composite weighted score
            total_score = (
                (self.vector_weight * v_score)
                + (self.metadata_weight * norm_meta_score)
                + (self.confidence_weight * conf_score)
            )

            scored_events.append((ev, round(total_score, 3)))

        scored_events.sort(key=lambda x: x[1], reverse=True)
        return scored_events
