"""Natural language query parser for surveillance event retrieval."""

from typing import Dict, Any, Optional, List
import re
from pydantic import BaseModel, Field
from server.src.anomaly.detector import CATEGORIES as ANOMALY_CATEGORIES


class ParsedQuery(BaseModel):
    """Structured search filters extracted from a natural language query."""

    original_query: str
    cleaned_query: str
    event_type: Optional[str] = None
    anomaly_category: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    time_min_sec: Optional[float] = None
    time_max_sec: Optional[float] = None


EVENT_KEYWORDS = {
    "enter": ["enter", "entered", "enters", "entering", "walks in", "came in", "arrival", "arrives"],
    "exit": ["exit", "exited", "exits", "exiting", "left", "leaves", "leaving", "departure"],
    "loiter": ["loiter", "loitering", "loitered", "hanging around", "lingering", "waiting"],
    "carry": ["carry", "carrying", "carried", "holding", "takes bag", "carrying bag"],
    "place": ["place", "placed", "placing", "put down", "leaves bag", "set down", "drop"],
    "leave": ["abandoned", "left behind", "unattended", "leave"],
    "stationary": ["stationary", "standing", "stopped", "parked", "idle"],
    "interaction": ["interact", "interaction", "fighting", "talking", "meeting"],
}

ENTITY_KEYWORDS = {
    "Person": ["person", "people", "man", "woman", "guy", "someone", "individual", "pedestrian"],
    "Car": ["car", "vehicle", "automobile", "sedan"],
    "Backpack": ["bag", "backpack", "pack", "rucksack"],
    "Suitcase": ["suitcase", "luggage", "briefcase"],
    "Bicycle": ["bike", "bicycle", "cyclist"],
}


class QueryParser:
    """Extracts structured filters and semantic intent from natural language surveillance queries."""

    def parse(self, query: str) -> ParsedQuery:
        """Parse natural language query into structured metadata constraints."""
        q_lower = query.lower()

        # 1. Match event type
        matched_event = None
        for etype, synonyms in EVENT_KEYWORDS.items():
            if any(syn in q_lower for syn in synonyms):
                matched_event = etype
                break

        # 2. Match anomaly category
        matched_anomaly = None
        for cat in ANOMALY_CATEGORIES:
            if cat.lower() != "normal" and cat.lower() in q_lower:
                matched_anomaly = cat
                break

        # 3. Match entities
        matched_entities = []
        for ent, synonyms in ENTITY_KEYWORDS.items():
            if any(syn in q_lower for syn in synonyms):
                matched_entities.append(ent)

        # 4. Match time constraints (e.g., "after 10s", "between 5 and 20s")
        time_min = None
        time_max = None

        m_between = re.search(r"between\s+(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)", q_lower)
        if m_between:
            time_min = float(m_between.group(1))
            time_max = float(m_between.group(2))
        else:
            m_after = re.search(r"(?:after|from)\s+(\d+(?:\.\d+)?)", q_lower)
            if m_after:
                time_min = float(m_after.group(1))
            m_before = re.search(r"(?:before|until)\s+(\d+(?:\.\d+)?)", q_lower)
            if m_before:
                time_max = float(m_before.group(1))

        return ParsedQuery(
            original_query=query,
            cleaned_query=query.strip(),
            event_type=matched_event,
            anomaly_category=matched_anomaly,
            entities=matched_entities,
            time_min_sec=time_min,
            time_max_sec=time_max,
        )


# Global query parser instance
query_parser = QueryParser()


