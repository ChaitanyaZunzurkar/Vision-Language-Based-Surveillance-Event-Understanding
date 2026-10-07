"""Persistent FAISS semantic index for SQLite surveillance events."""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import threading

import numpy as np

from server.src.storage.schemas import EventRecord
from server.src.utils.paths import paths
from server.src.utils.logger import logger
from server.src.config.loader import config_loader


class VectorStore:
    """Sentence-Transformer embeddings backed by a normalized FAISS index.

    SQLite owns event records. This class only owns vectors and the
    ``FAISS row -> event_id`` mapping, which can always be rebuilt.
    """

    def __init__(self, index_path: Optional[Path | str] = None):
        configured_path = config_loader.get(
            "retrieval.index_path", "server/data/vectors/events.faiss"
        )
        self.index_path = Path(index_path) if index_path else (paths.root_dir / configured_path)
        self.mapping_path = self.index_path.with_suffix(".mapping.json")
        self.model_name = config_loader.get(
            "retrieval.embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._model = None
        self._faiss = None
        self._lock = threading.RLock()
        self.event_ids: List[str] = []
        self.dimension: Optional[int] = None
        self.index = None
        self.load()

    def _load_dependencies(self) -> None:
        if self._faiss is None:
            try:
                import faiss
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic retrieval requires faiss and sentence-transformers. "
                    "Install server/requirements.txt."
                ) from exc
            self._faiss = faiss
            self._model = SentenceTransformer(self.model_name)
            self.dimension = int(self._model.get_sentence_embedding_dimension())
            logger.info("Loaded embedding model %s (dimension=%s)", self.model_name, self.dimension)

    @staticmethod
    def _text(event: EventRecord) -> str:
        entities = " ".join(str(entity) for entity in event.entity_ids)
        return (
            f"event type {event.event_type}; {event.description}; "
            f"anomaly {event.anomaly_category}; entities {entities}"
        )

    def _encode(self, texts: List[str]) -> np.ndarray:
        self._load_dependencies()
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def add_events(self, events: List[EventRecord]) -> None:
        if not events:
            return
        with self._lock:
            known = set(self.event_ids)
            new_events = [event for event in events if event.event_id not in known]
            if self.index is not None and new_events:
                vectors = self._encode([self._text(event) for event in new_events])
                self.index.add(vectors)
                self.event_ids.extend(event.event_id for event in new_events)
                self._records = {
                    **getattr(self, "_records", {}),
                    **{event.event_id: event for event in new_events},
                }
                self.save()
                return
            self.rebuild_from_events(self._events_with_updates(events))

    def _events_with_updates(self, updates: List[EventRecord]) -> List[EventRecord]:
        current: Dict[str, EventRecord] = getattr(self, "_records", {})
        current.update({event.event_id: event for event in updates})
        self._records = current
        return list(current.values())

    def rebuild_from_events(self, events: List[EventRecord]) -> None:
        """Recreate the FAISS index from authoritative SQLite records."""
        with self._lock:
            self._records = {event.event_id: event for event in events}
            self.event_ids = list(self._records)
            if not self.event_ids:
                self.clear()
                return
            vectors = self._encode([self._text(event) for event in self._records.values()])
            self.index = self._faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
            self.dimension = vectors.shape[1]
            self.save()

    def rebuild(self, events: List[EventRecord]) -> None:
        self.rebuild_from_events(events)

    def clear(self) -> None:
        with self._lock:
            self.event_ids = []
            self._records = {}
            self.index = None
            if self.index_path.exists():
                self.index_path.unlink()
            if self.mapping_path.exists():
                self.mapping_path.unlink()

    def remove_events(self, event_ids: List[str]) -> None:
        removed = set(event_ids)
        if not removed:
            return
        retained = [
            event for event_id, event in getattr(self, "_records", {}).items()
            if event_id not in removed
        ]
        self.rebuild_from_events(retained)

    def search(
        self, query_text: str, top_k: int = 5, min_score: float = 0.10
    ) -> List[Tuple[str, float]]:
        if not query_text.strip() or self.index is None or not self.event_ids:
            return []
        with self._lock:
            query_vector = self._encode([query_text])
            scores, positions = self.index.search(
                query_vector, min(top_k, len(self.event_ids))
            )
        return [
            (self.event_ids[int(position)], float(score))
            for score, position in zip(scores[0], positions[0])
            if position >= 0 and float(score) >= min_score
        ]

    def save(self) -> None:
        if self.index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(self.index_path))
        self.mapping_path.write_text(
            json.dumps({
                "event_ids": self.event_ids,
                "model_name": self.model_name,
                "dimension": self.dimension,
            }),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.index_path.exists() or not self.mapping_path.exists():
            return
        try:
            self._load_dependencies()
            self.index = self._faiss.read_index(str(self.index_path))
            metadata = json.loads(self.mapping_path.read_text(encoding="utf-8"))
            if metadata.get("model_name") != self.model_name:
                logger.warning("Ignoring FAISS index built with a different embedding model")
                self.index = None
                self.event_ids = []
                return
            self.event_ids = metadata.get("event_ids", [])
            self.dimension = metadata.get("dimension")
            if self.index.ntotal != len(self.event_ids):
                raise ValueError(
                    f"FAISS index contains {self.index.ntotal} vectors but mapping contains "
                    f"{len(self.event_ids)} event IDs"
                )
            if self.dimension != self.index.d:
                raise ValueError(
                    f"FAISS index dimension {self.index.d} does not match mapping dimension "
                    f"{self.dimension}"
                )
            logger.info("Loaded FAISS event index with %s records", len(self.event_ids))
        except Exception as exc:
            logger.warning("Could not load FAISS index: %s", exc)
            self.index = None
            self.event_ids = []
            self.dimension = None


vector_store = VectorStore()
