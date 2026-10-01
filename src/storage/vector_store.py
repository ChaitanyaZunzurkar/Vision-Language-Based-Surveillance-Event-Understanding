"""Vector index for semantic surveillance event descriptions."""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.paths import paths
from src.utils.logger import logger
from src.storage.schemas import EventRecord


class VectorStore:
    """Stores semantic representations of events and provides cosine similarity search."""

    def __init__(self, index_path: Optional[Path | str] = None):
        self.index_path = (
            Path(index_path) if index_path else (paths.vectors_dir / "event_vectors.pkl")
        )
        self.event_ids: List[str] = []
        self.corpus: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.load()

    def add_events(self, events: List[EventRecord]) -> None:
        """Add new events to the vector index and recompute embeddings."""
        if not events:
            return

        new_events_map = {ev.event_id: ev.description for ev in events}

        # Update corpus
        for ev_id, desc in new_events_map.items():
            if ev_id in self.event_ids:
                idx = self.event_ids.index(ev_id)
                self.corpus[idx] = desc
            else:
                self.event_ids.append(ev_id)
                self.corpus.append(desc)

        # Re-fit TF-IDF vectorizer
        if self.corpus:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                lowercase=True,
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
            self.save()
            logger.info(
                f"Updated VectorStore with {len(events)} events (total indexed: {len(self.event_ids)})"
            )

    def search(
        self, query_text: str, top_k: int = 5, min_score: float = 0.10
    ) -> List[Tuple[str, float]]:
        """Perform semantic cosine similarity search for a natural-language query.

        Args:
            query_text: Natural language user query.
            top_k: Max results to return.
            min_score: Minimum cosine similarity threshold.

        Returns:
            List of (event_id, similarity_score) tuples, sorted descending by score.
        """
        if not self.vectorizer or self.tfidf_matrix is None or not self.corpus:
            return []

        try:
            query_vec = self.vectorizer.transform([query_text])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            results: List[Tuple[str, float]] = []
            for idx, score in enumerate(similarities):
                if score >= min_score:
                    results.append((self.event_ids[idx], float(score)))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def save(self) -> None:
        """Persist vector index to disk."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "event_ids": self.event_ids,
                "corpus": self.corpus,
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
            }
            with open(self.index_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save vector store to {self.index_path}: {e}")

    def load(self) -> None:
        """Load vector index from disk if available."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.event_ids = data.get("event_ids", [])
                    self.corpus = data.get("corpus", [])
                    self.vectorizer = data.get("vectorizer")
                    self.tfidf_matrix = data.get("tfidf_matrix")
                logger.info(
                    f"Loaded VectorStore index with {len(self.event_ids)} records from {self.index_path}"
                )
            except Exception as e:
                logger.warning(f"Could not load vector store from {self.index_path}: {e}")


# Global vector store instance
vector_store = VectorStore()
