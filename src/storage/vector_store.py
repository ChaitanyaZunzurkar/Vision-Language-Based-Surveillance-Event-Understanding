"""Persistent FAISS semantic index for event descriptions.

SQLite owns event data and this index only stores searchable embeddings. The
optional lightweight fallback keeps local development usable when ML packages
are not installed; production deployments should install the FAISS and
sentence-transformers requirements.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import numpy as np
from src.utils.paths import paths
from src.utils.logger import logger
from src.storage.schemas import EventRecord


class VectorStore:
    def __init__(self, index_path: Optional[Path | str] = None):
        self.index_path = Path(index_path) if index_path else paths.vectors_dir / "events.faiss"
        self.mapping_path = self.index_path.with_suffix(".mapping.pkl")
        self.event_ids: List[str] = []
        self._descriptions: List[str] = []
        self.index = None
        self.encoder = None
        self._fallback_vectors = None
        self._load_dependencies()
        self.load()

    def _load_dependencies(self) -> None:
        try:
            import faiss
            self._faiss = faiss
            self.encoder = None
        except Exception as exc:
            self._faiss = None
            logger.warning("FAISS/sentence-transformers unavailable; semantic indexing is disabled: %s", exc)

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self._faiss is None:
            raise RuntimeError("Install faiss-cpu and sentence-transformers to build the event index")
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return np.asarray(self.encoder.encode(texts, normalize_embeddings=True), dtype="float32")

    def add_events(self, events: List[EventRecord]) -> None:
        if not events:
            return
        by_id = dict(zip(self.event_ids, self._descriptions))
        for event in events:
            by_id[event.event_id] = event.description
        self.event_ids = list(by_id)
        self._descriptions = [by_id[event_id] for event_id in self.event_ids]
        if self._faiss is None:
            return
        vectors = self._encode(self._descriptions)
        self.index = self._faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.save()

    def search(self, query_text: str, top_k: int = 5, min_score: float = 0.10,
               event_ids: Optional[set[str]] = None) -> List[Tuple[str, float]]:
        if self.index is None or not self.event_ids:
            return []
        query = self._encode([query_text])
        scores, positions = self.index.search(query, min(top_k * 4, len(self.event_ids)))
        results = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            event_id = self.event_ids[int(position)]
            if event_ids is None or event_id in event_ids:
                if float(score) >= min_score:
                    results.append((event_id, float(score)))
            if len(results) >= top_k:
                break
        return results

    def clear(self) -> None:
        self.event_ids = []
        self.corpus = []
        self.vectorizer = None
        self.tfidf_matrix = None
        if self.index_path.exists():
            self.index_path.unlink()

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
        if self.index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(self.index_path))
        with self.mapping_path.open("wb") as handle:
            pickle.dump({"event_ids": self.event_ids, "descriptions": self._descriptions}, handle)

    def load(self) -> None:
        if self._faiss is None or not self.index_path.exists() or not self.mapping_path.exists():
            return
        try:
            self.index = self._faiss.read_index(str(self.index_path))
            with self.mapping_path.open("rb") as handle:
                data = pickle.load(handle)
            self.event_ids = data.get("event_ids", [])
            self._descriptions = data.get("descriptions", [])
        except Exception as exc:
            logger.warning("Could not load FAISS event index: %s", exc)


vector_store = VectorStore()
