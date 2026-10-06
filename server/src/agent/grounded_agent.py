"""Qwen-backed conversational agent with a deterministic grounded fallback."""

from typing import Any, Dict, List, Optional
import os
from server.src.retrieval.retriever import SurveillanceRetriever, retriever
from server.src.storage.metadata_store import MetadataStore, metadata_store


class GroundedAgent:
    """Retrieval-first agent. The optional Qwen model receives only database evidence."""

    def __init__(self, store: Optional[MetadataStore] = None, searcher: Optional[SurveillanceRetriever] = None):
        self.store = store or metadata_store
        self.searcher = searcher or retriever
        self.model_name = "Qwen/Qwen3-8B"
        self.model = None

    def search_events(self, query: str, video_id: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        result = self.searcher.query(query, top_k=top_k, min_score=0.0, video_id=video_id)
        return result.model_dump()

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        event = self.store.get_event(event_id)
        return event.model_dump() if event else None

    def search_anomalies(self, video_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.store.search_anomalies(video_id)

    def search_objects(self, video_id: Optional[str] = None, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.store.search_tracks(video_id, class_name)

    def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        video = self.store.get_video(video_id)
        return video.model_dump() if video else None

    def get_event_context(self, event_id: str) -> Dict[str, Any]:
        event = self.get_event(event_id)
        if not event:
            return {}
        return {"event": event, "video": self.get_video_info(event["video_id"]), "objects": self.search_objects(event["video_id"])}

    def get_evidence(self, event_id: str) -> Dict[str, Any]:
        event = self.get_event(event_id)
        return {"event_id": event_id, "clip_path": event.get("clip_path") if event else None}

    def answer(self, question: str, video_id: Optional[str] = None, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        result = self.search_events(question, video_id=video_id)
        matches = result["matched_events"]
        if not matches:
            return {"content": "I could not find a verified event matching that question in the available processed data.", "evidence": []}
        if os.getenv("VISTA_ENABLE_QWEN", "").lower() in {"1", "true", "yes"}:
            generated = self._qwen_answer(question, matches, history or [])
            if generated:
                return {"content": generated, "evidence": matches}
        lines = []
        for event in matches:
            lines.append(
                f"{event['description']} ({event['start_sec']:.1f}sâ€“{event['end_sec']:.1f}s; "
                f"confidence {event['confidence']:.2f}; anomaly {event['anomaly_category']})."
            )
        return {
            "content": "Based on the processed video evidence:\n" + "\n".join(lines),
            "evidence": matches,
        }

    def _qwen_answer(self, question: str, evidence: List[Dict[str, Any]],
                     history: List[Dict[str, Any]]) -> Optional[str]:
        """Generate from a compact evidence-only prompt when Qwen is explicitly enabled."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if self.model is None:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto",
                )
                self.model = (tokenizer, model)
            tokenizer, model = self.model
            facts = "\n".join(
                f"- {item['event_id']}: {item['description']} [{item['start_sec']}-{item['end_sec']}s]"
                for item in evidence
            )
            prompt = (
                "Answer only from VERIFIED FACTS. If the facts do not answer the question, say so. "
                "Do not invent timestamps, objects, intent, or events.\n"
                f"Question: {question}\nVERIFIED FACTS:\n{facts}\nAnswer:"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception:
            return None


grounded_agent = GroundedAgent()


