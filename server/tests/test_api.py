"""Tests for FastAPI backend endpoints."""

import io
from types import SimpleNamespace
from fastapi.testclient import TestClient
from server.app.api.deps import get_pipeline, get_retriever
from server.app.main import app
from server.src.retrieval.retriever import QueryResult

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_videos_list():
    response = client.get("/api/videos")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_events_list():
    response = client.get("/api/events")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_stats_dashboard():
    response = client.get("/api/stats/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "total_videos" in data
    assert "total_events" in data


def test_natural_language_query():
    response = client.post(
        "/api/query",
        json={"query": "Show clips where a person entered or loitered", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert "grounded_summary" in data
    assert "matched_events" in data
    assert "parsed_filters" in data


def test_natural_language_query_uses_conversation_history_for_local_answer():
    evidence = [{
        "event_id": "event-1",
        "video_id": "video-1",
        "event_type": "exit",
        "start_sec": 12.0,
        "end_sec": 14.0,
        "entity_ids": [3],
        "description": "Person #3 exits through the left side.",
        "confidence": 0.9,
        "relevance_score": 0.8,
        "anomaly_category": "Normal",
        "clip_path": None,
    }]
    retrieval_queries = []
    retriever = SimpleNamespace(query=lambda **kwargs: (
        retrieval_queries.append(kwargs["query"]) or QueryResult(
            query=kwargs["query"],
            parsed_filters={},
            total_matches=1,
            matched_events=evidence,
            grounded_summary="Deterministic evidence summary.",
        )
    ))
    calls = []
    vlm = SimpleNamespace(generate_grounded_answer=lambda *args: (
        calls.append(args) or "They exited at 12–14 seconds [event-1]."
    ))
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_pipeline] = lambda: SimpleNamespace(vlm=vlm)
    try:
        response = client.post("/api/query", json={
            "query": "when did they leave?",
            "history": [
                {"role": "user", "content": "What happened to the person?"},
                {"role": "assistant", "content": "A person was seen in the video."},
            ],
        })
    finally:
        app.dependency_overrides.pop(get_retriever, None)
        app.dependency_overrides.pop(get_pipeline, None)

    assert response.status_code == 200
    assert response.json()["grounded_summary"] == "They exited at 12–14 seconds [event-1]."
    assert calls[0][0] == "when did they leave?"
    assert calls[0][2][0]["content"] == "What happened to the person?"
    assert retrieval_queries == [
        "What happened to the person?\nFollow-up: when did they leave?"
    ]


def test_frontend_ui():
    # Test /ui endpoint
    response = client.get("/ui/")
    assert response.status_code == 200
    assert "SurveillanceLens" in response.text

    # Test browser navigation to /
    response_html = client.get("/", headers={"accept": "text/html"})
    assert response_html.status_code == 200
    assert "SurveillanceLens" in response_html.text
