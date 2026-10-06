"""Tests for FastAPI backend endpoints."""

import io
from fastapi.testclient import TestClient
from server.app.main import app

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


def test_frontend_ui():
    # Test /ui endpoint
    response = client.get("/ui/")
    assert response.status_code == 200
    assert "SurveillanceLens" in response.text

    # Test browser navigation to /
    response_html = client.get("/", headers={"accept": "text/html"})
    assert response_html.status_code == 200
    assert "SurveillanceLens" in response_html.text

