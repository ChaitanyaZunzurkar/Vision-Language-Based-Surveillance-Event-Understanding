"""Tests for data schemas across modules."""

from server.src.detection.schemas import BoundingBox, Detection
from server.src.tracking.schemas import Track, TrackObservation
from server.src.event_understanding.schemas import CandidateEvent, SemanticEvent
from server.src.anomaly.schemas import ClipAnomalyScore


def test_bounding_box_geometry():
    box = BoundingBox(x1=10, y1=20, x2=50, y2=100)
    assert box.width == 40
    assert box.height == 80
    assert box.center_x == 30
    assert box.center_y == 60
    assert box.area == 3200


def test_track_properties():
    box1 = BoundingBox(x1=10, y1=20, x2=50, y2=100)
    box2 = BoundingBox(x1=30, y1=40, x2=70, y2=120)

    obs1 = TrackObservation(frame_idx=1, timestamp_sec=0.25, bbox=box1, confidence=0.85)
    obs2 = TrackObservation(frame_idx=2, timestamp_sec=0.50, bbox=box2, confidence=0.88)

    track = Track(
        track_id=1,
        class_name="Person",
        start_sec=0.25,
        end_sec=0.50,
        observations=[obs1, obs2],
    )

    assert track.duration_sec == 0.25
    assert track.displacement > 0.0
    assert track.median_height == 80.0


def test_candidate_and_semantic_event_schemas():
    cand = CandidateEvent(
        event_type="enter",
        start_sec=1.0,
        end_sec=2.5,
        entity_ids=[1],
        confidence=0.9,
    )
    assert cand.event_type == "enter"

    sem = SemanticEvent(
        event_id="ev_001",
        video_id="vid_test",
        event_type="loiter",
        start_sec=2.0,
        end_sec=7.0,
        entity_ids=[1],
        description="Person loiters in doorway",
        confidence=0.85,
    )
    assert sem.anomaly_category == "Normal"
