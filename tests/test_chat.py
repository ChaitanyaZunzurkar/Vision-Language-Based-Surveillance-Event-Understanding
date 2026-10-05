from src.storage.metadata_store import MetadataStore
from src.storage.schemas import EventRecord


def test_conversation_and_evidence_persist(tmp_path):
    store = MetadataStore(tmp_path / "chat.db")
    store.create_conversation("chat-1", "Suspicious activity", "video-1")
    store.add_message("m-1", "chat-1", "user", "Show activity")
    store.add_message("m-2", "chat-1", "assistant", "Verified event.", ["event-1"])
    assert [m["role"] for m in store.list_messages("chat-1")] == ["user", "assistant"]
    assert store.get_message("m-2")["evidence_ids"] == ["event-1"]
    store2 = MetadataStore(tmp_path / "chat.db")
    assert store2.get_conversation("chat-1")["title"] == "Suspicious activity"
    assert store2.list_messages("chat-1")[1]["evidence_ids"] == ["event-1"]


def test_event_filters_are_sqlite_grounded(tmp_path):
    store = MetadataStore(tmp_path / "events.db")
    store.insert_events([
        EventRecord(event_id="e1", video_id="v1", event_type="loiter", start_sec=2, end_sec=5,
                    description="Person loiters", confidence=.9, anomaly_category="Suspicious"),
        EventRecord(event_id="e2", video_id="v2", event_type="enter", start_sec=2, end_sec=5,
                    description="Person enters", confidence=.9),
    ])
    assert [e.event_id for e in store.get_events(video_id="v1", anomaly_category="Suspicious")] == ["e1"]
