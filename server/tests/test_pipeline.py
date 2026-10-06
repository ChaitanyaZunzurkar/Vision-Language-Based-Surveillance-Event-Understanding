"""End-to-end integration test for the full surveillance understanding pipeline."""

import tempfile
from pathlib import Path
import cv2
import numpy as np
from server.src.pipeline.ingestion import VideoIngestor
from server.src.pipeline.pipeline import SurveillancePipeline
from server.src.storage.metadata_store import MetadataStore
from server.src.storage.vector_store import VectorStore
from server.src.retrieval.retriever import SurveillanceRetriever


def generate_test_video(path: Path):
    """Generate a test surveillance video with simulated walking entity."""
    width, height = 320, 240
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    # 40 frames = 4 seconds
    for i in range(40):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Person box moving from left to right
        x = int(30 + i * 5)
        cv2.rectangle(frame, (x, 60), (x + 25, 160), (200, 200, 200), -1)
        out.write(frame)

    out.release()


def test_full_pipeline_and_retrieval():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_file = tmp_path / "test_events.db"
        vec_file = tmp_path / "test_vectors.pkl"
        uploads_dir = tmp_path / "uploads"
        clips_dir = tmp_path / "clips"

        store = MetadataStore(db_path=db_file)
        vec_store = VectorStore(index_path=vec_file)
        ingestor = VideoIngestor(uploads_dir=uploads_dir, store=store)

        # 1. Ingest Video
        raw_video = tmp_path / "surveillance_sample.mp4"
        generate_test_video(raw_video)

        video_record = ingestor.ingest_video(raw_video, original_filename="surveillance_sample.mp4")
        assert video_record.status == "pending"
        assert video_record.duration_sec >= 3.5

        # 2. Run Pipeline
        pipeline = SurveillancePipeline(store=store, vec_store=vec_store)
        pipeline.clip_generator.output_dir = clips_dir

        summary = pipeline.process_video(video_record.video_id)
        assert summary["status"] == "completed"
        assert summary["total_tracks"] >= 1

        # Check video record updated
        updated_video = store.get_video(video_record.video_id)
        assert updated_video.status == "completed"

        # Check events in store
        events = store.get_events(video_id=video_record.video_id)
        assert len(events) > 0

        # Check clips exist
        for ev in events:
            if ev.clip_path:
                assert Path(ev.clip_path).exists()

        # 3. Test Retrieval
        retriever = SurveillanceRetriever(meta_store=store, vec_store=vec_store)
        query_result = retriever.query("person entering surveillance scene", top_k=3)
        assert query_result.total_matches > 0
        assert len(query_result.matched_events) > 0
        assert "Retrieved" in query_result.grounded_summary


