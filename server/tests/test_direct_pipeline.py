import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import shutil
from fastapi.testclient import TestClient
from server.src.anomaly.nn_predictor import SEGMENT_COUNT, _pool_features
from server.app.main import app
from server.app.api.deps import get_ingestor, get_metadata_store, get_pipeline
from server.src.detection.schemas import BoundingBox
from server.src.event_understanding.qwen import VLMEventUnderstanding
from server.src.event_understanding.schemas import TemporalWindow
from server.src.pipeline.json_input import (
    clip_event_times_to_video,
    validate_windows_payload,
)
from server.src.pipeline.pipeline import SurveillancePipeline
from server.src.config.loader import config_loader
from server.src.storage.metadata_store import MetadataStore
from server.src.storage.schemas import VideoRecord
from server.src.tracking.schemas import Track, TrackObservation, TrackingResult


def _make_video(path: Path, duration_sec: int = 4) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64)
    )
    for frame_index in range(duration_sec * 10):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + frame_index, 10), (25 + frame_index, 50), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _windows(video_id: str):
    return [{
        "video_id": video_id,
        "camera_id": video_id,
        "window": [0.0, 4.0],
        "tracks": [{
            "track_id": 3,
            "class": "Person",
            "dwell_sec": 2.0,
            "displacement_px": 20.0,
            "mean_speed": 10.0,
            "zones": ["center"],
            "near": [],
            "candidate_events": ["walk"],
        }],
        "events": [{
            "event_type": "walk",
            "subject_track": 3,
            "object_track": None,
            "start": 1.0,
            "end": 2.0,
            "abs_start": 1.0,
            "abs_end": 2.0,
            "zone": "center",
            "confidence": 0.9,
            "description": "Person #3 walks across the center of the frame.",
        }],
    }]


def test_windows_json_validation_checks_video_and_track_references():
    valid = _windows("sample")
    assert validate_windows_payload({"windows": valid}, {"sample"}, 4.0) == valid

    invalid = _windows("other")
    with pytest.raises(ValueError, match="do not match"):
        validate_windows_payload({"windows": invalid}, {"sample"}, 4.0)
    invalid = _windows("sample")
    invalid[0]["events"][0]["subject_track"] = 99
    with pytest.raises(ValueError, match="unknown subject_track"):
        validate_windows_payload({"windows": invalid}, {"sample"}, 4.0)


def test_precomputed_events_are_clipped_to_video_duration():
    payload = {"windows": [{
        "video_id": "sample",
        "window": [5.0, 15.0],
        "tracks": [],
        "events": [{
            "event_type": "standing",
            "subject_track": None,
            "object_track": None,
            "start": 0.0,
            "end": 10.0,
            "abs_start": 5.0,
            "abs_end": 15.0,
            "confidence": 0.8,
            "description": "Person is standing.",
        }],
    }]}

    clipped = clip_event_times_to_video(payload, 8.0)
    windows = validate_windows_payload(clipped, {"sample"}, 8.0)

    assert windows[0]["events"][0]["abs_end"] == 8.0
    assert windows[0]["events"][0]["end"] == 3.0
    assert payload["windows"][0]["events"][0]["abs_end"] == 15.0


def test_existing_notebook_events_are_normalized_without_model_loading(tmp_path):
    vlm = VLMEventUnderstanding()
    enriched, model_available = vlm.process_windows(
        "sample", _windows("sample"), tmp_path / "unused.mp4"
    )
    event = enriched[0]["events"][0]
    assert model_available is False
    assert (event["start"], event["end"]) == (1.0, 2.0)
    assert (event["abs_start"], event["abs_end"]) == (1.0, 2.0)
    assert event["vlm_verified"] is False


def test_qwen_event_output_is_track_grounded_and_gets_absolute_times():
    window = _windows("sample")[0]
    window.pop("events")
    events = VLMEventUnderstanding._validate_response(
        json.dumps({
            "events": [{
                "event_type": "walk",
                "subject_track": 3,
                "object_track": None,
                "start": 1.0,
                "end": 2.0,
                "confidence": 0.9,
                "description": "Person #3 walks through the frame.",
            }]
        }),
        window,
        0,
    )
    assert events[0]["abs_start"] == 1.0
    assert events[0]["abs_end"] == 2.0
    assert events[0]["vlm_verified"] is True

    with pytest.raises(ValueError, match="not present"):
        VLMEventUnderstanding._validate_response(
            '{"events":[{"event_type":"walk","subject_track":99,"start":0,"end":1,"description":"Walk"}]}',
            window,
            0,
        )


def test_frame_stride_respects_sample_rate_and_manual_override(tmp_path, monkeypatch):
    config = {"stage1b.frame_stride": 0, "video.sample_fps": 4.0}
    monkeypatch.setattr(config_loader, "get", lambda key, default=None: config.get(key, default))
    monkeypatch.setattr(
        "server.src.pipeline.pipeline.get_video_info", lambda _: {"fps": 29.97}
    )
    assert SurveillancePipeline._frame_stride(tmp_path / "source.mp4") == 8

    config["stage1b.frame_stride"] = 3
    assert SurveillancePipeline._frame_stride(tmp_path / "source.mp4") == 3


def test_window_track_proximity_uses_shared_frames_and_timestamps():
    def make_track(track_id, offset):
        return Track(
            track_id=track_id,
            class_name="Person",
            start_sec=0.0,
            end_sec=1.0,
            observations=[
                TrackObservation(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp,
                    bbox=BoundingBox(
                        x1=offset, y1=0, x2=offset + 20, y2=40
                    ),
                    confidence=0.9,
                )
                for frame_idx, timestamp in [(0, 0.0), (5, 0.5), (10, 1.0)]
            ],
        )

    serialized = SurveillancePipeline._serialize_windows(
        "sample",
        [TemporalWindow(window_id=1, start_sec=0.0, end_sec=1.0)],
        TrackingResult(
            video_id="sample",
            total_tracks=2,
            tracks=[make_track(1, 0), make_track(2, 10)],
        ),
        frame_width=100,
        frame_height=100,
    )

    assert serialized[0]["tracks"][0]["near"][0]["overlap_sec"] == 1.0
    assert serialized[0]["tracks"][1]["near"][0]["overlap_sec"] == 1.0


def test_direct_json_pipeline_skips_tracking_and_writes_analysis_json(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.mp4"
    _make_video(video_path)
    store = MetadataStore(db_path=tmp_path / "events.db")
    record = VideoRecord(
        video_id="video-test",
        filename=video_path.name,
        file_path=str(video_path),
        duration_sec=4.0,
        fps=10.0,
        frame_count=40,
        resolution="96x64",
    )
    store.save_video(record)

    class VectorStoreStub:
        def add_events(self, events):
            self.events = events

    pipeline = SurveillancePipeline(store=store, vec_store=VectorStoreStub())
    pipeline.clip_generator.output_dir = tmp_path / "clips"
    pipeline.clip_generator.cut_evidence_clip = lambda *args, **kwargs: (
        pytest.fail("Direct JSON mode must not re-encode clips for timestamped events")
    )
    windows = _windows("sample")
    windows[0]["events"][0]["anomaly_class"] = "Robbery"
    windows[0]["events"][0]["anomaly_confidence"] = 0.91

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Direct JSON mode must not run YOLO or ByteTrack")

    pipeline._run_stage1b = fail_if_called
    monkeypatch.setattr("server.src.pipeline.pipeline.paths.outputs_dir", tmp_path / "outputs")
    monkeypatch.setattr("server.src.pipeline.pipeline.paths.windows_dir", tmp_path / "windows")
    monkeypatch.setattr("server.src.pipeline.pipeline.paths.tracks_dir", tmp_path / "tracks")

    result = pipeline.process_video(
        "video-test",
        precomputed_windows=windows,
        pipeline_mode="direct_json",
        source_json_sha256="a" * 64,
    )

    output_path = Path(result["analysis_json"])
    assert output_path.is_file()
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["pipeline_mode"] == "direct_json"
    assert output["summary"]["total_windows"] == 1
    assert output["summary"]["tracking_backend"] == "precomputed_json"
    assert output["summary"]["total_anomalies"] == 1
    assert output["summary"]["total_anomaly_segments"] == 1
    assert output["source_json_sha256"] == "a" * 64
    assert output["windows"][0]["events"][0]["abs_start"] == 1.0
    assert output["windows"][0]["events"][0]["anomaly_category"] == "Robbery"
    assert output["windows"][0]["events"][0]["anomaly_prediction"]["source"] == "uploaded_json"
    assert store.get_events(video_id="video-test")[0].start_sec == 1.0
    assert store.get_events(video_id="video-test")[0].anomaly_category == "Robbery"
    assert store.get_events(video_id="video-test")[0].clip_path is None


def test_direct_json_anomaly_labels_are_used_instead_of_running_prediction():
    windows = _windows("sample")
    event = windows[0]["events"][0]
    event["anomaly_class"] = "Robbery"
    event["anomaly_confidence"] = 0.91

    summary = SurveillancePipeline._anomaly_summary_from_windows(
        "video-test", windows
    )

    assert summary.overall_category == "Robbery"
    assert summary.is_anomalous is True
    assert summary.anomaly_segments[0].confidence == 0.91
    assert summary.anomaly_segments[0].metadata["source"] == "uploaded_json"
    assert SurveillancePipeline._uploaded_anomaly_category(event) == "Robbery"


def test_direct_json_accepts_model_anomaly_prediction_fields():
    event = {
        "model_anomaly_class": "Burglary",
        "model_anomaly_confidence": 0.79,
        "model_p_anomaly": 0.42,
    }

    assert SurveillancePipeline._uploaded_anomaly_category(event) == "Burglary"
    assert SurveillancePipeline._uploaded_anomaly_confidence(event) == 0.79


def test_anomaly_feature_pooling_produces_notebook_temporal_shape():
    features = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    pooled = _pool_features(features, np.asarray([0.0, 1.0]), 2.0)

    assert pooled.shape == (SEGMENT_COUNT, 2)
    assert np.isfinite(pooled).all()
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0)


def test_direct_upload_api_validates_pair_and_starts_direct_mode(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.mp4"
    _make_video(video_path)
    saved_video_path = tmp_path / "stored" / video_path.name
    saved_video_path.parent.mkdir()

    class StoreStub:
        def __init__(self):
            self.status_updates = []

        def update_video_status(self, video_id, status):
            self.status_updates.append((video_id, status))

    class IngestorStub:
        def ingest_video(self, source_path, original_filename):
            shutil.copy2(source_path, saved_video_path)
            return VideoRecord(
                video_id="video-test",
                filename=original_filename,
                file_path=str(saved_video_path),
                duration_sec=4.0,
                fps=10.0,
                frame_count=40,
                resolution="96x64",
            )

    class PipelineStub:
        def __init__(self):
            self.calls = []

        def process_video(self, *args):
            self.calls.append(args)

    store = StoreStub()
    pipeline = PipelineStub()
    app.dependency_overrides[get_metadata_store] = lambda: store
    app.dependency_overrides[get_ingestor] = IngestorStub
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        response = TestClient(app).post(
            "/api/videos/upload-direct",
            files={
                "file": (video_path.name, video_path.read_bytes(), "video/mp4"),
                "json_file": (
                    "sample.json",
                    json.dumps({"windows": _windows("sample")}).encode(),
                    "application/json",
                ),
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["analysis_url"] == "/api/pipeline/output/video-test"
    assert response.json()["status"] == "processing"
    assert store.status_updates == [("video-test", "processing")]
    assert pipeline.calls[0][0] == "video-test"
    assert pipeline.calls[0][1][0]["video_id"] == "sample"
    assert pipeline.calls[0][2] == "direct_json"


def test_import_demo_api_registers_matching_sidecar_pairs(tmp_path, monkeypatch):
    uploads_dir = tmp_path / "uploads"
    repository_root = tmp_path / "project"
    reference_dir = repository_root / "notebooks" / "directjsonreference"
    video_dir = reference_dir / "video"
    output_dir = reference_dir / "pipeline_outputs"
    video_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    video_path = video_dir / "sample.mp4"
    _make_video(video_path)
    payload = {"windows": _windows("sample")}
    payload["windows"][0]["events"][0]["anomaly_class"] = "Robbery"
    payload["windows"][0]["events"][0]["anomaly_confidence"] = 0.91
    payload["windows"][0]["events"][0]["end"] = 10.0
    payload["windows"][0]["events"][0]["abs_end"] = 10.0
    stale_sidecar = {"windows": _windows("sample")}
    stale_sidecar["windows"][0]["events"][0]["anomaly_class"] = "Normal"
    (video_dir / "sample.json").write_text(
        json.dumps(stale_sidecar), encoding="utf-8"
    )
    (output_dir / "sample.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr("server.app.api.routes_videos.paths.uploads_dir", uploads_dir)
    monkeypatch.setattr("server.app.api.routes_videos.paths.root_dir", repository_root)
    monkeypatch.setattr("server.app.api.routes_videos.paths.outputs_dir", tmp_path / "outputs")

    class StoreStub:
        def __init__(self):
            self.videos = {}
            self.status_updates = []

        def get_video(self, video_id):
            return self.videos.get(video_id)

        def update_video_status(self, video_id, status):
            self.status_updates.append((video_id, status))
            self.videos[video_id].status = status

    class IngestorStub:
        def __init__(self):
            self.calls = []

        def ingest_video(self, source_path, original_filename):
            self.calls.append((source_path, original_filename))
            video = VideoRecord(
                video_id="video-test",
                filename=original_filename,
                file_path=str(source_path),
                duration_sec=4.0,
                fps=10.0,
                frame_count=40,
                resolution="96x64",
            )
            store.videos[video.video_id] = video
            return video

    class PipelineStub:
        def __init__(self):
            self.calls = []

        def process_video(self, *args):
            self.calls.append(args)

    store = StoreStub()
    ingestor = IngestorStub()
    pipeline = PipelineStub()
    app.dependency_overrides[get_metadata_store] = lambda: store
    app.dependency_overrides[get_ingestor] = lambda: ingestor
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        response = TestClient(app).post("/api/videos/import-demo")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert [video["filename"] for video in response.json()] == ["sample.mp4"]
    assert response.json()[0]["status"] == "processing"
    assert store.status_updates == [("video-test", "processing")]
    assert len(ingestor.calls) == 1
    assert pipeline.calls[0][0] == "video-test"
    assert pipeline.calls[0][1][0]["video_id"] == "sample"
    assert pipeline.calls[0][1][0]["events"][0]["abs_end"] == 4.0
    assert pipeline.calls[0][2] == "direct_json"
    assert pipeline.calls[0][1][0]["events"][0]["anomaly_class"] == "Robbery"
    assert pipeline.calls[0][3] == hashlib.sha256(
        (output_dir / "sample.json").read_bytes()
    ).hexdigest()

    app.dependency_overrides[get_metadata_store] = lambda: store
    app.dependency_overrides[get_ingestor] = lambda: ingestor
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        store.videos["video-test"].status = "processing"
        response = TestClient(app).post("/api/videos/import-demo")
        assert response.status_code == 200
        assert response.json()[0]["status"] == "processing"
        assert len(pipeline.calls) == 2

        analysis_path = tmp_path / "outputs" / "video-test_analysis.json"
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(
            json.dumps({
                "source_json_sha256": hashlib.sha256(
                    (output_dir / "sample.json").read_bytes()
                ).hexdigest()
            }),
            encoding="utf-8",
        )
        response = TestClient(app).post("/api/videos/import-demo")
        assert response.status_code == 200
        assert response.json()[0]["status"] == "completed"
        assert store.videos["video-test"].status == "completed"
        assert len(pipeline.calls) == 2
    finally:
        app.dependency_overrides.clear()


def test_recover_staged_api_requeues_interrupted_processing_video(tmp_path, monkeypatch):
    repository_root = tmp_path / "project"
    video_dir = tmp_path / "uploads"
    video_dir.mkdir()
    output_dir = repository_root / "notebooks" / "directjsonreference" / "pipeline_outputs"
    output_dir.mkdir(parents=True)
    video_path = video_dir / "vid_123456789abc_sample.mp4"
    _make_video(video_path)
    json_path = output_dir / "sample.json"
    json_path.write_text(
        json.dumps({"windows": _windows("sample")}), encoding="utf-8"
    )
    monkeypatch.setattr("server.app.api.routes_videos.paths.root_dir", repository_root)
    monkeypatch.setattr(
        "server.app.api.routes_videos.paths.outputs_dir", tmp_path / "outputs"
    )

    video = VideoRecord(
        video_id="video-test",
        filename="sample.mp4",
        file_path=str(video_path),
        duration_sec=4.0,
        fps=10.0,
        frame_count=40,
        resolution="96x64",
        status="processing",
    )

    class StoreStub:
        def __init__(self):
            self.status_updates = []

        def list_videos(self):
            return [video]

        def update_video_status(self, video_id, status, error_message=None):
            self.status_updates.append((video_id, status, error_message))
            video.status = status
            video.error_message = error_message

    class PipelineStub:
        def __init__(self):
            self.calls = []

        def process_video(self, *args):
            self.calls.append(args)

    store = StoreStub()
    pipeline = PipelineStub()
    app.dependency_overrides[get_metadata_store] = lambda: store
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        response = TestClient(app).post("/api/videos/recover-staged")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()[0]["status"] == "processing"
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0][0] == "video-test"
    assert pipeline.calls[0][1][0]["video_id"] == "sample"
    assert pipeline.calls[0][2] == "direct_json"
    assert pipeline.calls[0][3] == hashlib.sha256(json_path.read_bytes()).hexdigest()


def test_recover_staged_api_ignores_unpaired_video_processing(tmp_path, monkeypatch):
    video_path = tmp_path / "ordinary-upload.mp4"
    _make_video(video_path)
    monkeypatch.setattr(
        "server.app.api.routes_videos.paths.root_dir", tmp_path / "project"
    )
    monkeypatch.setattr(
        "server.app.api.routes_videos.paths.outputs_dir", tmp_path / "outputs"
    )
    video = VideoRecord(
        video_id="ordinary-video",
        filename=video_path.name,
        file_path=str(video_path),
        duration_sec=4.0,
        fps=10.0,
        frame_count=40,
        resolution="96x64",
        status="processing",
    )

    class StoreStub:
        def list_videos(self):
            return [video]

        def update_video_status(self, *args, **kwargs):
            pytest.fail("Unpaired uploads must not be changed by staged recovery")

    app.dependency_overrides[get_metadata_store] = lambda: StoreStub()
    try:
        response = TestClient(app).post("/api/videos/recover-staged")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == []
    assert video.status == "processing"
