"""End-to-end master pipeline orchestrator for surveillance video understanding."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import importlib.util
import math
import os
import tempfile
from threading import Lock
import uuid
import yaml
from time import perf_counter
from server.src.utils.paths import paths
from server.src.utils.logger import logger
from server.src.storage.metadata_store import MetadataStore, metadata_store
from server.src.storage.vector_store import VectorStore, vector_store
from server.src.tracking.schemas import Track, TrackObservation, TrackingResult
from server.src.tracking.tracker import SurveillanceTracker as LocalByteTrack
from server.src.detection.detector import SurveillanceDetector
from server.src.detection.schemas import BoundingBox
from server.src.event_understanding.event_generator import TemporalWindower
from server.src.config.loader import config_loader
from server.src.anomaly.detector import CATEGORIES as ANOMALY_CATEGORIES
from server.src.anomaly.nn_predictor import TemporalAnomalyPredictor
from server.src.anomaly.schemas import ClipAnomalyScore, VideoAnomalySummary
from server.src.event_understanding.qwen import VLMEventUnderstanding
from server.src.storage.schemas import EventRecord, AnomalyRecord, TrackRecord
from server.src.evidence.clip_generator import ClipGenerator
from server.src.utils.timing import StageTimings
from server.src.utils.device import log_device_diagnostics
from server.src.utils.video_utils import get_video_info
from server.src.pipeline.json_input import matching_video_names, validate_windows_payload


class SurveillancePipeline:
    """Master orchestrator executing the full surveillance understanding pipeline on a video."""

    def __init__(
        self,
        store: Optional[MetadataStore] = None,
        vec_store: Optional[VectorStore] = None,
    ):
        self.store = store or metadata_store
        self.vec_store = vec_store or vector_store
        self._stage1b_lock = Lock()
        self._stage1b_tracker = None
        self._stage1b_tracker_signature = None

        # Pipeline submodules
        self.windower = TemporalWindower(
            window_duration_sec=config_loader.get("temporal.window_duration_sec", 10.0),
            window_stride_sec=config_loader.get("temporal.window_stride_sec", 5.0),
            loiter_dwell_sec=config_loader.get("temporal.loiter_dwell_sec", 4.0),
        )
        self.anomaly_detector = TemporalAnomalyPredictor(
            model_path=paths.root_dir / config_loader.get(
                "anomaly.nn_model_weights", "models/anomaly/nn_models_fixed.pt"
            ),
            feature_model=config_loader.get(
                "anomaly.feature_model", "openai/clip-vit-base-patch16"
            ),
            sample_fps=float(config_loader.get("anomaly.sample_fps", 2.0)),
            batch_size=int(config_loader.get("anomaly.batch_size", 16)),
            anomaly_threshold=float(config_loader.get("anomaly.anomaly_threshold", 0.5)),
            device=config_loader.get("anomaly.device", "auto"),
        )
        self.vlm = VLMEventUnderstanding(
            model_name=config_loader.get("vlm.model_name", "Qwen/Qwen2.5-VL-3B-Instruct"),
            device=config_loader.get("vlm.device", "auto"),
            temperature=float(config_loader.get("vlm.temperature", 0.2)),
        )
        self.clip_generator = ClipGenerator(
            padding_before_sec=float(config_loader.get("evidence.padding_before_sec", 1.5)),
            padding_after_sec=float(config_loader.get("evidence.padding_after_sec", 1.5)),
        )

    def _run_stage1b(self, video_path: Path, video_id: str) -> TrackingResult:
        """Run the project's actual VISTA YOLO+ByteTrack implementation."""
        tracker_script = (paths.root_dir / config_loader.get(
            "stage1b.tracker_script", "server/src/tracking/surveillance_tracker.py"
        )).resolve()
        weights_path = (paths.root_dir / config_loader.get(
            "stage1b.weights_path", "models/stage1b/yolo11x_surv_v1.pt"
        )).resolve()
        bytetrack_yaml = (paths.root_dir / config_loader.get(
            "stage1b.bytetrack_yaml", "server/configs/bytetrack_custom.yaml"
        )).resolve()
        stage1_config = (paths.root_dir / config_loader.get(
            "stage1b.class_config", "server/configs/stage1_model.yaml"
        )).resolve()
        required = [tracker_script, weights_path, bytetrack_yaml, stage1_config]
        missing = [str(path) for path in required if not path.exists()]
        if weights_path.exists() and weights_path.stat().st_size < 1024:
            missing.append(f"{weights_path} (Git LFS pointer; model weights are not hydrated)")
        if importlib.util.find_spec("ultralytics") is None:
            missing.append("ultralytics package")
        if missing:
            if config_loader.get("stage1b.allow_detector_fallback", True):
                logger.warning(
                    "Stage 1b resources are missing; using the configured lightweight "
                    "detector and local ByteTrack fallback: " + ", ".join(missing)
                )
                return self._run_detector_fallback(video_path, video_id)
            raise FileNotFoundError("Stage 1b resources are missing: " + ", ".join(missing))

        with stage1_config.open("r", encoding="utf-8") as config_file:
            model_config = yaml.safe_load(config_file) or {}
        names = model_config.get("names", {})
        class_names = {int(key): str(value) for key, value in names.items()}
        if not class_names:
            raise ValueError(f"No class names are defined in {stage1_config}")

        spec = importlib.util.spec_from_file_location("vista_surveillance_tracker", tracker_script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Stage 1b tracker from {tracker_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frame_stride = self._frame_stride(video_path)
        tracker_signature = (
            str(weights_path),
            tuple(sorted(class_names.items())),
            float(config_loader.get("stage1b.track_conf_threshold", 0.10)),
            int(model_config.get("imgsz", 640)),
            config_loader.get("runtime.device", "auto"),
        )
        tracker_config = yaml.safe_load(bytetrack_yaml.read_text(encoding="utf-8")) or {}
        original_buffer = int(tracker_config.get("track_buffer", 30))
        tracker_config["track_buffer"] = max(1, math.ceil(original_buffer / frame_stride))
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=f"{video_id}_bytetrack_",
            dir=paths.outputs_dir,
            delete=False,
            encoding="utf-8",
        ) as tracker_config_file:
            yaml.safe_dump(tracker_config, tracker_config_file, sort_keys=False)
            tracker_config_path = Path(tracker_config_file.name)
        try:
            with self._stage1b_lock:
                if (
                    self._stage1b_tracker is None
                    or self._stage1b_tracker_signature != tracker_signature
                ):
                    self._stage1b_tracker = module.SurveillanceTracker(
                        weights_path=str(weights_path),
                        class_names=class_names,
                        bytetrack_yaml=str(tracker_config_path),
                        detect_conf=tracker_signature[2],
                        imgsz=tracker_signature[3],
                        frame_stride=frame_stride,
                        device=tracker_signature[4],
                    )
                    self._stage1b_tracker_signature = tracker_signature
                tracker = self._stage1b_tracker
                tracker.frame_stride = frame_stride
                tracker.bytetrack_yaml = str(tracker_config_path)
                tracker.model.predictor = None
                tracks_df = tracker.process_video(
                    str(video_path), video_id=video_id, camera_id=video_id
                )
        finally:
            tracker_config_path.unlink(missing_ok=True)
        tracking_backend = "yolo_bytetrack"
        if tracks_df.empty and config_loader.get("stage1b.allow_detector_fallback", True):
            tracking_backend = "opencv_motion_byte_track"
            logger.warning(
                "Stage 1b returned no tracks; using the existing OpenCV-compatible "
                "detector with the local ByteTrack association implementation."
            )
            detector = SurveillanceDetector(
                # The fallback is intentionally independent of YOLO so that a
                # detector checkpoint with no matching classes cannot suppress
                # the existing lightweight motion path.
                weights_path=None,
                conf_threshold=float(config_loader.get("detection.conf_threshold", 0.20)),
                device=config_loader.get("detection.device", "auto"),
            )
            fallback_result = LocalByteTrack(
                track_high_thresh=float(config_loader.get("tracking.track_high_thresh", 0.50)),
                track_low_thresh=float(config_loader.get("tracking.track_low_thresh", 0.20)),
                new_track_thresh=float(config_loader.get("tracking.new_track_thresh", 0.60)),
                max_lost_frames=int(config_loader.get("tracking.track_buffer", 30)),
                match_thresh=min(float(config_loader.get("tracking.match_thresh", 0.30)), 0.30),
            ).track_video(detector.detect_video(video_path, video_id))
            tracks = []
            for track in fallback_result.tracks:
                for observation in track.observations:
                    tracks.append({
                        "video_id": video_id,
                        "frame_id": observation.frame_idx,
                        "timestamp_sec": observation.timestamp_sec,
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                        "x1": observation.bbox.x1,
                        "y1": observation.bbox.y1,
                        "x2": observation.bbox.x2,
                        "y2": observation.bbox.y2,
                        "confidence": observation.confidence,
                    })
            import pandas as pd
            tracks_df = pd.DataFrame(tracks)
        tracks_path = paths.tracks_dir / f"{video_id}_tracks.csv"
        tracks_df.to_csv(tracks_path, index=False)

        tracks = []
        if not tracks_df.empty:
            for track_id, group in tracks_df.groupby("track_id", sort=True):
                observations = [TrackObservation(
                    frame_idx=int(row.frame_id),
                    timestamp_sec=float(row.timestamp_sec),
                    bbox=BoundingBox(x1=float(row.x1), y1=float(row.y1), x2=float(row.x2), y2=float(row.y2)),
                    confidence=float(row.confidence),
                ) for row in group.itertuples(index=False)]
                tracks.append(Track(
                    track_id=int(track_id),
                    class_name=str(group.iloc[0].class_name),
                    start_sec=observations[0].timestamp_sec,
                    end_sec=observations[-1].timestamp_sec,
                    observations=observations,
                ))
        if not tracks and config_loader.get("stage1b.allow_detector_fallback", True):
            return self._run_detector_fallback(video_path, video_id)
        return TrackingResult(
            video_id=video_id,
            total_tracks=len(tracks),
            backend=tracking_backend,
            tracks=tracks,
        )

    @staticmethod
    def _frame_stride(video_path: Path) -> int:
        """Resolve a configured source-frame stride or derive one from sample_fps."""
        configured = int(config_loader.get("stage1b.frame_stride", 0))
        if configured > 0:
            return configured
        video_fps = get_video_info(video_path)["fps"]
        target_fps = float(config_loader.get("video.sample_fps", 4.0))
        if target_fps <= 0:
            raise ValueError("video.sample_fps must be greater than zero")
        return max(1, math.ceil(video_fps / target_fps))

    def _run_detector_fallback(self, video_path: Path, video_id: str) -> TrackingResult:
        """Use the repository's lightweight detector when Stage 1b emits no usable tracks."""
        detector = SurveillanceDetector(
            weights_path=None,
            conf_threshold=float(config_loader.get("detection.conf_threshold", 0.20)),
            device=config_loader.get("detection.device", "auto"),
        )
        result = LocalByteTrack(
            track_high_thresh=float(config_loader.get("tracking.track_high_thresh", 0.50)),
            track_low_thresh=float(config_loader.get("tracking.track_low_thresh", 0.20)),
            new_track_thresh=float(config_loader.get("tracking.new_track_thresh", 0.60)),
            max_lost_frames=int(config_loader.get("tracking.track_buffer", 30)),
            match_thresh=min(float(config_loader.get("tracking.match_thresh", 0.30)), 0.30),
        ).track_video(detector.detect_video(video_path, video_id))
        rows = []
        for track in result.tracks:
            for observation in track.observations:
                rows.append({
                    "video_id": video_id,
                    "frame_id": observation.frame_idx,
                    "timestamp_sec": observation.timestamp_sec,
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "x1": observation.bbox.x1,
                    "y1": observation.bbox.y1,
                    "x2": observation.bbox.x2,
                    "y2": observation.bbox.y2,
                    "confidence": observation.confidence,
                })
        import pandas as pd
        pd.DataFrame(rows).to_csv(paths.tracks_dir / f"{video_id}_tracks.csv", index=False)
        result.backend = "opencv_motion_byte_track"
        return result

    def process_video(
        self,
        video_id: str,
        precomputed_windows: Optional[list[Dict[str, Any]]] = None,
        pipeline_mode: str = "full",
        source_json_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the video-only pipeline or continue from notebook temporal-window JSON."""
        video = self.store.get_video(video_id)
        if not video:
            raise ValueError(f"Video {video_id} not found in metadata store")

        video_path = Path(video.file_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file missing on disk: {video_path}")

        logger.info(f"=== Starting Surveillance Pipeline for {video_id} ({video.filename}) ===")
        processing_started = perf_counter()
        log_device_diagnostics()
        timings = StageTimings()
        self.store.update_video_status(video_id, "processing")

        try:
            video_info = get_video_info(video_path)
            tracking_result = None
            tracks_csv_path = paths.tracks_dir / f"{video_id}_tracks.csv"
            windows_json_path = paths.windows_dir / f"{video_id}_windows.json"
            if precomputed_windows is None:
                logger.info("Stage 1: Running sampled YOLO + ByteTrack...")
                with timings.measure("detection_tracking"):
                    tracking_result = self._run_stage1b(video_path, video_id)
                self.store.clear_video_tracks(video_id)
                self.store.insert_tracks([
                    TrackRecord(
                        video_id=video_id,
                        track_id=track.track_id,
                        class_name=track.class_name,
                        start_sec=track.start_sec,
                        end_sec=track.end_sec,
                        total_observations=len(track.observations),
                    )
                    for track in tracking_result.tracks
                ])
                with timings.measure("temporal_events"):
                    temporal_windows = self.windower.generate_windows(
                        tracking_result, video.duration_sec
                    )
                windows_data = self._serialize_windows(
                    video_id, temporal_windows, tracking_result,
                    video_info["width"], video_info["height"],
                )
                self._write_json(windows_json_path, {"windows": windows_data})
            else:
                logger.info("Direct JSON mode: validating supplied temporal windows...")
                self.store.clear_video_tracks(video_id)
                with timings.measure("temporal_events"):
                    windows_data = validate_windows_payload(
                        {"windows": precomputed_windows},
                        matching_video_names(video.filename, video_id),
                        video.duration_sec,
                    )
                for window in windows_data:
                    window["video_id"] = video_id
                    window.setdefault("camera_id", video_id)
                    self._inherit_window_anomaly_label(window)
                self._write_json(windows_json_path, {"windows": windows_data})

            logger.info("Stage 3: Running Qwen2.5-VL event understanding...")
            with timings.measure("vlm"):
                windows_data, vlm_model_available = self.vlm.process_windows(
                    video_id, windows_data, video_path
                )
            if pipeline_mode == "direct_json":
                for window in windows_data:
                    self._inherit_window_anomaly_label(window)

            if pipeline_mode == "direct_json":
                logger.info("Stage 4: Using anomaly classes supplied in the JSON...")
                anomaly_summary = self._anomaly_summary_from_windows(
                    video_id, windows_data
                )
            else:
                logger.info("Stage 4: Running the binary + 13-class anomaly network...")
                with timings.measure("anomaly"):
                    anomaly_summary = self.anomaly_detector.predict_video(
                        video_path, video_id, video.duration_sec
                    )
            previous_event_ids = self.store.clear_video_analysis(video_id)
            if previous_event_ids:
                self.vec_store.rebuild(self.store.get_events())
            self.store.insert_anomalies([
                AnomalyRecord(
                    anomaly_id=segment.clip_id,
                    video_id=video_id,
                    category=segment.category,
                    confidence=segment.confidence,
                    is_anomaly=segment.is_anomaly,
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    metadata=segment.metadata,
                )
                for segment in anomaly_summary.anomaly_segments
            ])

            records = []
            event_index = 0
            for window_index, window in enumerate(windows_data):
                for event in window.get("events", []):
                    start_sec = float(event.get("abs_start", window["window"][0] + event["start"]))
                    end_sec = float(event.get("abs_end", window["window"][0] + event["end"]))
                    overlapping = [
                        segment for segment in anomaly_summary.anomaly_segments
                        if segment.end_sec >= start_sec and segment.start_sec <= end_sec
                    ]
                    anomaly = max(
                        overlapping,
                        key=lambda item: item.metadata.get(
                            "anomaly_probability", item.confidence
                        ),
                        default=None,
                    )
                    supplied_category = self._uploaded_anomaly_category(event)
                    if supplied_category:
                        anomaly_category = supplied_category
                        anomaly_confidence = self._uploaded_anomaly_confidence(event)
                    else:
                        anomaly_category = anomaly.category if anomaly else "Unassessed"
                        anomaly_confidence = anomaly.confidence if anomaly else 0.0
                    event_id = event.get("event_id") or (
                        "ev_" + uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{video_id}:{window_index}:{event_index}:{event['event_type']}:{start_sec}",
                        ).hex[:20]
                    )
                    event_index += 1
                    event["abs_start"] = start_sec
                    event["abs_end"] = end_sec
                    event["anomaly_category"] = anomaly_category
                    event["anomaly_confidence"] = anomaly_confidence
                    event["anomaly_prediction"] = {
                        "category": anomaly_category,
                        "confidence": anomaly_confidence,
                        "source": (
                            "uploaded_json" if supplied_category
                            else anomaly_summary.model_name or "unassessed"
                        ),
                        **({
                            "anomaly_probability": anomaly.metadata["anomaly_probability"],
                            "type_probabilities": anomaly.metadata["type_probabilities"],
                        } if anomaly and "anomaly_probability" in anomaly.metadata else {}),
                    }
                    event["event_id"] = event_id
                    clip_path = None
                    if pipeline_mode != "direct_json":
                        clip_path = self.clip_generator.cut_evidence_clip(
                            video_path, start_sec, end_sec, event_id, video.duration_sec
                        )
                    entity_ids = [
                        int(track_id)
                        for track_id in (
                            event.get("subject_track"), event.get("object_track")
                        )
                        if track_id is not None
                    ]
                    records.append(EventRecord(
                        event_id=event_id,
                        video_id=video_id,
                        event_type=str(event["event_type"]).lower(),
                        start_sec=start_sec,
                        end_sec=end_sec,
                        entity_ids=entity_ids,
                        description=event["description"],
                        confidence=float(event.get("confidence", 0.5)),
                        anomaly_category=anomaly_category,
                        anomaly_confidence=anomaly_confidence,
                        vlm_verified=bool(event.get("vlm_verified", False)),
                        clip_path=clip_path,
                        metadata={
                            "window_id": window_index + 1,
                            "camera_id": window.get("camera_id"),
                            "zone": event.get("zone"),
                            "pipeline_mode": pipeline_mode,
                            "source_json_sha256": source_json_sha256,
                            "anomaly_model_available": anomaly_summary.model_available,
                            "anomaly_model": anomaly_summary.model_name,
                        },
                    ))

            with timings.measure("persistence_evidence"):
                self.store.insert_events(records)
                self.vec_store.add_events(records)

            analysis_path = paths.outputs_dir / f"{video_id}_analysis.json"
            summary = {
                "video_id": video_id,
                "pipeline_mode": pipeline_mode,
                "status": "completed",
                "tracking_backend": (
                    tracking_result.backend if tracking_result else "precomputed_json"
                ),
                "total_tracks": tracking_result.total_tracks if tracking_result else
                    len({int(track["track_id"]) for window in windows_data for track in window["tracks"]}),
                "total_windows": len(windows_data),
                "total_events": len(records),
                "total_anomalies": sum(
                    1 for segment in anomaly_summary.anomaly_segments if segment.is_anomaly
                ),
                "total_anomaly_segments": len(anomaly_summary.anomaly_segments),
                "vlm_model_available": vlm_model_available,
                "anomaly_model_available": anomaly_summary.model_available,
                "processing_time_sec": round(perf_counter() - processing_started, 4),
                "stage_timings_sec": timings.values,
            }
            output_payload = {
                "video_id": video_id,
                "filename": video.filename,
                "pipeline_mode": pipeline_mode,
                "source_json_sha256": source_json_sha256,
                "summary": summary,
                "anomaly_summary": anomaly_summary.model_dump(mode="json"),
                "windows": windows_data,
            }
            self._write_json(analysis_path, output_payload)
            self.store.update_video_status(video_id, "completed")
            logger.info(f"Complete pipeline finished for {video_id}: {len(records)} events")

            return {
                "video_id": video_id,
                "status": "completed",
                "message": "Temporal windows, Qwen event understanding, anomaly assessment, persistence, and final JSON export completed.",
                "total_tracks": summary["total_tracks"],
                "tracking_backend": summary["tracking_backend"],
                "total_windows": len(windows_data),
                "total_events": len(records),
                "total_anomalies": summary["total_anomalies"],
                "total_anomaly_segments": summary["total_anomaly_segments"],
                "anomaly_model_available": anomaly_summary.model_available,
                "tracks_csv": str(tracks_csv_path) if tracking_result else None,
                "windows_json": str(windows_json_path),
                "analysis_json": str(analysis_path),
                "vlm_model_available": vlm_model_available,
                "stage_timings_sec": timings.values,
                "processing_time_sec": round(perf_counter() - processing_started, 4),
            }

        except Exception as e:
            logger.exception(f"Pipeline failed for {video_id}: {e}")
            self.store.update_video_status(video_id, "failed", error_message=str(e))
            raise

    @staticmethod
    def _uploaded_anomaly_category(event: Dict[str, Any]) -> Optional[str]:
        fields = (
            "anomaly_category",
            "anomaly_class",
            "anomaly_label",
            "anomaly_type",
            "predicted_category",
            "pred_14class",
            "predicted_class",
            "predicted_anomaly",
            "model_anomaly_class",
        )
        prediction = event.get("anomaly_prediction")
        candidates = [event.get(field) for field in fields]
        if isinstance(prediction, dict):
            candidates.extend(
                prediction.get(field)
                for field in (
                    "category", "class", "label", "predicted_class",
                    "overall_category",
                )
            )
        else:
            candidates.append(prediction)
        summary = event.get("anomaly_summary")
        if isinstance(summary, dict):
            candidates.append(summary.get("overall_category"))
        anomaly = event.get("anomaly")
        if isinstance(anomaly, dict):
            candidates.extend(
                anomaly.get(key) for key in ("category", "class", "label")
            )
        else:
            candidates.append(anomaly)
        canonical = {
            "".join(character for character in category.casefold() if character.isalnum()): category
            for category in ANOMALY_CATEGORIES
        }
        for candidate in candidates:
            if isinstance(candidate, str):
                normalized = "".join(
                    character
                    for character in candidate.casefold()
                    if character.isalnum()
                )
                if normalized in canonical:
                    return canonical[normalized]
        return None

    @classmethod
    def _inherit_window_anomaly_label(cls, window: Dict[str, Any]) -> None:
        """Copy a window-level JSON label onto events that do not have their own."""
        window_label = cls._uploaded_anomaly_category(window)
        if not window_label:
            return
        for event in window.get("events", []):
            if not cls._uploaded_anomaly_category(event):
                event["anomaly_category"] = window_label

    @staticmethod
    def _uploaded_anomaly_confidence(event: Dict[str, Any]) -> float:
        prediction = event.get("anomaly_prediction")
        candidates = [
            event.get("anomaly_confidence"),
            event.get("anomaly_probability"),
            event.get("anomaly_score"),
            event.get("model_anomaly_confidence"),
        ]
        if isinstance(prediction, dict):
            candidates.extend(
                prediction.get(key)
                for key in ("confidence", "probability", "score")
            )
        for candidate in candidates:
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if 0.0 <= value <= 1.0 and math.isfinite(value):
                return value
        return 1.0

    @classmethod
    def _anomaly_summary_from_windows(cls, video_id, windows):
        """Build an anomaly summary from explicit labels in direct-upload JSON."""
        labeled_events = []
        for window in windows:
            for event in window.get("events", []):
                category = cls._uploaded_anomaly_category(event)
                if category:
                    start = float(event.get("abs_start", window["window"][0] + event["start"]))
                    end = float(event.get("abs_end", window["window"][0] + event["end"]))
                    labeled_events.append((event, category, start, end))

        segments = [
            ClipAnomalyScore(
                clip_id=str(event.get("event_id") or f"{video_id}_json_{index + 1:03d}"),
                start_sec=round(start, 2),
                end_sec=round(end, 2),
                category=category,
                confidence=round(cls._uploaded_anomaly_confidence(event), 4),
                is_anomaly=category != "Normal",
                metadata={"source": "uploaded_json"},
            )
            for index, (event, category, start, end) in enumerate(labeled_events)
        ]
        if not segments:
            return VideoAnomalySummary(
                video_id=video_id,
                overall_category="Unassessed",
                overall_confidence=0.0,
                is_anomalous=False,
                model_available=False,
                model_name="uploaded_json",
                anomaly_segments=[],
            )

        category_durations: Dict[str, float] = {}
        for segment in segments:
            if segment.category != "Normal":
                category_durations[segment.category] = (
                    category_durations.get(segment.category, 0.0)
                    + max(segment.end_sec - segment.start_sec, 0.1)
                )
        overall_category = (
            max(category_durations, key=category_durations.get)
            if category_durations else "Normal"
        )
        return VideoAnomalySummary(
            video_id=video_id,
            overall_category=overall_category,
            overall_confidence=1.0,
            is_anomalous=overall_category != "Normal",
            model_available=False,
            model_name="uploaded_json",
            anomaly_segments=segments,
        )

    @staticmethod
    def _serialize_windows(video_id, windows, tracking_result, frame_width, frame_height):
        """Serialize track summaries using the Stage 2/4 notebook window schema."""
        grid_names = [
            "top_left", "top_mid", "top_right", "mid_left", "center", "mid_right",
            "bottom_left", "bottom_mid", "bottom_right",
        ]
        output = []
        for window in windows:
            start_sec, end_sec = window.start_sec, window.end_sec
            tracks = []
            active_tracks = []
            for track in tracking_result.tracks:
                observations = [
                    item for item in track.observations
                    if start_sec <= item.timestamp_sec <= end_sec
                ]
                if not observations:
                    continue
                active_tracks.append((track, observations))
                centers = [(item.bbox.center_x, item.bbox.center_y) for item in observations]
                displacement = math.hypot(
                    centers[-1][0] - centers[0][0], centers[-1][1] - centers[0][1]
                )
                path_length = sum(
                    math.hypot(current[0] - previous[0], current[1] - previous[1])
                    for previous, current in zip(centers, centers[1:])
                )
                dwell = observations[-1].timestamp_sec - observations[0].timestamp_sec
                zones = sorted({
                    grid_names[
                        min(int(center_y / max(frame_height, 1) * 3), 2) * 3
                        + min(int(center_x / max(frame_width, 1) * 3), 2)
                    ]
                    for center_x, center_y in centers
                })
                candidates = sorted({
                    event.event_type
                    for event in window.candidate_events
                    if track.track_id in event.entity_ids
                })
                tracks.append({
                    "track_id": int(track.track_id),
                    "class": track.class_name,
                    "dwell_sec": round(float(dwell), 2),
                    "displacement_px": round(float(displacement), 1),
                    "mean_speed": round(float(path_length / dwell), 1) if dwell > 0 else 0.0,
                    "zones": zones,
                    "near": [],
                    "candidate_events": candidates,
                })

            track_summary = {item["track_id"]: item for item in tracks}
            median_person_height = sorted(
                track.median_height for track, _ in active_tracks
                if track.class_name.lower() == "person"
            )
            proximity_px = median_person_height[len(median_person_height) // 2] if median_person_height else 100.0
            observations_by_track = {
                track.track_id: {item.frame_idx: item for item in observations}
                for track, observations in active_tracks
            }
            for track, observations in active_tracks:
                for other, other_observations in active_tracks:
                    if track.track_id >= other.track_id:
                        continue
                    other_by_frame = observations_by_track[other.track_id]
                    shared_observations = [
                        (item, other_by_frame[item.frame_idx])
                        for item in observations
                        if item.frame_idx in other_by_frame
                    ]
                    distances = [
                        math.hypot(
                            item.bbox.center_x - other_item.bbox.center_x,
                            item.bbox.center_y - other_item.bbox.center_y,
                        )
                        for item, other_item in shared_observations
                    ]
                    if distances and min(distances) <= proximity_px:
                        shared_timestamps = [
                            item.timestamp_sec for item, _ in shared_observations
                        ]
                        overlap_sec = max(shared_timestamps) - min(shared_timestamps)
                        track_summary[track.track_id]["near"].append({
                            "track_id": int(other.track_id),
                            "class": other.class_name,
                            "overlap_sec": round(overlap_sec, 1),
                        })
                        track_summary[other.track_id]["near"].append({
                            "track_id": int(track.track_id),
                            "class": track.class_name,
                            "overlap_sec": round(overlap_sec, 1),
                        })

            output.append({
                "video_id": video_id,
                "camera_id": video_id,
                "window": [float(start_sec), float(end_sec)],
                "tracks": tracks,
            })
        return output

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temporary_path, path)


# Global pipeline instance
surveillance_pipeline = SurveillancePipeline()
