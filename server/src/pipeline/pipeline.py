"""End-to-end master pipeline orchestrator for surveillance video understanding."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import importlib.util
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
from server.src.anomaly.detector import AnomalyDetector
from server.src.anomaly.schemas import VideoAnomalySummary
from server.src.event_understanding.qwen import VLMEventUnderstanding
from server.src.storage.schemas import EventRecord, AnomalyRecord, TrackRecord
from server.src.evidence.clip_generator import ClipGenerator
from server.src.utils.timing import StageTimings
from server.src.utils.device import log_device_diagnostics


class SurveillancePipeline:
    """Master orchestrator executing the full surveillance understanding pipeline on a video."""

    def __init__(
        self,
        store: Optional[MetadataStore] = None,
        vec_store: Optional[VectorStore] = None,
    ):
        self.store = store or metadata_store
        self.vec_store = vec_store or vector_store

        # Pipeline submodules
        self.windower = TemporalWindower(
            window_duration_sec=config_loader.get("temporal.window_duration_sec", 10.0),
            window_stride_sec=config_loader.get("temporal.window_stride_sec", 5.0),
            loiter_dwell_sec=config_loader.get("temporal.loiter_dwell_sec", 4.0),
        )
        self.anomaly_detector = AnomalyDetector(
            model_path=paths.root_dir / config_loader.get("anomaly.model_weights", "weights/best_model.pkl"),
            scaler_path=paths.root_dir / config_loader.get("anomaly.scaler_weights", "weights/scaler.pkl"),
            clip_duration_sec=float(config_loader.get("anomaly.clip_duration_sec", 4.0)),
            clip_stride_sec=float(config_loader.get("anomaly.clip_stride_sec", 2.0)),
            anomaly_threshold=float(config_loader.get("anomaly.anomaly_threshold", 0.5)),
        )
        self.vlm = VLMEventUnderstanding(
            model_name=config_loader.get("vlm.model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
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
        tracker = module.SurveillanceTracker(
            weights_path=str(weights_path),
            class_names=class_names,
            bytetrack_yaml=str(bytetrack_yaml),
            detect_conf=float(config_loader.get("detection.conf_threshold", 0.20)),
            imgsz=int(model_config.get("imgsz", 640)),
            frame_stride=int(config_loader.get("tracking.frame_stride", 1)),
            device=config_loader.get("runtime.device", "auto"),
        )
        tracks_df = tracker.process_video(str(video_path), video_id=video_id, camera_id=video_id)
        if tracks_df.empty and config_loader.get("stage1b.allow_detector_fallback", True):
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
        return TrackingResult(video_id=video_id, total_tracks=len(tracks), tracks=tracks)

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
        return result

    def process_video(self, video_id: str) -> Dict[str, Any]:
        """Run complete understanding pipeline on an ingested video.

        Available stages: Stage 1b YOLO+ByteTrack and Stage 2 temporal windowing.
        Stage 4 events can be imported from the VISTA-generated JSON endpoint.
        """
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
            # Stage 1b: use the real VISTA detector+ByteTrack code, not a heuristic fallback.
            logger.info("Stage 1b: Running YOLO + ByteTrack...")
            with timings.measure("detection_tracking"):
                tracking_result = self._run_stage1b(video_path, video_id)
            self.store.insert_tracks(
                [
                    TrackRecord(
                        video_id=video_id,
                        track_id=track.track_id,
                        class_name=track.class_name,
                        start_sec=track.start_sec,
                        end_sec=track.end_sec,
                        total_observations=len(track.observations),
                    )
                    for track in tracking_result.tracks
                ],
            )

            # Stage 1b writes the per-video CSV as part of _run_stage1b.
            tracks_csv_path = paths.tracks_dir / f"{video_id}_tracks.csv"

            # Stage 2: Temporal Windowing
            logger.info("Stage 2/5: Generating Temporal Windows & Candidate Events...")
            with timings.measure("temporal_events"):
                windows = self.windower.generate_windows(tracking_result, video.duration_sec)

            # Export windows JSON
            windows_json_path = paths.windows_dir / f"{video_id}_windows.json"
            windows_data = []
            for window in windows:
                start_sec, end_sec = window.start_sec, window.end_sec
                window_tracks = []
                for track in tracking_result.tracks:
                    if track.track_id not in window.track_ids:
                        continue
                    observations = [
                        obs for obs in track.observations
                        if start_sec <= obs.timestamp_sec <= end_sec
                    ]
                    if len(observations) > 1:
                        first, last = observations[0].bbox, observations[-1].bbox
                        displacement = ((last.center_x - first.center_x) ** 2 + (last.center_y - first.center_y) ** 2) ** 0.5
                        dwell = observations[-1].timestamp_sec - observations[0].timestamp_sec
                    else:
                        displacement, dwell = 0.0, 0.0
                    candidates = [
                        event.event_type for event in window.candidate_events
                        if track.track_id in event.entity_ids
                    ]
                    window_tracks.append({
                        "track_id": int(track.track_id),
                        "class": track.class_name,
                        "dwell_sec": round(float(dwell), 3),
                        "displacement_px": round(float(displacement), 2),
                        "mean_speed": round(float(displacement / dwell), 2) if dwell > 0 else 0.0,
                        "zones": [],
                        "near": [],
                        "candidate_events": candidates,
                    })
                windows_data.append({
                    "video_id": video_id,
                    "camera_id": video_id,
                    "window": [float(start_sec), float(end_sec)],
                    "tracks": window_tracks,
                })
            with open(windows_json_path, "w", encoding="utf-8") as f:
                json.dump({"windows": windows_data}, f, indent=2)

            logger.info("Stage 3: Running AnomalyCLIP feature classifier...")
            with timings.measure("anomaly"):
                anomaly_summary = self.anomaly_detector.detect_anomalies(
                    video_path, video_id, video.duration_sec, tracking_result
                )
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

            logger.info("Stage 4: Grounding candidate events with Qwen2.5-VL...")
            with timings.measure("vlm"):
                understanding = self.vlm.enrich_and_verify_events(
                    video_id, windows, tracking_result, video_path
                )
            records = []
            for semantic in understanding.events:
                overlapping = [
                    segment for segment in anomaly_summary.anomaly_segments
                    if segment.end_sec >= semantic.start_sec and segment.start_sec <= semantic.end_sec
                ]
                anomaly = max(overlapping, key=lambda item: item.confidence, default=None)
                clip_path = self.clip_generator.cut_evidence_clip(
                    video_path,
                    semantic.start_sec,
                    semantic.end_sec,
                    semantic.event_id,
                    video.duration_sec,
                )
                records.append(EventRecord(
                    event_id=semantic.event_id,
                    video_id=video_id,
                    event_type=semantic.event_type,
                    start_sec=semantic.start_sec,
                    end_sec=semantic.end_sec,
                    entity_ids=semantic.entity_ids,
                    description=semantic.description,
                    confidence=semantic.confidence,
                    anomaly_category=anomaly.category if anomaly else "Unassessed",
                    anomaly_confidence=anomaly.confidence if anomaly else 0.0,
                    vlm_verified=semantic.vlm_verified,
                    clip_path=clip_path,
                    metadata={
                        **semantic.metadata,
                        "anomaly_model_available": anomaly_summary.model_available,
                        "anomaly_model": anomaly_summary.model_name,
                    },
                ))
            with timings.measure("persistence_evidence"):
                self.store.insert_events(records)
                self.vec_store.add_events(records)

            self.store.update_video_status(video_id, "completed")
            logger.info(f"Complete pipeline finished for {video_id}: {len(records)} events")

            return {
                "video_id": video_id,
                "status": "completed",
                "message": "Detection, tracking, temporal events, anomaly assessment, VLM grounding, persistence, and evidence generation completed.",
                "total_tracks": tracking_result.total_tracks,
                "total_windows": len(windows),
                "total_events": len(records),
                "total_anomalies": len(anomaly_summary.anomaly_segments),
                "anomaly_model_available": anomaly_summary.model_available,
                "tracks_csv": str(tracks_csv_path),
                "windows_json": str(windows_json_path),
                "stage_timings_sec": timings.values,
                "processing_time_sec": round(perf_counter() - processing_started, 4),
            }

        except Exception as e:
            logger.exception(f"Pipeline failed for {video_id}: {e}")
            self.store.update_video_status(video_id, "failed", error_message=str(e))
            raise e


# Global pipeline instance
surveillance_pipeline = SurveillancePipeline()
