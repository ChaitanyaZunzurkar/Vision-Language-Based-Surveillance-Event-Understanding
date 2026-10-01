"""End-to-end master pipeline orchestrator for surveillance video understanding."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from src.utils.paths import paths
from src.utils.logger import logger
from src.storage.schemas import VideoRecord, EventRecord, AnomalyRecord
from src.storage.metadata_store import MetadataStore, metadata_store
from src.storage.vector_store import VectorStore, vector_store
from src.detection.detector import SurveillanceDetector
from src.tracking.tracker import SurveillanceTracker
from src.event_understanding.event_generator import TemporalWindower
from src.event_understanding.qwen import VLMEventUnderstanding
from src.anomaly.detector import AnomalyDetector
from src.evidence.clip_generator import ClipGenerator
from src.config.loader import config_loader


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
        self.detector = SurveillanceDetector(
            weights_path=config_loader.get("detection.model_weights"),
            conf_threshold=config_loader.get("detection.conf_threshold", 0.20),
        )
        self.tracker = SurveillanceTracker(
            track_high_thresh=config_loader.get("tracking.track_high_thresh", 0.50),
            track_low_thresh=config_loader.get("tracking.track_low_thresh", 0.20),
            new_track_thresh=config_loader.get("tracking.new_track_thresh", 0.60),
        )
        self.windower = TemporalWindower(
            window_duration_sec=config_loader.get("temporal.window_duration_sec", 10.0),
            window_stride_sec=config_loader.get("temporal.window_stride_sec", 5.0),
            loiter_dwell_sec=config_loader.get("temporal.loiter_dwell_sec", 4.0),
        )
        self.vlm = VLMEventUnderstanding()
        self.anomaly_detector = AnomalyDetector(
            model_path=config_loader.get("anomaly.model_weights"),
            scaler_path=config_loader.get("anomaly.scaler_weights"),
        )
        self.clip_generator = ClipGenerator(
            output_dir=paths.clips_dir,
            padding_before_sec=config_loader.get("evidence.padding_before_sec", 1.5),
            padding_after_sec=config_loader.get("evidence.padding_after_sec", 1.5),
        )

    def process_video(self, video_id: str) -> Dict[str, Any]:
        """Run complete understanding pipeline on an ingested video.

        Stages:
            1. Detection & Tracking (YOLO + ByteTrack)
            2. Temporal Windowing (Rule-based candidate extraction)
            3. Vision-Language Understanding (Qwen2.5-VL verification)
            4. Anomaly Detection (UCF-Crime 14-category classification)
            5. Evidence Clip Generation (OpenCV / FFmpeg cutter)
            6. Metadata & Vector Store Ingestion
        """
        video = self.store.get_video(video_id)
        if not video:
            raise ValueError(f"Video {video_id} not found in metadata store")

        video_path = Path(video.file_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file missing on disk: {video_path}")

        logger.info(f"=== Starting Surveillance Pipeline for {video_id} ({video.filename}) ===")
        self.store.update_video_status(video_id, "processing")

        try:
            # Stage 1: Detection & Tracking
            logger.info("Stage 1/5: Running Entity Detection & Tracking...")
            detection_result = self.detector.detect_video(video_path, video_id)
            tracking_result = self.tracker.track_video(detection_result)

            # Export tracks CSV
            tracks_df = self.tracker.export_tracks_dataframe(tracking_result)
            tracks_csv_path = paths.tracks_dir / f"{video_id}_tracks.csv"
            tracks_df.to_csv(tracks_csv_path, index=False)

            # Stage 2: Temporal Windowing
            logger.info("Stage 2/5: Generating Temporal Windows & Candidate Events...")
            windows = self.windower.generate_windows(tracking_result, video.duration_sec)

            # Export windows JSON
            windows_json_path = paths.windows_dir / f"{video_id}_windows.json"
            windows_data = [w.model_dump() for w in windows]
            with open(windows_json_path, "w", encoding="utf-8") as f:
                json.dump(windows_data, f, indent=2)

            # Stage 3: Vision-Language Understanding (Qwen2.5-VL)
            logger.info("Stage 3/5: VLM Event Understanding & Grounding...")
            vlm_result = self.vlm.enrich_and_verify_events(
                video_id=video_id,
                windows=windows,
                tracking_result=tracking_result,
                video_path=video_path,
            )

            # Stage 4: Video Anomaly Detection
            logger.info("Stage 4/5: Running Anomaly Detection...")
            anomaly_summary = self.anomaly_detector.detect_anomalies(
                video_path=video_path,
                video_id=video_id,
                video_duration_sec=video.duration_sec,
                tracking_result=tracking_result,
            )

            # Cross-reference anomaly category with events
            for event in vlm_result.events:
                # Find matching anomaly clip
                for anom_clip in anomaly_summary.anomaly_segments:
                    if (
                        anom_clip.is_anomaly
                        and not (anom_clip.end_sec < event.start_sec or anom_clip.start_sec > event.end_sec)
                    ):
                        event.anomaly_category = anom_clip.category
                        event.anomaly_confidence = anom_clip.confidence
                        break
                else:
                    event.anomaly_category = anomaly_summary.overall_category
                    event.anomaly_confidence = anomaly_summary.overall_confidence

            # Stage 5: Evidence Clip Extraction
            logger.info("Stage 5/5: Cutting Evidence Video Clips for Playback...")
            event_records: List[EventRecord] = []
            for ev in vlm_result.events:
                clip_path = self.clip_generator.cut_evidence_clip(
                    video_path=video_path,
                    start_sec=ev.start_sec,
                    end_sec=ev.end_sec,
                    clip_name=ev.event_id,
                    video_duration_sec=video.duration_sec,
                )
                rec = EventRecord(
                    event_id=ev.event_id,
                    video_id=ev.video_id,
                    event_type=ev.event_type,
                    start_sec=ev.start_sec,
                    end_sec=ev.end_sec,
                    entity_ids=ev.entity_ids,
                    description=ev.description,
                    confidence=ev.confidence,
                    anomaly_category=ev.anomaly_category,
                    anomaly_confidence=ev.anomaly_confidence,
                    clip_path=clip_path,
                    metadata=ev.metadata,
                )
                event_records.append(rec)

            # Stage 6: Database & Vector Index Ingestion
            logger.info("Ingesting records into SQLite store and Vector index...")
            self.store.insert_events(event_records)

            anomaly_records = [
                AnomalyRecord(
                    anomaly_id=c.clip_id,
                    video_id=video_id,
                    category=c.category,
                    confidence=c.confidence,
                    is_anomaly=c.is_anomaly,
                    start_sec=c.start_sec,
                    end_sec=c.end_sec,
                )
                for c in anomaly_summary.anomaly_segments
            ]
            self.store.insert_anomalies(anomaly_records)

            # Add to vector store for natural language queries
            self.vec_store.add_events(event_records)

            self.store.update_video_status(video_id, "completed")
            logger.info(f"=== Pipeline completed successfully for video {video_id} ===")

            return {
                "video_id": video_id,
                "status": "completed",
                "total_detections": detection_result.total_detections,
                "total_tracks": tracking_result.total_tracks,
                "total_windows": len(windows),
                "total_events": len(event_records),
                "overall_anomaly": anomaly_summary.overall_category,
                "is_anomalous": anomaly_summary.is_anomalous,
                "tracks_csv": str(tracks_csv_path),
                "windows_json": str(windows_json_path),
            }

        except Exception as e:
            logger.exception(f"Pipeline failed for {video_id}: {e}")
            self.store.update_video_status(video_id, "failed", error_message=str(e))
            raise e


# Global pipeline instance
surveillance_pipeline = SurveillancePipeline()
