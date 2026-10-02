"""End-to-end master pipeline orchestrator for surveillance video understanding."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import importlib.util
import yaml
from src.utils.paths import paths
from src.utils.logger import logger
from src.storage.metadata_store import MetadataStore, metadata_store
from src.storage.vector_store import VectorStore, vector_store
from src.tracking.schemas import Track, TrackObservation, TrackingResult
from src.detection.schemas import BoundingBox
from src.event_understanding.event_generator import TemporalWindower
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
        self.windower = TemporalWindower(
            window_duration_sec=config_loader.get("temporal.window_duration_sec", 10.0),
            window_stride_sec=config_loader.get("temporal.window_stride_sec", 5.0),
            loiter_dwell_sec=config_loader.get("temporal.loiter_dwell_sec", 4.0),
        )

    def _run_stage1b(self, video_path: Path, video_id: str) -> TrackingResult:
        """Run the project's actual VISTA YOLO+ByteTrack implementation."""
        tracker_script = (paths.root_dir / config_loader.get(
            "stage1b.tracker_script", "src/tracking/surveillance_tracker.py"
        )).resolve()
        weights_path = (paths.root_dir / config_loader.get(
            "stage1b.weights_path", "models/stage1b/yolo11x_surv_v1.pt"
        )).resolve()
        bytetrack_yaml = (paths.root_dir / config_loader.get(
            "stage1b.bytetrack_yaml", "configs/bytetrack_custom.yaml"
        )).resolve()
        stage1_config = (paths.root_dir / config_loader.get(
            "stage1b.class_config", "configs/stage1_model.yaml"
        )).resolve()
        required = [tracker_script, weights_path, bytetrack_yaml, stage1_config]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
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
            frame_stride=int(config_loader.get("stage1b.frame_stride", 1)),
        )
        tracks_df = tracker.process_video(str(video_path), video_id=video_id, camera_id=video_id)
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
        return TrackingResult(video_id=video_id, total_tracks=len(tracks), tracks=tracks)

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
        self.store.update_video_status(video_id, "processing")

        try:
            # Stage 1b: use the real VISTA detector+ByteTrack code, not a heuristic fallback.
            logger.info("Stage 1b: Running YOLO + ByteTrack...")
            tracking_result = self._run_stage1b(video_path, video_id)

            # Stage 1b writes the per-video CSV as part of _run_stage1b.
            tracks_csv_path = paths.tracks_dir / f"{video_id}_tracks.csv"

            # Stage 2: Temporal Windowing
            logger.info("Stage 2/5: Generating Temporal Windows & Candidate Events...")
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

            self.store.update_video_status(video_id, "awaiting_events")
            logger.info(f"Stages 1b and 2 completed for {video_id}; waiting for Stage 4 JSON import")

            return {
                "video_id": video_id,
                "status": "awaiting_events",
                "message": "Stages 1b and 2 completed. Import the Stage 4 events.json output to enable event search and evidence playback.",
                "total_tracks": tracking_result.total_tracks,
                "total_windows": len(windows),
                "tracks_csv": str(tracks_csv_path),
                "windows_json": str(windows_json_path),
            }

        except Exception as e:
            logger.exception(f"Pipeline failed for {video_id}: {e}")
            self.store.update_video_status(video_id, "failed", error_message=str(e))
            raise e


# Global pipeline instance
surveillance_pipeline = SurveillancePipeline()
