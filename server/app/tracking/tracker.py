"""Surveillance multi-object tracker implementing two-tier ByteTrack association logic."""

from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from server.app.detection.schemas import BoundingBox, Detection, DetectionResult
from server.app.tracking.schemas import Track, TrackObservation, TrackingResult
from server.app.utils.logger import logger
from server.app.config.loader import config_loader


def compute_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    areaA = boxA.area
    areaB = boxB.area
    union_area = areaA + areaB - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class SurveillanceTracker:
    """Tracks detected surveillance entities across frames with per-video isolation."""

    def __init__(
        self,
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.20,
        new_track_thresh: float = 0.60,
        max_lost_frames: int = 15,
        match_thresh: float = 0.30,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.max_lost_frames = max_lost_frames
        self.match_thresh = match_thresh
        self.reset()

    def reset(self) -> None:
        """Reset internal tracking state for new video (persist=False)."""
        self.next_track_id = 1
        self.active_tracks: Dict[int, Track] = {}
        self.lost_counters: Dict[int, int] = {}
        self.finished_tracks: List[Track] = []

    def update_frame(
        self, detections: List[Detection], frame_idx: int, timestamp_sec: float
    ) -> List[TrackObservation]:
        """Update tracker with detections for one frame.

        Implements ByteTrack two-tier association:
        1. Separate detections into high-score (>= track_high_thresh) and low-score.
        2. Match high-score detections with active tracks using IoU.
        3. Match remaining unmatched active tracks with low-score detections to recover occluded objects.
        4. Initialize new tracks from remaining unmatched high-score detections (>= new_track_thresh).
        5. Mark missing tracks as lost; retire tracks lost for > max_lost_frames.
        """
        # Split detections
        high_dets = [d for d in detections if d.confidence >= self.track_high_thresh]
        low_dets = [
            d
            for d in detections
            if self.track_low_thresh <= d.confidence < self.track_high_thresh
        ]

        active_ids = list(self.active_tracks.keys())
        matched_tracks = set()
        matched_high_dets = set()
        matched_low_dets = set()

        # Step 1: Match active tracks with high-score detections
        for tid in active_ids:
            track = self.active_tracks[tid]
            last_bbox = track.observations[-1].bbox

            best_iou = 0.0
            best_det_idx = -1
            for didx, det in enumerate(high_dets):
                if didx in matched_high_dets:
                    continue
                # Class compatibility preference
                if det.class_name == track.class_name:
                    iou = compute_iou(last_bbox, det.bbox)
                    if iou > best_iou and iou >= self.match_thresh:
                        best_iou = iou
                        best_det_idx = didx

            if best_det_idx >= 0:
                det = high_dets[best_det_idx]
                obs = TrackObservation(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    bbox=det.bbox,
                    confidence=det.confidence,
                )
                track.observations.append(obs)
                track.end_sec = timestamp_sec
                self.lost_counters[tid] = 0
                matched_tracks.add(tid)
                matched_high_dets.add(best_det_idx)

        # Step 2: Match remaining active tracks with low-score detections (occlusion recovery)
        unmatched_track_ids = [tid for tid in active_ids if tid not in matched_tracks]
        for tid in unmatched_track_ids:
            track = self.active_tracks[tid]
            last_bbox = track.observations[-1].bbox

            best_iou = 0.0
            best_det_idx = -1
            for didx, det in enumerate(low_dets):
                if didx in matched_low_dets:
                    continue
                if det.class_name == track.class_name:
                    iou = compute_iou(last_bbox, det.bbox)
                    if iou > best_iou and iou >= self.match_thresh:
                        best_iou = iou
                        best_det_idx = didx

            if best_det_idx >= 0:
                det = low_dets[best_det_idx]
                obs = TrackObservation(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    bbox=det.bbox,
                    confidence=det.confidence,
                )
                track.observations.append(obs)
                track.end_sec = timestamp_sec
                self.lost_counters[tid] = 0
                matched_tracks.add(tid)
                matched_low_dets.add(best_det_idx)

        # Step 3: Initialize new tracks from unmatched high-confidence detections
        for didx, det in enumerate(high_dets):
            if didx not in matched_high_dets and det.confidence >= self.new_track_thresh:
                new_track = Track(
                    track_id=self.next_track_id,
                    class_name=det.class_name,
                    start_sec=timestamp_sec,
                    end_sec=timestamp_sec,
                    observations=[
                        TrackObservation(
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp_sec,
                            bbox=det.bbox,
                            confidence=det.confidence,
                        )
                    ],
                )
                self.active_tracks[self.next_track_id] = new_track
                self.lost_counters[self.next_track_id] = 0
                self.next_track_id += 1

        # Step 4: Handle lost tracks
        for tid in active_ids:
            if tid not in matched_tracks:
                self.lost_counters[tid] = self.lost_counters.get(tid, 0) + 1
                if self.lost_counters[tid] > self.max_lost_frames:
                    # Retire track
                    self.finished_tracks.append(self.active_tracks.pop(tid))
                    self.lost_counters.pop(tid, None)

        current_observations = []
        for tid, track in self.active_tracks.items():
            if track.observations and track.observations[-1].frame_idx == frame_idx:
                current_observations.append(track.observations[-1])

        return current_observations

    def finalize(self) -> List[Track]:
        """Finalize and return all tracks."""
        for tid, track in self.active_tracks.items():
            self.finished_tracks.append(track)
        self.active_tracks.clear()
        self.lost_counters.clear()
        return self.finished_tracks

    def track_video(self, detection_result: DetectionResult) -> TrackingResult:
        """Process detection result for a full video and return TrackingResult."""
        self.reset()
        video_id = detection_result.video_id

        # Group detections by frame_idx
        frames_dict: Dict[int, List[Detection]] = {}
        frame_timestamps: Dict[int, float] = {}

        for det in detection_result.detections:
            frames_dict.setdefault(det.frame_idx, []).append(det)
            frame_timestamps[det.frame_idx] = det.timestamp_sec

        sorted_frames = sorted(frames_dict.keys())
        for f_idx in sorted_frames:
            dets = frames_dict[f_idx]
            ts = frame_timestamps[f_idx]
            self.update_frame(dets, f_idx, ts)

        tracks = self.finalize()

        # Filter out trivial 1-frame noise tracks
        valid_tracks = [t for t in tracks if len(t.observations) >= 2 or t.duration_sec >= 0.5]

        logger.info(
            f"Tracking completed for {video_id}: generated {len(valid_tracks)} persistent tracks"
        )
        return TrackingResult(
            video_id=video_id,
            total_tracks=len(valid_tracks),
            tracks=valid_tracks,
        )

    def export_tracks_dataframe(self, tracking_result: TrackingResult) -> pd.DataFrame:
        """Convert TrackingResult into a pandas DataFrame matching tracks.csv specification."""
        rows = []
        for track in tracking_result.tracks:
            for obs in track.observations:
                rows.append(
                    {
                        "video_id": tracking_result.video_id,
                        "frame_id": obs.frame_idx,
                        "timestamp_sec": obs.timestamp_sec,
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                        "x1": obs.bbox.x1,
                        "y1": obs.bbox.y1,
                        "x2": obs.bbox.x2,
                        "y2": obs.bbox.y2,
                        "confidence": obs.confidence,
                    }
                )
        return pd.DataFrame(rows)

