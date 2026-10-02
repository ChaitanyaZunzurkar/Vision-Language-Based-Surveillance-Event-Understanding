"""
SurveillanceTracker -- YOLO + ByteTrack for surveillance video.

Generated from Stage 1b. Import this module wherever the pipeline needs
detection+tracking (Stage 2 temporal windowing, the testing notebook, etc.).
"""

import os
import glob
import re
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


def resolve_video_camera_id(video_path: str):
    """Returns (video_id, camera_id). See Stage 1b notebook Section 2 for details
    and the caveat about verifying the VIRAT scene-code regex against real filenames."""
    stem = Path(video_path).stem
    video_id = stem
    m = re.match(r"VIRAT_S_(\d{2})(\d{4})", stem)
    if m:
        return video_id, f"scene_{m.group(1)}"
    return video_id, video_id


class SurveillanceTracker:
    """YOLO detection + ByteTrack multi-object tracking for surveillance video."""

    SCHEMA_COLUMNS = [
        "video_id", "camera_id", "frame_id", "timestamp_sec",
        "x1", "y1", "x2", "y2",
        "confidence", "class_id", "class_name", "track_id",
    ]

    def __init__(self, weights_path, class_names, bytetrack_yaml,
                 detect_conf=0.10, imgsz=640, frame_stride=1):
        self.model = YOLO(weights_path)
        self.class_names = class_names
        self.bytetrack_yaml = bytetrack_yaml
        self.detect_conf = detect_conf
        self.imgsz = imgsz
        self.frame_stride = frame_stride

    def process_video(self, video_path, video_id=None, camera_id=None, resolver=None):
        if video_id is None or camera_id is None:
            if resolver is not None:
                video_id, camera_id = resolver(video_path)
            else:
                video_id = Path(video_path).stem
                camera_id = video_id

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        rows = []
        results_gen = self.model.track(
            source=video_path, tracker=self.bytetrack_yaml,
            conf=self.detect_conf, imgsz=self.imgsz,
            vid_stride=self.frame_stride, persist=False,
            stream=True, verbose=False,
        )

        for frame_idx, result in enumerate(results_gen):
            true_frame = frame_idx * self.frame_stride
            timestamp = true_frame / fps
            if result.boxes is None or result.boxes.id is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                rows.append({
                    "video_id": video_id, "camera_id": camera_id,
                    "frame_id": true_frame, "timestamp_sec": round(timestamp, 3),
                    "x1": round(float(boxes.xyxy[i][0]), 1),
                    "y1": round(float(boxes.xyxy[i][1]), 1),
                    "x2": round(float(boxes.xyxy[i][2]), 1),
                    "y2": round(float(boxes.xyxy[i][3]), 1),
                    "confidence": round(float(boxes.conf[i]), 3),
                    "class_id": cls_id,
                    "class_name": self.class_names.get(cls_id, str(cls_id)),
                    "track_id": int(boxes.id[i]),
                })
        return pd.DataFrame(rows, columns=self.SCHEMA_COLUMNS)

    def process_folder(self, folder, resolver, pattern="*.mp4"):
        video_paths = sorted(glob.glob(os.path.join(folder, pattern)))
        all_dfs = []
        for vp in video_paths:
            video_id, camera_id = resolver(vp)
            df = self.process_video(vp, video_id=video_id, camera_id=camera_id)
            all_dfs.append(df)
        if not all_dfs:
            return pd.DataFrame(columns=self.SCHEMA_COLUMNS)
        return pd.concat(all_dfs, ignore_index=True)
