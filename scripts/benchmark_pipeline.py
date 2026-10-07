"""Run a reproducible Phase 1 pipeline benchmark.

Usage:
    python scripts/benchmark_pipeline.py path/to/video.mp4 --device cpu
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.src.pipeline.ingestion import VideoIngestor
from server.src.pipeline.pipeline import SurveillancePipeline
from server.src.storage.metadata_store import metadata_store
from server.src.utils.device import get_device_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if args.device:
        # The runtime resolver reads configuration at import time; this option
        # is intended for comparing already configured environments.
        raise SystemExit("Set runtime.device in server/configs/config.yaml before running.")

    started = perf_counter()
    info = get_device_info()
    record = VideoIngestor().ingest_video(args.video, args.video.name)
    result = SurveillancePipeline().process_video(record.video_id)
    elapsed = perf_counter() - started
    print(json.dumps({
        "video_id": record.video_id,
        "device": info.device,
        "gpu": info.gpu_name,
        "video_duration_sec": record.duration_sec,
        "video_fps": record.fps,
        "frame_count": record.frame_count,
        "processing_time_sec": round(elapsed, 3),
        "real_time_factor": round(elapsed / max(record.duration_sec, 0.001), 3),
        **result,
    }, indent=2))


if __name__ == "__main__":
    main()
