"""Pipeline execution and status management API endpoints."""

from typing import Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel
from fastapi.responses import FileResponse
from server.src.storage.metadata_store import MetadataStore
from server.src.pipeline.pipeline import SurveillancePipeline
from server.app.api.deps import get_metadata_store, get_pipeline
from server.app.api.routes_videos import (
    _is_video_pipeline_active,
    _mark_video_pipeline_active,
    _process_video_task,
)
from server.src.utils.logger import logger
from server.src.utils.paths import paths

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


class PipelineRunResponse(BaseModel):
    video_id: str
    status: str
    message: str


@router.post("/run/{video_id}", response_model=Dict[str, Any])
def run_pipeline(
    video_id: str,
    background_tasks: BackgroundTasks,
    run_in_background: bool = Query(default=True),
    store: MetadataStore = Depends(get_metadata_store),
    pipeline: SurveillancePipeline = Depends(get_pipeline),
):
    """Trigger the multi-stage surveillance understanding pipeline on a video."""
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    if video.status == "processing" and _is_video_pipeline_active(video_id):
        return {
            "video_id": video_id,
            "status": "processing",
            "message": "Pipeline is already running for this video.",
        }

    if not _mark_video_pipeline_active(video_id):
        return {
            "video_id": video_id,
            "status": "processing",
            "message": "Pipeline is already running for this video.",
        }

    if run_in_background:
        background_tasks.add_task(_process_video_task, pipeline, video_id)
        return {
            "video_id": video_id,
            "status": "processing",
            "message": "Pipeline execution started in background.",
        }
    else:
        # Run synchronously
        try:
            result = _process_video_task(pipeline, video_id)
            return result
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Pipeline execution failed: {str(e)}"
            )


@router.get("/status/{video_id}")
def get_pipeline_status(
    video_id: str, store: MetadataStore = Depends(get_metadata_store)
):
    """Check pipeline processing status and event counts for a video."""
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    events = store.get_events(video_id=video_id)
    anomalies = store.get_anomalies(video_id=video_id)
    return {
        "video_id": video.video_id,
        "filename": video.filename,
        "status": video.status,
        "error_message": video.error_message,
        "total_events": len(events),
        "total_anomalies": sum(1 for anomaly in anomalies if anomaly.is_anomaly),
        "total_anomaly_segments": len(anomalies),
        "evidence_clips": sum(1 for event in events if event.clip_path),
    }


@router.get("/output/{video_id}")
def get_pipeline_output(video_id: str, store: MetadataStore = Depends(get_metadata_store)):
    """Download the final per-video analysis JSON after pipeline completion."""
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    if video.status != "completed":
        raise HTTPException(status_code=409, detail="Analysis JSON is available after processing completes")
    output_path = paths.outputs_dir / f"{video_id}_analysis.json"
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="Analysis JSON is not available yet")
    return FileResponse(
        output_path,
        media_type="application/json",
        filename=f"{Path(video.filename).stem}_analysis.json",
    )
