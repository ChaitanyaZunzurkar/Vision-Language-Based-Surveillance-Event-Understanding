"""Pipeline execution and status management API endpoints."""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel
from server.src.storage.metadata_store import MetadataStore
from server.src.pipeline.pipeline import SurveillancePipeline
from server.app.api.deps import get_metadata_store, get_pipeline
from server.src.utils.logger import logger

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

    if video.status == "processing":
        return {
            "video_id": video_id,
            "status": "processing",
            "message": "Pipeline is already running for this video.",
        }

    if run_in_background:
        background_tasks.add_task(pipeline.process_video, video_id)
        return {
            "video_id": video_id,
            "status": "processing",
            "message": "Pipeline execution started in background.",
        }
    else:
        # Run synchronously
        try:
            result = pipeline.process_video(video_id)
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
    return {
        "video_id": video.video_id,
        "filename": video.filename,
        "status": video.status,
        "error_message": video.error_message,
        "total_events": len(events),
    }


