"""Video upload, retrieval, and management API endpoints."""

from pathlib import Path
from typing import List, Optional
import shutil
import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from src.storage.schemas import VideoRecord
from src.storage.metadata_store import MetadataStore
from src.pipeline.ingestion import VideoIngestor
from src.pipeline.pipeline import SurveillancePipeline
from src.utils.paths import paths
from src.utils.logger import logger
from app.api.deps import get_metadata_store, get_ingestor, get_pipeline

router = APIRouter(prefix="/videos", tags=["Videos"])


class VideoResponse(BaseModel):
    video_id: str
    filename: str
    duration_sec: float
    fps: float
    frame_count: int
    resolution: str
    status: str
    error_message: Optional[str] = None
    created_at: str
    stream_url: str


def to_response(v: VideoRecord) -> VideoResponse:
    # URL for streaming the video file
    stream_url = f"/api/media/video/{Path(v.file_path).name}"
    return VideoResponse(
        video_id=v.video_id,
        filename=v.filename,
        duration_sec=v.duration_sec,
        fps=v.fps,
        frame_count=v.frame_count,
        resolution=v.resolution,
        status=v.status,
        error_message=v.error_message,
        created_at=v.created_at,
        stream_url=stream_url,
    )


@router.post("/upload", response_model=VideoResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_run: bool = Form(default=False),
    store: MetadataStore = Depends(get_metadata_store),
    ingestor: VideoIngestor = Depends(get_ingestor),
    pipeline: SurveillancePipeline = Depends(get_pipeline),
):
    """Upload a surveillance video file, extract metadata, and register in system."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename provided")

    # Validate video extension
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{suffix}'. Allowed: {allowed_exts}",
        )

    # Save uploaded file temporarily to disk
    temp_path = paths.uploads_dir / f"tmp_{file.filename}"
    try:
        async with aiofiles.open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)

        # Ingest into store
        record = ingestor.ingest_video(temp_path, original_filename=file.filename)

        # Clean up temp file
        if temp_path.exists() and temp_path.resolve() != Path(record.file_path).resolve():
            temp_path.unlink()

        # Optionally trigger background pipeline
        if auto_run:
            background_tasks.add_task(pipeline.process_video, record.video_id)

        return to_response(record)

    except Exception as e:
        logger.exception(f"Error handling video upload: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")


@router.get("", response_model=List[VideoResponse])
def list_videos(store: MetadataStore = Depends(get_metadata_store)):
    """List all registered surveillance videos."""
    videos = store.list_videos()
    return [to_response(v) for v in videos]


@router.get("/{video_id}", response_model=VideoResponse)
def get_video(video_id: str, store: MetadataStore = Depends(get_metadata_store)):
    """Get metadata for a specific video."""
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return to_response(video)
