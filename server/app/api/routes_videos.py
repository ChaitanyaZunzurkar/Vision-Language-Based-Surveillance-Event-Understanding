"""Video upload, retrieval, and management API endpoints."""

from pathlib import Path
from typing import List, Optional
import hashlib
import shutil
import uuid
import json
import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from server.src.storage.schemas import VideoRecord
from server.src.storage.metadata_store import MetadataStore
from server.src.pipeline.ingestion import VideoIngestor
from server.src.pipeline.pipeline import SurveillancePipeline
from server.src.utils.paths import paths
from server.src.utils.logger import logger
from server.src.utils.video_utils import get_video_info
from server.src.pipeline.json_input import validate_windows_payload
from server.src.storage.vector_store import VectorStore
from server.app.api.deps import get_metadata_store, get_ingestor, get_pipeline, get_vector_store

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
    analysis_url: str


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
        analysis_url=f"/api/pipeline/output/{v.video_id}",
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
    safe_filename = Path(file.filename).name
    temp_path = paths.uploads_dir / f"tmp_{uuid.uuid4().hex}{Path(safe_filename).suffix.lower()}"
    try:
        async with aiofiles.open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)

        # Ingest into store
        record = ingestor.ingest_video(temp_path, original_filename=safe_filename)

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


@router.post("/upload-direct", response_model=VideoResponse)
async def upload_video_with_windows(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    json_file: UploadFile = File(...),
    store: MetadataStore = Depends(get_metadata_store),
    ingestor: VideoIngestor = Depends(get_ingestor),
    pipeline: SurveillancePipeline = Depends(get_pipeline),
):
    """Ingest a video plus notebook-format windows JSON and skip detection/tracking."""
    if not file.filename or not json_file.filename:
        raise HTTPException(status_code=400, detail="Select both a video and a windows JSON file")
    if Path(file.filename).suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")
    if Path(json_file.filename).suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="The accompanying file must be .json")

    try:
        json_bytes = await json_file.read(50 * 1024 * 1024 + 1)
        if len(json_bytes) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Windows JSON exceeds the 50 MB upload limit")
        try:
            payload = json.loads(json_bytes.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}") from exc

        safe_filename = Path(file.filename).name
        temp_path = paths.uploads_dir / (
            f"tmp_{uuid.uuid4().hex}{Path(safe_filename).suffix.lower()}"
        )
        async with aiofiles.open(temp_path, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                await output.write(chunk)

        video_info = get_video_info(temp_path)
        try:
            windows = validate_windows_payload(
                payload, {Path(safe_filename).stem}, video_info["duration_sec"]
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        record = ingestor.ingest_video(temp_path, original_filename=safe_filename)
        if temp_path.exists() and temp_path.resolve() != Path(record.file_path).resolve():
            temp_path.unlink()
        if record.status == "processing":
            raise HTTPException(status_code=409, detail="This video is already being processed")
        store.update_video_status(record.video_id, "processing")
        record.status = "processing"
        background_tasks.add_task(
            pipeline.process_video,
            record.video_id,
            windows,
            "direct_json",
        )
        return to_response(record)
    except HTTPException:
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()
        raise
    except Exception as exc:
        logger.exception("Error handling direct video + JSON upload: %s", exc)
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process direct upload: {exc}") from exc


@router.get("", response_model=List[VideoResponse])
def list_videos(store: MetadataStore = Depends(get_metadata_store)):
    """List all registered surveillance videos."""
    videos = store.list_videos()
    return [to_response(v) for v in videos]


@router.post("/import-demo", response_model=List[VideoResponse])
def import_demo_videos(
    background_tasks: BackgroundTasks,
    store: MetadataStore = Depends(get_metadata_store),
    ingestor: VideoIngestor = Depends(get_ingestor),
    pipeline: SurveillancePipeline = Depends(get_pipeline),
):
    """Register staged reference videos and process their paired windows JSON."""
    reference_dir = paths.root_dir / "notebooks" / "directjsonreference"
    video_dirs = (
        paths.uploads_dir / "demo",
        reference_dir / "video",
        reference_dir / "videos",
    )
    output_json_dir = reference_dir / "pipeline_outputs"

    imported = []
    seen_video_ids = set()
    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue
        for video_path in sorted(video_dir.iterdir()):
            if not video_path.is_file() or video_path.suffix.lower() not in {
                ".mp4", ".avi", ".mov", ".mkv"
            }:
                continue

            json_path = video_path.with_suffix(".json")
            if not json_path.is_file():
                json_path = output_json_dir / f"{video_path.stem}.json"
            if not json_path.is_file():
                logger.warning("Skipping video without matching JSON: %s", video_path.name)
                continue
            if json_path.stat().st_size > 50 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"Reference JSON exceeds the 50 MB limit: {json_path.name}",
                )
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
                video_info = get_video_info(video_path)
                windows = validate_windows_payload(
                    payload, {video_path.stem}, video_info["duration_sec"]
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid video/JSON pair {video_path.name} + {json_path.name}: {exc}",
                ) from exc

            hasher = hashlib.md5()
            with video_path.open("rb") as video_file:
                hasher.update(video_file.read(1024 * 1024))
            video_id = f"vid_{hasher.hexdigest()[:12]}"
            if video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)

            record = store.get_video(video_id)
            if record is None or not Path(record.file_path).is_file():
                record = ingestor.ingest_video(
                    video_path, original_filename=video_path.name
                )

            if record.status not in {"completed", "processing"}:
                store.update_video_status(record.video_id, "processing")
                record.status = "processing"
                record.error_message = None
                background_tasks.add_task(
                    pipeline.process_video,
                    record.video_id,
                    windows,
                    "direct_json",
                )
            imported.append(record)

    return [to_response(video) for video in imported]


@router.get("/{video_id}", response_model=VideoResponse)
def get_video(video_id: str, store: MetadataStore = Depends(get_metadata_store)):
    """Get metadata for a specific video."""
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return to_response(video)


@router.delete("/{video_id}")
def delete_video(
    video_id: str,
    store: MetadataStore = Depends(get_metadata_store),
    vec_store: VectorStore = Depends(get_vector_store),
):
    """Delete a video and its derived app-managed data."""
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    if video.status == "processing":
        raise HTTPException(status_code=409, detail="Wait for video processing to finish before deleting it")

    events = store.get_events(video_id=video_id)
    remaining_events = [
        event for event in store.get_events() if event.video_id != video_id
    ]

    managed_files = [
        (Path(video.file_path), paths.uploads_dir),
        *((Path(event.clip_path), paths.clips_dir) for event in events if event.clip_path),
        (paths.tracks_dir / f"{video_id}_tracks.csv", paths.tracks_dir),
        (paths.windows_dir / f"{video_id}_windows.json", paths.windows_dir),
        (paths.outputs_dir / f"{video_id}_analysis.json", paths.outputs_dir),
    ]
    for file_path, managed_dir in managed_files:
        try:
            resolved_path = file_path.resolve()
            resolved_root = managed_dir.resolve()
            if resolved_path.is_relative_to(resolved_root) and resolved_path.is_file():
                resolved_path.unlink()
        except OSError as exc:
            logger.warning(f"Could not remove managed file {file_path}: {exc}")

    if not store.delete_video(video_id):
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    vec_store.rebuild(remaining_events)
    return {"video_id": video_id, "status": "deleted"}
