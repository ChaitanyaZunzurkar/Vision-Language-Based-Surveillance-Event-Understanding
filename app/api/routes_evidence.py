"""Media streaming and evidence clip retrieval API endpoints."""

from pathlib import Path
from typing import Optional
import os
from fastapi import APIRouter, HTTPException, Header, Response, status
from fastapi.responses import FileResponse, StreamingResponse
from src.utils.paths import paths
from src.utils.logger import logger

router = APIRouter(prefix="/media", tags=["Media"])


def stream_video_file(file_path: Path, range_header: Optional[str] = None):
    """Stream video file with support for HTTP 206 Partial Content (seeking/scrubbing)."""
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Requested media file not found")

    file_size = file_path.stat().st_size
    content_type = "video/mp4"

    if range_header:
        # Parse range header: "bytes=start-end"
        try:
            byte_range = range_header.replace("bytes=", "").split("-")
            start = int(byte_range[0]) if byte_range[0] else 0
            end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
            start = max(0, start)
            end = min(file_size - 1, end)
            content_length = (end - start) + 1

            def chunk_generator():
                with open(file_path, "rb") as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk_size = min(remaining, 1024 * 512)  # 512KB chunks
                        data = f.read(chunk_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Type": content_type,
            }
            return StreamingResponse(
                chunk_generator(),
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                headers=headers,
            )
        except Exception as e:
            logger.warning(f"Error handling range header '{range_header}': {e}")

    return FileResponse(file_path, media_type=content_type)


@router.get("/video/{filename}")
def get_uploaded_video(filename: str, range: Optional[str] = Header(None)):
    """Stream raw surveillance video from uploads directory."""
    # Prevent path traversal
    safe_name = Path(filename).name
    file_path = paths.uploads_dir / safe_name
    return stream_video_file(file_path, range)


@router.get("/clip/{filename}")
def get_evidence_clip(filename: str, range: Optional[str] = Header(None)):
    """Stream cut evidence video clip for direct playback and verification."""
    safe_name = Path(filename).name
    file_path = paths.clips_dir / safe_name
    return stream_video_file(file_path, range)
