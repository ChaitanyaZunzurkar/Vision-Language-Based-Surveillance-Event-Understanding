"""Video ingestion, validation, and metadata extraction."""

from pathlib import Path
from typing import Dict, Any, Optional
import shutil
import hashlib
import uuid
from server.src.utils.paths import paths
from server.src.utils.logger import logger
from server.src.utils.video_utils import get_video_info
from server.src.storage.schemas import VideoRecord
from server.src.storage.metadata_store import MetadataStore, metadata_store


class VideoIngestor:
    """Ingests and validates surveillance video files into the storage system."""

    def __init__(
        self,
        uploads_dir: Optional[Path] = None,
        store: Optional[MetadataStore] = None,
    ):
        self.uploads_dir = Path(uploads_dir) if uploads_dir else paths.uploads_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or metadata_store

    def ingest_video(
        self, source_path: Path | str, original_filename: Optional[str] = None
    ) -> VideoRecord:
        """Ingest a video file, extract metadata, and register in database.

        Args:
            source_path: Path to the source video file.
            original_filename: Optional display name for the file.

        Returns:
            VideoRecord instance.
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Video file does not exist: {src}")

        filename = original_filename or src.name

        # Generate deterministic video_id from file contents
        hasher = hashlib.md5()
        with open(src, "rb") as f:
            # Read first 1MB to quickly hash
            hasher.update(f.read(1024 * 1024))
        video_id = f"vid_{hasher.hexdigest()[:12]}"

        # Copy to uploads directory if not already there
        target_path = self.uploads_dir / f"{video_id}_{filename}"
        if not target_path.exists() or target_path.resolve() != src.resolve():
            shutil.copy2(src, target_path)

        # Extract video metadata
        meta = get_video_info(target_path)

        record = VideoRecord(
            video_id=video_id,
            filename=filename,
            file_path=str(target_path.resolve()),
            duration_sec=meta["duration_sec"],
            fps=meta["fps"],
            frame_count=meta["frame_count"],
            resolution=meta["resolution"],
            status="pending",
        )

        self.store.save_video(record)
        logger.info(
            f"Ingested video '{filename}' as {video_id} ({meta['duration_sec']}s, {meta['resolution']})"
        )
        return record


# Global video ingestor instance
video_ingestor = VideoIngestor()
