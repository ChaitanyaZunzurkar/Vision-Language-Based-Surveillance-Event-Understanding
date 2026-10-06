"""SQLite metadata store for surveillance videos, events, tracks, and anomalies."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import sqlite3
from datetime import datetime
from server.src.utils.paths import paths
from server.src.utils.logger import logger
from server.src.storage.schemas import VideoRecord, EventRecord, AnomalyRecord, TrackRecord


from contextlib import contextmanager


class MetadataStore:
    """Manages persistent SQLite storage of surveillance metadata."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path) if db_path else paths.db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    fps REAL NOT NULL,
                    frame_count INTEGER NOT NULL,
                    resolution TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    entity_ids_json TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    anomaly_category TEXT NOT NULL DEFAULT 'Normal',
                    anomaly_confidence REAL NOT NULL DEFAULT 0.0,
                    clip_path TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id) ON DELETE CASCADE
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS anomalies (
                    anomaly_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_anomaly INTEGER NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id) ON DELETE CASCADE
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    total_observations INTEGER NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    video_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id) ON DELETE SET NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS message_evidence (
                    message_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    PRIMARY KEY (message_id, event_id),
                    FOREIGN KEY (message_id) REFERENCES messages (id) ON DELETE CASCADE,
                    FOREIGN KEY (event_id) REFERENCES events (event_id) ON DELETE CASCADE
                )
                """
            )

            conn.commit()
            logger.info(f"Initialized SQLite database at {self.db_path}")

    def clear_all(self) -> None:
        """Remove persisted app records while preserving the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events")
            cursor.execute("DELETE FROM anomalies")
            cursor.execute("DELETE FROM tracks")
            cursor.execute("DELETE FROM videos")
            conn.commit()

    def delete_video(self, video_id: str) -> bool:
        """Delete a video and all associated events, tracks, and anomaly records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events WHERE video_id = ?", (video_id,))
            cursor.execute("DELETE FROM anomalies WHERE video_id = ?", (video_id,))
            cursor.execute("DELETE FROM tracks WHERE video_id = ?", (video_id,))
            cursor.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def save_video(self, record: VideoRecord) -> None:
        """Insert or replace a video record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO videos (
                    video_id, filename, file_path, duration_sec, fps, frame_count, resolution, status, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.video_id,
                    record.filename,
                    record.file_path,
                    record.duration_sec,
                    record.fps,
                    record.frame_count,
                    record.resolution,
                    record.status,
                    record.error_message,
                    record.created_at,
                ),
            )
            conn.commit()

    def update_video_status(
        self, video_id: str, status: str, error_message: Optional[str] = None
    ) -> None:
        """Update video processing status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE videos SET status = ?, error_message = ? WHERE video_id = ?",
                (status, error_message, video_id),
            )
            conn.commit()

    def get_video(self, video_id: str) -> Optional[VideoRecord]:
        """Retrieve video record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return VideoRecord(
                video_id=row["video_id"],
                filename=row["filename"],
                file_path=row["file_path"],
                duration_sec=row["duration_sec"],
                fps=row["fps"],
                frame_count=row["frame_count"],
                resolution=row["resolution"],
                status=row["status"],
                error_message=row["error_message"],
                created_at=row["created_at"],
            )

    def list_videos(self) -> List[VideoRecord]:
        """List all registered videos."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [
                VideoRecord(
                    video_id=r["video_id"],
                    filename=r["filename"],
                    file_path=r["file_path"],
                    duration_sec=r["duration_sec"],
                    fps=r["fps"],
                    frame_count=r["frame_count"],
                    resolution=r["resolution"],
                    status=r["status"],
                    error_message=r["error_message"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def insert_events(self, events: List[EventRecord]) -> None:
        """Insert batch of semantic events."""
        if not events:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for ev in events:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO events (
                        event_id, video_id, event_type, start_sec, end_sec, entity_ids_json, description,
                        confidence, anomaly_category, anomaly_confidence, clip_path, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ev.event_id,
                        ev.video_id,
                        ev.event_type,
                        ev.start_sec,
                        ev.end_sec,
                        json.dumps(ev.entity_ids),
                        ev.description,
                        ev.confidence,
                        ev.anomaly_category,
                        ev.anomaly_confidence,
                        ev.clip_path,
                        json.dumps(ev.metadata),
                        ev.created_at,
                    ),
                )
            conn.commit()

    def get_events(
        self,
        video_id: Optional[str] = None,
        event_type: Optional[str] = None,
        anomaly_category: Optional[str] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        min_confidence: float = 0.0,
    ) -> List[EventRecord]:
        """Query events with filtering criteria."""
        query = "SELECT * FROM events WHERE confidence >= ?"
        params: List[Any] = [min_confidence]

        if video_id:
            query += " AND video_id = ?"
            params.append(video_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.lower())
        if anomaly_category:
            query += " AND anomaly_category = ?"
            params.append(anomaly_category)
        if time_start is not None:
            query += " AND end_sec >= ?"
            params.append(time_start)
        if time_end is not None:
            query += " AND start_sec <= ?"
            params.append(time_end)

        query += " ORDER BY start_sec ASC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [
                EventRecord(
                    event_id=r["event_id"],
                    video_id=r["video_id"],
                    event_type=r["event_type"],
                    start_sec=r["start_sec"],
                    end_sec=r["end_sec"],
                    entity_ids=json.loads(r["entity_ids_json"]),
                    description=r["description"],
                    confidence=r["confidence"],
                    anomaly_category=r["anomaly_category"],
                    anomaly_confidence=r["anomaly_confidence"],
                    clip_path=r["clip_path"],
                    metadata=json.loads(r["metadata_json"] or "{}"),
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def get_event(self, event_id: str) -> Optional[EventRecord]:
        """Get single event by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
            r = cursor.fetchone()
            if not r:
                return None
            return EventRecord(
                event_id=r["event_id"],
                video_id=r["video_id"],
                event_type=r["event_type"],
                start_sec=r["start_sec"],
                end_sec=r["end_sec"],
                entity_ids=json.loads(r["entity_ids_json"]),
                description=r["description"],
                confidence=r["confidence"],
                anomaly_category=r["anomaly_category"],
                anomaly_confidence=r["anomaly_confidence"],
                clip_path=r["clip_path"],
                metadata=json.loads(r["metadata_json"] or "{}"),
                created_at=r["created_at"],
            )

    def insert_tracks(self, tracks: List[TrackRecord]) -> None:
        """Persist the track summaries emitted by the processing pipeline."""
        if not tracks:
            return
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO tracks (video_id, track_id, class_name, start_sec, end_sec, total_observations)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(t.video_id, t.track_id, t.class_name, t.start_sec, t.end_sec, t.total_observations) for t in tracks],
            )
            conn.commit()

    def search_tracks(self, video_id: Optional[str] = None, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT video_id, track_id, class_name, start_sec, end_sec, total_observations FROM tracks WHERE 1=1"
        params: List[Any] = []
        if video_id:
            query += " AND video_id = ?"
            params.append(video_id)
        if class_name:
            query += " AND lower(class_name) = lower(?)"
            params.append(class_name)
        query += " ORDER BY start_sec"
        with self._get_connection() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def search_anomalies(self, video_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM anomalies WHERE is_anomaly = 1"
        params: List[Any] = []
        if video_id:
            query += " AND video_id = ?"
            params.append(video_id)
        with self._get_connection() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def create_conversation(self, conversation_id: str, title: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, video_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, title, video_id, now, now),
            )
            conn.commit()
        return self.get_conversation(conversation_id)

    def list_conversations(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM conversations"
        params: List[Any] = []
        if search:
            query += " WHERE title LIKE ?"
            params.append(f"%{search}%")
        query += " ORDER BY updated_at DESC"
        with self._get_connection() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            return dict(row) if row else None

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            return cursor.rowcount > 0

    def add_message(self, message_id: str, conversation_id: str, role: str, content: str,
                    event_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, now),
            )
            for event_id in event_ids or []:
                conn.execute(
                    "INSERT OR IGNORE INTO message_evidence (message_id, event_id) VALUES (?, ?)",
                    (message_id, event_id),
                )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            conn.commit()
        return self.get_message(message_id)

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                """SELECT m.*, COALESCE(json_group_array(me.event_id) FILTER (WHERE me.event_id IS NOT NULL), '[]') AS evidence_ids
                   FROM messages m LEFT JOIN message_evidence me ON me.message_id = m.id
                   WHERE m.id = ? GROUP BY m.id""", (message_id,)
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["evidence_ids"] = json.loads(result["evidence_ids"])
            return result

    def list_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT m.*, COALESCE(json_group_array(me.event_id) FILTER (WHERE me.event_id IS NOT NULL), '[]') AS evidence_ids
                   FROM messages m LEFT JOIN message_evidence me ON me.message_id = m.id
                   WHERE m.conversation_id = ? GROUP BY m.id ORDER BY m.created_at""",
                (conversation_id,),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["evidence_ids"] = json.loads(item["evidence_ids"])
                result.append(item)
            return result

    def insert_anomalies(self, anomalies: List[AnomalyRecord]) -> None:
        """Insert batch of video anomaly segments."""
        if not anomalies:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for anom in anomalies:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO anomalies (
                        anomaly_id, video_id, category, confidence, is_anomaly, start_sec, end_sec
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        anom.anomaly_id,
                        anom.video_id,
                        anom.category,
                        anom.confidence,
                        1 if anom.is_anomaly else 0,
                        anom.start_sec,
                        anom.end_sec,
                    ),
                )
            conn.commit()

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Aggregate high-level metrics for dashboard."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as cnt FROM videos")
            total_videos = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM events")
            total_events = cursor.fetchone()["cnt"]

            cursor.execute(
                "SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type"
            )
            event_type_counts = {r["event_type"]: r["cnt"] for r in cursor.fetchall()}

            cursor.execute(
                "SELECT anomaly_category, COUNT(*) as cnt FROM events WHERE anomaly_category NOT IN ('Normal', 'Unassessed') GROUP BY anomaly_category"
            )
            anomaly_counts = {
                r["anomaly_category"]: r["cnt"] for r in cursor.fetchall()
            }

            return {
                "total_videos": total_videos,
                "total_events": total_events,
                "event_type_distribution": event_type_counts,
                "anomaly_distribution": anomaly_counts,
            }


# Global metadata store instance
metadata_store = MetadataStore()
