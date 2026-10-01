# Vision-Language Surveillance Backend API

The FastAPI backend for the **Vision-Language-Based Surveillance Event Understanding** project. It integrates YOLO object detection, ByteTrack multi-object tracking, temporal windowing, Vision-Language Model (Qwen2.5-VL) understanding, UCF-Crime 14-category anomaly detection, evidence video clipping, and retrieval-augmented generation (RAG) querying.

---

## Architecture Overview

```
                                 Raw Surveillance Video
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │      Video Ingestion        │
                            │  (Format, Hash, Metadata)   │
                            └──────────────┬──────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │    Stage 1b: Perception     │
                            │  (YOLOv11 + ByteTrack MOT)  │
                            └──────────────┬──────────────┘
                                           │ tracks.csv
                                           ▼
                            ┌─────────────────────────────┐
                            │ Stage 2: Temporal Windower  │
                            │ (Overlapping Sliding Windows│
                            │  & Candidate Event Rules)   │
                            └──────────────┬──────────────┘
                                           │ windows.json
                                           ▼
                            ┌─────────────────────────────┐
                            │   Stage 4: VLM Grounding    │
                            │  (Qwen2.5-VL Event Semantic │
                            │     Description Engine)     │
                            └──────────────┬──────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
       ┌───────────────────────────┐               ┌───────────────────────────┐
       │   UCF-Crime Anomaly Head  │               │   Evidence Clip Cutter    │
       │  (14-Class Categorization │               │(Buffer-Padded Video Segs) │
       │     & Anomaly Scoring)    │               └─────────────┬─────────────┘
       └─────────────┬─────────────┘                             │
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │       Dual Data Store       │
                            │  SQLite DB  + Vector Index  │
                            │ (Metadata)     (Embeddings) │
                            └──────────────┬──────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │  Natural-Language Retriever │
                            │ (Hybrid Semantic + SQL RAG) │
                            └─────────────────────────────┘
```

---

## API Endpoints

### 1. Surveillance Videos
- `POST /api/videos/upload` - Upload surveillance video (`.mp4`, `.avi`, `.mov`, `.mkv`), extracts metadata (resolution, fps, duration), and registers it. Optionally set `auto_run=true` to trigger processing immediately.
- `GET /api/videos` - List all registered videos with their current status (`pending`, `processing`, `completed`, `failed`).
- `GET /api/videos/{video_id}` - Retrieve details for a specific video.

### 2. Pipeline Execution
- `POST /api/pipeline/run/{video_id}` - Trigger the multi-stage surveillance understanding pipeline (`run_in_background=true` by default).
- `GET /api/pipeline/status/{video_id}` - Check execution progress and total events generated.

### 3. Structured Events
- `GET /api/events` - Filter extracted events by:
  - `video_id`: Specific video
  - `event_type`: `enter`, `exit`, `loiter`, `carry`, `place`, `leave`, `stationary`, `interaction`
  - `anomaly_category`: `Burglary`, `Robbery`, `Fighting`, `Normal`, etc.
  - `time_start` / `time_end`: Timestamp constraints in seconds
  - `min_confidence`: Minimum confidence threshold
- `GET /api/events/{event_id}` - Retrieve detailed event record with attached evidence clip URL.

### 4. Natural-Language Querying & Retrieval (RAG)
- `POST /api/query` - Natural-language query interface. Combines vector cosine similarity search with structured metadata filtering to return evidence-grounded reports with playback links.
  ```json
  {
    "query": "Show clips where a person placed a bag or luggage",
    "top_k": 5,
    "min_score": 0.15
  }
  ```

### 5. Media Streaming & Playback
- `GET /api/media/video/{filename}` - Stream raw surveillance videos (supports HTTP 206 Partial Content for video seeking).
- `GET /api/media/clip/{filename}` - Stream cut evidence video clips for instant visual verification.

### 6. Analytics & Statistics
- `GET /api/stats/dashboard` - High-level metrics for dashboard visualization: total videos, total events, event type distribution, and anomaly counts.
- `GET /api/health` - Health check endpoint.

---

## Running Locally

Activate the project virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

Start the FastAPI backend with Uvicorn:
```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Running Tests

Run the complete automated test suite:
```powershell
.\.venv\Scripts\pytest.exe tests/
```
