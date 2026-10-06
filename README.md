<<<<<<< HEAD
# Backend

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

# Frontend

```powershell
cd frontend
npm install
npm run dev
```
=======
# Vision-Language Surveillance Event Understanding

This repository is organized as a small monorepo while preserving the existing
surveillance pipeline and browser dashboard. The backend runs YOLO + ByteTrack,
temporal event processing, SQLite metadata storage, retrieval, and evidence
serving. The existing frontend is kept intact under `frontend/`.

## Repository layout

```text
server/
  app/       FastAPI application and API routes
  src/       pipeline, storage, retrieval, tracking, and utility code
  configs/   backend YAML configuration
  data/      runtime uploads, outputs, clips, and vector indexes
  tests/     backend tests
frontend/    existing browser dashboard assets
notebooks/   detection and anomaly research notebooks
models/      model weights and ML artifacts
```

Runtime data and SQLite state are created under `server/data/` and
`server/events.db`; they are ignored by Git. Model files belong under the root
`models/` hierarchy. Large model files should be stored with Git LFS or an
artifact store rather than committed directly.

## Backend

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r server/requirements.txt
python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open <http://127.0.0.1:8000/> for the dashboard or
<http://127.0.0.1:8000/docs> for the API.

Run tests from the repository root:

```powershell
$env:PYTHONPATH = "."
python -m pytest server/tests -q
```

## Frontend

The current dashboard is served directly by FastAPI from `frontend/`. No
frontend package manager is required for the preserved implementation. Start
the backend and open `/` or `/ui`.

## Processing workflow

1. Upload a video in the dashboard.
2. Run the local Stage 1b + Stage 2 pipeline.
3. Import the generated Stage 4 `events.json`.
4. Search events and open evidence clips from the conversation interface.

The restructuring changes only file locations, imports, and path
configuration; processing algorithms, API routes, database schema, and UI
behavior remain unchanged.
>>>>>>> db838f6 (Refactor project into production monorepo structure)
