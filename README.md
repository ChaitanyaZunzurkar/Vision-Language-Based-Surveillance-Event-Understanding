# Vision-Language Surveillance Event Understanding

FastAPI and browser dashboard for the available surveillance pipeline stages. The app runs YOLO + ByteTrack (Stage 1b) and the project's Stage 2 temporal windowing code. Since no Qwen model is present in the app, Stage 4 output is imported from a generated `events.json`; the app does not fabricate descriptions. Anomaly classification is shown as unassessed until the classifier and its raw-video feature extractor are available.

## Run locally (PowerShell)

Build the React interface before starting the API server:

```powershell
cd frontend
npm install
npm run build
cd ..
```

For frontend development with hot reload, keep the API running on port 8000 and run `npm run dev` from `frontend/`. Vite serves the interface on port 5173 and proxies API requests to FastAPI.

The 114 MB YOLO checkpoint is stored with Git LFS. On a fresh clone, install/initialize Git LFS and fetch the model before running the app:

```powershell
git lfs install
git lfs pull
```

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open <http://127.0.0.1:8000/> for the dashboard or <http://127.0.0.1:8000/docs> for the API.

## Available workflow

1. Upload a video. Then choose Path A in the dashboard to run local Stage 1b + 2, or Path B to skip local YOLO and import a final Stage 4 `events.json`.
2. Wait for the video status to become **awaiting_events**. The app writes `{video_id}_tracks.csv` and `{video_id}_windows.json` under `data/outputs/`.
3. If desired, run the existing Stage 4 notebook against that video's windows JSON and source video. In the dashboard, select the same video, choose the resulting `events.json`, and import it. Imported events are saved to SQLite, indexed for natural-language retrieval, and given evidence clips where timestamps are valid.

The importer accepts either the VISTA format (`{"windows": [{..., "events": [...]}]}`) or a flat `{"events": [...]}` list. For windowed output it accounts for event times expressed relative to that window, and checks source video IDs against the selected video. Imported event records retain source track and zone fields in metadata.

## Notes

- The dashboard clears app data on backend startup and whenever its root page (`/`) is opened. This removes uploaded videos, events, clips, generated tracks/windows, and the search index. The behavior is controlled by `server.reset_data_on_startup` and `server.reset_data_on_dashboard_open` in `configs/config.yaml`; set either to `false` to keep that data at that point. Model resources are outside the cleared paths.
- Stage 1b code, ByteTrack configuration, class configuration, and YOLO weights are stored in project-owned paths. `VISTA/` is ignored and can be removed without breaking the app. The YOLO checkpoint is tracked with Git LFS.
- Qwen configuration and its existing interface remain in the project, but the app does not invoke a missing model.
- The current anomaly notebook uses precomputed VideoMAE features. No feature extractor or trained classifier/scaler files are present here, so the app reports anomaly status as **Unassessed**.
- This is a single-user local dashboard; authentication and production deployment configuration are not included.
