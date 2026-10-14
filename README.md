# Surveillance Video Event Understanding

This application analyzes surveillance videos and presents detected events in a
web UI. It includes two processing workflows:

1. **Full video pipeline:** sampled video frames → YOLO detection → ByteTrack →
   temporal windows → Qwen2.5-VL event understanding → anomaly scoring → one
   analysis JSON file.
2. **Video + JSON pipeline:** upload a video with a precomputed notebook-format
   windows JSON file. This validates and uses the supplied detections/tracks and
   events, skipping YOLO and ByteTrack.

## Requirements

- Windows 10/11 and PowerShell
- Python 3.10 or newer
- Node.js 18 or newer (includes npm)
- Git, if cloning the repository
- Internet access for initial Python package installation and model downloads

For Qwen inference, use a machine with sufficient RAM/VRAM for the selected
model and PyTorch configuration. CPU inference is supported but can be very
slow. Model weights are large and are downloaded only when required, not during
the normal app startup.

## First-time setup (Windows)

Run these steps in **PowerShell**, starting from the repository root—the folder
containing `server`, `frontend`, and this `README.md`. If your terminal is
currently inside `server` or `frontend`, return to the root first:

```powershell
cd C:\path\to\Vision-Language-Based-Surveillance-Event-Understanding
```

Create the Python environment inside `server` and install all backend
requirements into it. This location matters: always use this exact interpreter
to install packages and launch the backend.

```powershell
py -3 -m venv server\.venv
.\server\.venv\Scripts\python.exe -m pip install --upgrade pip
.\server\.venv\Scripts\python.exe -m pip install -r server\requirements.txt
```

The requirements include `uvicorn`, `faiss-cpu`, and `sentence-transformers`
used for semantic search. Install the frontend's Node packages once:

```powershell
cd frontend
npm install
cd ..
```

At this point, the first-time installation is complete. Python packages live in
`server\.venv`; frontend packages live in `frontend\node_modules`. The Qwen and
sentence-transformer model files are downloaded separately from Hugging Face
the first time their respective features need them.

To enable faster Groq chat with local Qwen fallback, copy
`server\.env.example` to `server\.env` and insert your key in `GROQ_API_KEY`.
The file is ignored by Git and loaded by the backend at startup. Groq receives
retrieved event records and recent chat turns; leave `CHAT_PROVIDER=qwen` (or
unset it) to keep chat fully local.

## Normal startup (Windows)

Each time you want to run the app, open **two PowerShell terminals**. In both,
start at the repository root (the folder containing `server` and `frontend`).
Do not activate a different Python environment; the backend command below uses
`server\.venv` directly.

**Terminal 1 — backend** (keep this terminal open):

```powershell
.\server\.venv\Scripts\python.exe -m uvicorn server.app.main:app --host 127.0.0.1 --port 8080 --reload
```

Wait until Uvicorn reports that startup is complete.

**Terminal 2 — frontend** (keep this terminal open):

```powershell
cd frontend
npm run dev -- --host 127.0.0.1
```

Open the frontend at <http://127.0.0.1:5173/ui/>. API documentation is at
<http://127.0.0.1:8080/docs>. The frontend development server forwards API
requests to the backend on port 8080. Keep both terminals running while using
the app; press `Ctrl+C` in each terminal to stop the corresponding server.

### Optional: launch both services with one command

After first-time setup, this PowerShell command from the repository root opens
the backend and frontend in separate windows:

```powershell
$root = (Get-Location).Path; Start-Process -FilePath "$root\server\.venv\Scripts\python.exe" -ArgumentList @('-m','uvicorn','server.app.main:app','--host','127.0.0.1','--port','8080','--reload') -WorkingDirectory $root; Start-Process -FilePath 'powershell' -WorkingDirectory (Join-Path $root 'frontend') -ArgumentList '-NoExit','-Command','npm run dev -- --host 127.0.0.1'
```

### If the retrieval dependency error appears

If the UI or API reports:

> Semantic retrieval requires faiss and sentence-transformers. Install server/requirements.txt.

install the requirements into the backend environment (not a global Python or
an unrelated `.venv`) from the repository root:

```powershell
.\server\.venv\Scripts\python.exe -m pip install -r server\requirements.txt
```

Then stop and restart the backend using the command in **Terminal 1**. Confirm
the packages import from the same environment with:

```powershell
.\server\.venv\Scripts\python.exe -c "import faiss, sentence_transformers; print('retrieval dependencies OK')"
```

If `server\.venv\Scripts\python.exe` is not found, create the environment
first using the first-time setup instructions above. If `server\.venv` already
exists and you installed packages into it, do not create a new `.venv` at the
repository root or launch the API with another Python executable.

### Start either service separately

Backend, from the repository root:

```powershell
.\server\.venv\Scripts\python.exe -m uvicorn server.app.main:app --host 127.0.0.1 --port 8080 --reload
```

Frontend, from the repository root:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1
```

## Using the web app

### Full video analysis

1. Open the web app and choose **Add footage** to upload a video.
2. Select the uploaded video and choose **Run full pipeline**.
3. Processing runs in the backend. When it completes, use **Download analysis
   JSON** on the video card.

The full pipeline samples frames rather than decoding every frame for object
detection. It sends detections to tracking, summarizes tracks in temporal
windows, and passes representative frames and track information to the VLM.

### Precomputed video + JSON

1. Choose **Video + JSON**.
2. Select the video and its corresponding Stage 2/Stage 4 notebook JSON.
3. Submit the pair. The backend validates the JSON against the video and
   processes its windows without running YOLO or ByteTrack.
4. Download the resulting analysis JSON when processing completes.

The input file must have a top-level `windows` array. Each window should include
`video_id`, `camera_id`, a `[start, end]` `window`, and a `tracks` array.
Notebook-produced `events` are optional. If present, their track IDs and
timestamps are validated and the events are reused; windows without events
require VLM inference. The input `video_id` must correspond to the uploaded
video filename stem. Event `start`/`end` values are relative to their window;
`abs_start`/`abs_end` are supported for absolute timestamps.

## Qwen2.5-VL: model download and operation

**Yes. The application downloads Qwen model files automatically on first use**
if they are not already present in the local Hugging Face cache. The configured
default is `Qwen/Qwen2.5-VL-3B-Instruct`, loaded with Transformers
`from_pretrained`. The download is lazy: app startup alone does not load or
download Qwen. The first video window that needs generated events triggers model
loading and may require a long wait and several gigabytes of disk, RAM, and GPU
memory. The 3B model is selected to reduce resource use compared with the
previous 7B model, but it can still consume significant RAM and may offload
weights when GPU VRAM is insufficient.
Subsequent runs reuse the in-process model and cached files.

Qwen inference can be skipped for direct-upload windows that already contain
events. Windows without supplied events still need the model to generate them.
Hugging Face model files are stored in the normal Hugging Face cache, not in the
repository. Ensure the machine has network access on the first inference. No
Hugging Face login is normally needed for this public model.

Device selection is controlled by `runtime.device` and `vlm.device` in
`server/configs/config.yaml`. With `auto`, the app chooses an available
accelerator when the installed PyTorch build supports it; otherwise it can run
on CPU. GPU loading attempts 4-bit quantization when `bitsandbytes` is
available. On Windows that optional package may not be available, in which case
Qwen falls back to half precision on GPU. If memory is insufficient, reduce
other GPU workloads or use a machine with more memory. On a 6 GB VRAM laptop GPU,
do not assume the 3B model will fit wholly in VRAM; monitor memory during the
first inference.

If the Qwen/PyTorch/Transformers runtime cannot be imported, the application
keeps grounded motion-rule candidates and marks them **unverified** rather than
claiming they were generated by Qwen. Other model-loading or inference failures
are reported by the processing pipeline.

## Semantic retrieval models

Semantic search uses `faiss-cpu` to index event vectors and
`sentence-transformers/all-MiniLM-L6-v2` to create embeddings. The embedding
model is downloaded from Hugging Face on first use if it is not cached already.
These are separate from Qwen and are used for event search, not VLM event
generation.

The vector index is kept under `server/data/vectors/`; event metadata remains in
SQLite. If a query fails with the missing FAISS/Sentence Transformers message,
install the backend requirements using the **same `server\.venv` Python executable**
used to start Uvicorn, then restart the backend.

## Configuration and model files

Main settings are in `server/configs/config.yaml`:

- `video.sample_fps`: target object-detection sampling cadence (4 FPS by
  default).
- `stage1b.frame_stride`: explicit source-frame stride; `0` derives the stride
  from source FPS and `video.sample_fps`.
- `runtime.device`: shared runtime preference (`auto`, `cuda`, or `cpu`).
- `temporal`: temporal window size, overlap/stride, and motion-rule thresholds.
- `vlm`: Qwen model ID, representative frames per window, frame resize limit,
  generation tokens, and retry count.
- `retrieval`: embedding model and search defaults.

YOLO weights are loaded from paths configured in the YAML files under
`server/configs/`. A checkpoint represented by a Git LFS pointer is not usable
until the actual weights are hydrated. When the configured YOLO model is
missing, the configured lightweight OpenCV motion + local ByteTrack fallback is
used and identified in the output's `tracking_backend`.

The repository's anomaly detector requires a classifier and a compatible
feature extractor. If either is unavailable, clip scores are labeled
**Unassessed** with `model_available: false`; this should not be interpreted as
a normal-video prediction.

## Output files and API

Completed analysis JSON files are written to:

```text
server/data/outputs/<video_id>_analysis.json
```

The final JSON contains notebook-compatible `windows`, video-level `summary`,
and `anomaly_summary` data. It preserves event timestamps and model/backend
availability information. The completed video card provides a download link.
The corresponding API endpoint is:

```text
GET http://127.0.0.1:8080/api/pipeline/output/{video_id}
```

Other useful endpoints include:

- `POST /api/videos/upload` — upload a video for full-pipeline processing.
- `POST /api/videos/upload-direct` — upload a video and its precomputed windows
  JSON.
- `POST /api/pipeline/run/{video_id}` — start processing an existing upload.
- `GET /api/pipeline/status/{video_id}` — read processing status.
- `POST /api/query` — search processed events.
- `POST /api/query/rebuild-index` — rebuild the semantic search index.

See <http://127.0.0.1:8080/docs> for the complete interactive API reference.

## Data locations

- Uploaded videos: `server/data/uploads/`
- Final analysis JSON: `server/data/outputs/`
- Generated track/window data and evidence clips: `server/data/outputs/`
- Event database: `server/events.db` (configured paths can change this)
- Semantic retrieval index: `server/data/vectors/`
- YOLO model files: paths configured under `server/configs/` (some project
  checkpoints are under `models/` or `weights/`)

Generated data and model caches can be large. Keep the database and output
directories if you need prior analysis results; do not remove them as part of
routine setup.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `Semantic retrieval requires faiss and sentence-transformers` | Run `.\server\.venv\Scripts\python.exe -m pip install -r server\requirements.txt`, then restart the API. Confirm the server is launched with `server\.venv` Python. |
| `No module named uvicorn` | Install `server\requirements.txt` into `server\.venv` and use the documented startup command. |
| First Qwen inference is slow | The first run may download Qwen weights and initialize the model; allow network access and disk space. Later runs use the local cache. |
| Qwen imports are unavailable | Install the project requirements and a PyTorch build appropriate for the machine. The application will otherwise mark motion-rule output as unverified. |
| CUDA is unavailable | Install a PyTorch build compatible with the installed CUDA/driver, or set `runtime.device: "cpu"` in `server/configs/config.yaml`. |
| YOLO weights cannot be loaded | Verify the configured weights path and ensure the Git LFS object, not just its pointer file, is present. |
| Frontend cannot reach the API | Start the backend on port 8080 and the Vite frontend on port 5173; use the Vite URL shown above. |
| Analysis says anomaly is `Unassessed` | The anomaly classifier and feature extractor are not currently a compatible loaded pair. This is an explicit unavailable state, not a normal/anomaly classification. |

## Development checks

From the repository root:

```powershell
.\server\.venv\Scripts\python.exe -m pytest server\tests -q
Push-Location frontend
npm run build
Pop-Location
```
