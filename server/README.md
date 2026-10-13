# FastAPI application

See the repository [README](../README.md) for setup. The service provides both a complete video-only pipeline and a direct video + notebook windows JSON workflow.

The API is started from the repository root with `python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8080 --reload`. Interactive endpoint documentation is available at `/docs`.

## Runtime device and benchmarking

The backend selects a single runtime device from `server/configs/config.yaml`:

```yaml
runtime:
  device: auto   # auto, cuda, or cpu
  use_cuda: true
  half_precision: true
  gpu_id: 0
```

`cuda` fails clearly when the installed PyTorch build cannot access CUDA. `auto`
uses CUDA only when it is available; otherwise it uses CPU. YOLO and the
Stage 1b tracker reuse their loaded model and use inference-only execution.
FP16 is enabled only for CUDA. Qwen2.5-VL remains conservative on GPUs with
less than 6 GB VRAM and uses CPU/offload behavior rather than risking an OOM.

Run a reproducible benchmark with:

```powershell
python scripts/benchmark_pipeline.py path\to\video.mp4
```

The pipeline response and logs include stage timings, total processing time,
device, and real-time factor. No CPU/CUDA speedup is reported unless both
runs are actually measured.
## Phase 2 semantic retrieval

Event descriptions are embedded locally with
`sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS inner-product
index. The index is only a semantic accelerator; event records and all
filtering remain in SQLite. The model is downloaded by Sentence Transformers
to its normal local Hugging Face cache on first use.

## Video understanding workflows

Install `server/requirements.txt` before running the API. The full workflow
samples video at `video.sample_fps` (4 fps by default), passes those frames
through the configured YOLO checkpoint and ByteTrack, builds sliding temporal
windows, runs Qwen2.5-VL on five evenly spaced frames per window, then runs the
configured anomaly detector. `stage1b.frame_stride: 0` derives the YOLO stride
from the video's native FPS and `video.sample_fps`; set a positive stride to
override it. Track-buffer length is adjusted for the sampled cadence.
The YOLO model and the lazily loaded Qwen model are reused across videos; YOLO
tracking is serialized so tracker state cannot leak between concurrent videos.
The default VLM is `Qwen/Qwen2.5-VL-3B-Instruct` to limit resource use; its
weights are downloaded from Hugging Face on first inference if not cached.
Insufficient VRAM can still cause CPU/RAM offloading, so monitor memory during
the first run.
When the YOLO checkpoint is missing or is only a Git LFS pointer, the configured
OpenCV motion + local ByteTrack fallback is used and identified in the analysis
JSON's `tracking_backend`.

In the UI, **Add footage** adds a video, then **Run full pipeline** starts all
stages. **Video + JSON** accepts a video and a matching Stage 2/Stage 4
notebook JSON (`{"windows":[...]}`), skips YOLO and ByteTrack, and reuses any
events already present in each window. Windows without events are sent through
Qwen before anomaly scoring. The JSON `video_id` must match the video filename
stem; event track IDs and relative/absolute timestamps are validated.

Both workflows write `server/data/outputs/<video_id>_analysis.json` with the
notebook-compatible `windows` records plus a video-level `anomaly_summary`.
Download it from `GET /api/pipeline/output/{video_id}` or the completed video
card. If Qwen's Python dependencies are unavailable, only grounded
motion-rule candidates are retained and marked unverified. The anomaly module
reports `Unassessed` when its configured classifier and feature extractor are
not available; the JSON records model availability rather than inventing a
score.

The default index files are:

```text
server/data/vectors/events.faiss
server/data/vectors/events.mapping.json
```

New pipeline/import events are indexed automatically. Deleting a video rebuilds
the index from the remaining SQLite events. To repair or rebuild it manually:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8080/api/query/rebuild-index
```

Search through `POST /api/query` with `query`, `top_k`, `min_score`, and
optional `video_id`. Natural-language event, anomaly, entity, and time filters
are parsed and applied against SQLite before hybrid ranking.

The chat UI sends the last eight conversation turns with each query. When
verified event records match, the local Qwen2.5-VL model answers using those
records as its only footage evidence and cites their event IDs. Chat reuses the
pipeline's lazily loaded Qwen instance rather than loading a second copy. If the
model cannot load or returns missing/unknown citations, the API returns the
deterministic evidence summary instead. A first chat query with matching events
can therefore trigger local model loading; it may use substantial RAM/VRAM.

On UI startup, the frontend asks the backend to import staged reference pairs
from `server\data\uploads\demo` and
`notebooks\directjsonreference\video` (also accepts `videos`). Each video is
paired with a same-name JSON sidecar or the corresponding file in
`notebooks\directjsonreference\pipeline_outputs`. Valid pairs are registered
and processed through direct-JSON mode, then appear under **FOOTAGE**. Keep
large local videos in one of these ignored/local data folders rather than
committing them.
