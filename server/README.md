# FastAPI application

See the repository [README](../README.md) for setup, the Stage 1b/Stage 2 workflow, and Stage 4 JSON import instructions.

The API is started from the repository root with `python -m uvicorn server.app.main:app --reload`. Interactive endpoint documentation is available at `/docs`.

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

The default index files are:

```text
server/data/vectors/events.faiss
server/data/vectors/events.mapping.json
```

New pipeline/import events are indexed automatically. Deleting a video rebuilds
the index from the remaining SQLite events. To repair or rebuild it manually:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/query/rebuild-index
```

Search through `POST /api/query` with `query`, `top_k`, `min_score`, and
optional `video_id`. Natural-language event, anomaly, entity, and time filters
are parsed and applied against SQLite before hybrid ranking.
