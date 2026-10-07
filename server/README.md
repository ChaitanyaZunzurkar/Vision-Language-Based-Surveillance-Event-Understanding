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
