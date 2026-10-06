"""FastAPI entrypoint for Vision-Language Surveillance Event Understanding."""

from pathlib import Path
from contextlib import asynccontextmanager
import shutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from server.src.utils.paths import paths
from server.src.utils.logger import logger
from server.src.config.loader import config_loader
from server.src.storage.metadata_store import metadata_store
from server.src.storage.vector_store import vector_store
from server.app.api.routes_videos import router as videos_router
from server.app.api.routes_pipeline import router as pipeline_router
from server.app.api.routes_events import router as events_router
from server.app.api.routes_query import router as query_router
from server.app.api.routes_evidence import router as evidence_router, records_router as evidence_records_router
from server.app.api.routes_stats import router as stats_router
from server.app.api.routes_chats import router as chats_router, search_router


def clear_local_app_state() -> None:
    """Clear only app-managed runtime data; never touch model/source directories."""
    data_root = paths.data_dir.resolve()
    for managed_dir in (paths.uploads_dir, paths.outputs_dir, paths.vectors_dir):
        target = managed_dir.resolve()
        if not target.is_relative_to(data_root):
            raise RuntimeError(f"Refusing to clear path outside app data directory: {target}")
        for child in target.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        target.mkdir(parents=True, exist_ok=True)
    metadata_store.clear_all()
    vector_store.clear()
    logger.info("Cleared previous local videos, events, clips, pipeline outputs, and search index.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Initializing Surveillance Event Understanding API...")
    paths.ensure_directories()
    if config_loader.get("server.reset_data_on_startup", False):
        clear_local_app_state()
    yield
    logger.info("Surveillance Event Understanding API shut down.")


app = FastAPI(
    title="Vision-Language Surveillance Event Understanding API",
    description="Backend API integrating YOLO detection, ByteTrack tracking, temporal windowing, "
                "Qwen2.5-VL vision-language understanding, UCF-Crime anomaly detection, and "
                "evidence-based video retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware for frontend integration (React, Vite, Next.js)
cors_origins = config_loader.get("server.cors_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers under /api
app.include_router(videos_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api")
app.include_router(events_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(evidence_router, prefix="/api")
app.include_router(evidence_records_router, prefix="/api")
app.include_router(stats_router, prefix="/api")
app.include_router(chats_router, prefix="/api")
app.include_router(search_router, prefix="/api")

from fastapi import Request
from fastapi.responses import FileResponse

# Mount static file directories for direct media and UI serving
paths.ensure_directories()
frontend_dir = paths.root_dir / "frontend" / "dist"
frontend_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static/uploads", StaticFiles(directory=str(paths.uploads_dir)), name="uploads")
app.mount("/static/clips", StaticFiles(directory=str(paths.clips_dir)), name="clips")
app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")


@app.get("/")
def root(request: Request):
    """Root status endpoint; serves frontend interface for browser requests."""
    accept = request.headers.get("accept", "")
    index_file = frontend_dir / "index.html"
    if "text/html" in accept and index_file.exists():
        if config_loader.get("server.reset_data_on_dashboard_open", False):
            clear_local_app_state()
        return FileResponse(index_file)

    return {
        "service": "Vision-Language Surveillance Event Understanding API",
        "status": "online",
        "ui": "/ui",
        "documentation": "/docs",
        "version": "1.0.0",
    }


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": paths.db_path.exists(),
        "vector_store": paths.vectors_dir.exists(),
    }


if __name__ == "__main__":
    import uvicorn

    host = config_loader.get("server.host", "0.0.0.0")
    port = config_loader.get("server.port", 8000)
    uvicorn.run("server.app.main:app", host=host, port=port, reload=True)
