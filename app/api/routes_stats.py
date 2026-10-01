"""Surveillance dashboard metrics and statistics API endpoint."""

from typing import Dict, Any
from fastapi import APIRouter, Depends
from src.storage.metadata_store import MetadataStore
from app.api.deps import get_metadata_store

router = APIRouter(prefix="/stats", tags=["Statistics"])


@router.get("/dashboard")
def get_dashboard_metrics(store: MetadataStore = Depends(get_metadata_store)) -> Dict[str, Any]:
    """Retrieve aggregate surveillance metrics for system overview and dashboard visualization."""
    stats = store.get_dashboard_stats()
    return {
        "status": "healthy",
        **stats,
    }
