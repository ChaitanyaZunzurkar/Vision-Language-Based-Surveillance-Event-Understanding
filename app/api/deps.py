"""FastAPI dependency injection utilities."""

from src.storage.metadata_store import MetadataStore, metadata_store
from src.storage.vector_store import VectorStore, vector_store
from src.pipeline.pipeline import SurveillancePipeline, surveillance_pipeline
from src.retrieval.retriever import SurveillanceRetriever, retriever
from src.pipeline.ingestion import VideoIngestor, video_ingestor


def get_metadata_store() -> MetadataStore:
    return metadata_store


def get_vector_store() -> VectorStore:
    return vector_store


def get_pipeline() -> SurveillancePipeline:
    return surveillance_pipeline


def get_retriever() -> SurveillanceRetriever:
    return retriever


def get_ingestor() -> VideoIngestor:
    return video_ingestor
