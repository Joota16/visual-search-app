"""Endpoints para comprobar el estado del sistema."""

from __future__ import annotations

import torch
from fastapi import APIRouter, Depends

from src.core.config import Settings
from src.services.qdrant_service import QdrantService
from src.api.dependencies import (
    get_qdrant_service,
    get_settings_from_app,
)


router = APIRouter(tags=["Estado"])


@router.get("/health")
def health() -> dict[str, str]:
    """Indica que el proceso de la API está activo."""
    return {
        "status": "ok",
        "service": "visual-search-api",
    }


@router.get("/ready")
def ready(
    settings: Settings = Depends(
        get_settings_from_app
    ),
    qdrant_service: QdrantService = Depends(
        get_qdrant_service
    ),
) -> dict:
    """Comprueba modelo, GPU, Qdrant y colección."""
    collection_info = (
        qdrant_service.client.get_collection(
            settings.qdrant_collection
        )
    )

    points_count = int(
        collection_info.points_count or 0
    )

    indexed_vectors = int(
        collection_info.indexed_vectors_count or 0
    )

    return {
        "status": "ready",
        "image_model": settings.image_model,
        "text_model": settings.text_model,
        "model": "CLIP multilingüe",
        "device": settings.device,
        "cuda_available": torch.cuda.is_available(),
        "collection": settings.qdrant_collection,
        "collection_status": str(
            collection_info.status
        ),
        "points_count": points_count,
        "indexed_vectors_count": indexed_vectors,
    }


@router.get("/api/v1/stats")
def stats(
    settings: Settings = Depends(
        get_settings_from_app
    ),
    qdrant_service: QdrantService = Depends(
        get_qdrant_service
    ),
) -> dict:
    """Devuelve estadísticas principales del buscador."""
    collection_info = (
        qdrant_service.client.get_collection(
            settings.qdrant_collection
        )
    )

    return {
        "dataset": settings.hf_dataset,
        "model": "CLIP multilingüe",
        "image_model": settings.image_model,
        "text_model": settings.text_model,
        "vector_dimension": 512,
        "distance": "cosine",
        "collection": settings.qdrant_collection,
        "points_count": int(
            collection_info.points_count or 0
        ),
        "indexed_vectors_count": int(
            collection_info.indexed_vectors_count
            or 0
        ),
        "collection_status": str(
            collection_info.status
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }