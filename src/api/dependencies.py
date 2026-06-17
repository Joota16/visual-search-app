"""Dependencias compartidas por los endpoints."""

from __future__ import annotations

from fastapi import Request

from src.core.config import Settings
from src.services.qdrant_service import QdrantService
from src.services.search_service import SearchService


def get_settings_from_app(
    request: Request,
) -> Settings:
    """Recupera la configuración cargada al iniciar la API."""
    return request.app.state.settings


def get_search_service(
    request: Request,
) -> SearchService:
    """Recupera el servicio multimodal."""
    return request.app.state.search_service


def get_qdrant_service(
    request: Request,
) -> QdrantService:
    """Recupera el servicio de Qdrant."""
    return request.app.state.qdrant_service