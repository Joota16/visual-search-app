"""Esquemas de entrada y salida para búsquedas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TextSearchRequest(BaseModel):
    """Solicitud de búsqueda mediante texto."""

    query: str = Field(
        min_length=1,
        max_length=300,
        examples=["a photo of grapes"],
    )

    limit: int = Field(
        default=8,
        ge=1,
        le=50,
    )


class SearchResult(BaseModel):
    """Una imagen recuperada por el buscador."""

    position: int
    point_id: str
    score: float
    label: str
    label_id: int
    split: str
    row_index: int
    image_url: str
    thumbnail_url: str


class SearchResponse(BaseModel):
    """Respuesta común de búsqueda multimodal."""

    query_type: Literal["text", "image"]
    query: str | None = None

    embedding_ms: float
    search_ms: float
    total_ms: float

    results: list[SearchResult]