"""Aplicación principal de FastAPI."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from PIL import Image

from src.api.routes import (
    health,
    media,
    search,
)
from src.core.config import get_settings
from src.services.embedding_service import EmbeddingService
from src.services.qdrant_service import QdrantService
from src.services.search_service import SearchService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga una única instancia del modelo y de Qdrant."""
    settings = get_settings()

    print("=" * 70)
    print("INICIANDO VISUAL SEARCH API")
    print("=" * 70)

    embedding_service = EmbeddingService(
        settings=settings
    )

    qdrant_service = QdrantService(
        settings=settings
    )

    qdrant_service.check_connection()

    collection_info = (
        qdrant_service.client.get_collection(
            settings.qdrant_collection
        )
    )

    if int(collection_info.points_count or 0) == 0:
        qdrant_service.close()

        raise RuntimeError(
            "La colección de Qdrant está vacía."
        )

    search_service = SearchService(
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
    )

    # Calentamiento inicial de texto e imagen.
    embedding_service.encode_texts(
        ["warmup query"]
    )

    dummy_image = Image.new(
        mode="RGB",
        size=(224, 224),
        color="white",
    )

    embedding_service.encode_images(
        [dummy_image]
    )

    dummy_image.close()

    app.state.settings = settings
    app.state.embedding_service = (
        embedding_service
    )
    app.state.qdrant_service = (
        qdrant_service
    )
    app.state.search_service = search_service

    print(
        f"Colección: {settings.qdrant_collection}"
    )
    print(
        f"Puntos: "
        f"{int(collection_info.points_count or 0):,}"
    )
    print("API LISTA")
    print("=" * 70)

    try:
        yield

    finally:
        qdrant_service.close()
        print("Visual Search API detenida.")


app = FastAPI(
    title="Visual Search API",
    description=(
        "API multimodal para buscar imágenes de "
        "Caltech-256 mediante texto o imágenes."
    ),
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

app.include_router(health.router)
app.include_router(search.router)
app.include_router(media.router)