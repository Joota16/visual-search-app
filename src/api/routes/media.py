"""Endpoints para entregar imágenes y miniaturas."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import FileResponse

from src.api.dependencies import (
    get_qdrant_service,
    get_settings_from_app,
)
from src.core.config import Settings
from src.services.qdrant_service import QdrantService


router = APIRouter(
    prefix="/api/v1",
    tags=["Imágenes"],
)


def safe_path(
    root: Path,
    relative_path: str,
) -> Path:
    """Evita resolver archivos fuera de la carpeta permitida."""
    resolved_root = root.resolve()

    candidate = (
        resolved_root
        / relative_path
    ).resolve()

    if (
        candidate != resolved_root
        and resolved_root not in candidate.parents
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ruta de archivo no válida.",
        )

    if not candidate.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Archivo no encontrado.",
        )

    return candidate


def get_payload(
    point_id: str,
    qdrant_service: QdrantService,
) -> dict:
    """Recupera los metadatos de un punto."""
    points = qdrant_service.client.retrieve(
        collection_name=(
            qdrant_service
            .settings
            .qdrant_collection
        ),
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No existe la imagen solicitada.",
        )

    return points[0].payload or {}


def return_media(
    point_id: str,
    media_type: Literal["image", "thumbnail"],
    settings: Settings,
    qdrant_service: QdrantService,
) -> FileResponse:
    """Localiza y entrega el archivo solicitado."""
    payload = get_payload(
        point_id=point_id,
        qdrant_service=qdrant_service,
    )

    if media_type == "image":
        root = settings.image_path
        relative_path = str(
            payload.get("image_relpath", "")
        )
    else:
        root = settings.thumbnail_path
        relative_path = str(
            payload.get("thumbnail_relpath", "")
        )

    if not relative_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El punto no contiene una ruta válida.",
        )

    file_path = safe_path(
        root=root,
        relative_path=relative_path,
    )

    return FileResponse(
        path=file_path,
        headers={
            "Cache-Control": (
                "public, max-age=86400"
            )
        },
    )


@router.get(
    "/images/{point_id}",
    name="get_image",
)
def get_image(
    point_id: str,
    settings: Settings = Depends(
        get_settings_from_app
    ),
    qdrant_service: QdrantService = Depends(
        get_qdrant_service
    ),
) -> FileResponse:
    """Entrega una imagen original."""
    return return_media(
        point_id=point_id,
        media_type="image",
        settings=settings,
        qdrant_service=qdrant_service,
    )


@router.get(
    "/thumbnails/{point_id}",
    name="get_thumbnail",
)
def get_thumbnail(
    point_id: str,
    settings: Settings = Depends(
        get_settings_from_app
    ),
    qdrant_service: QdrantService = Depends(
        get_qdrant_service
    ),
) -> FileResponse:
    """Entrega una miniatura WebP."""
    return return_media(
        point_id=point_id,
        media_type="thumbnail",
        settings=settings,
        qdrant_service=qdrant_service,
    )