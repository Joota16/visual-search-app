"""Endpoints de búsqueda mediante texto e imagen."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from src.api.dependencies import (
    get_search_service,
    get_settings_from_app,
)
from src.core.config import Settings
from src.schemas.search import (
    SearchResponse,
    TextSearchRequest,
)
from src.services.search_service import SearchService


router = APIRouter(
    prefix="/api/v1/search",
    tags=["Búsqueda"],
)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


def build_response(
    raw_result: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    """Añade las URLs públicas y metadatos de cada imagen."""
    serialized_results: list[dict[str, Any]] = []

    for result in raw_result["results"]:
        point_id = result["point_id"]

        serialized_results.append(
        {
            "position": result["position"],
            "point_id": point_id,
            "score": result["score"],

            "label": result["label"],
            "raw_label": result.get("raw_label"),

            "label_id": result["label_id"],
            "split": result["split"],
            "row_index": result["row_index"],

            "dataset_id": result.get("dataset_id"),
            "dataset_name": result.get("dataset_name"),
            "domain": result.get("domain"),
            "source_dataset": result.get("source_dataset"),

            "width": result.get("width"),
            "height": result.get("height"),

            "image_url": str(
                request.url_for(
                    "get_image",
                    point_id=point_id,
                )
            ),
            "thumbnail_url": str(
                request.url_for(
                    "get_thumbnail",
                    point_id=point_id,
                )
            ),
        }
    )

    return {
        "query_type": raw_result["query_type"],
        "query": raw_result.get("query"),
        "embedding_ms": raw_result["embedding_ms"],
        "search_ms": raw_result["search_ms"],
        "total_ms": raw_result["total_ms"],
        "results": serialized_results,
    }


@router.post(
    "/text",
    response_model=SearchResponse,
)
async def search_by_text(
    payload: TextSearchRequest,
    request: Request,
    search_service: SearchService = Depends(
        get_search_service
    ),
) -> dict[str, Any]:
    """Busca imágenes mediante una descripción textual."""
    try:
        raw_result = await run_in_threadpool(
            search_service.search_text,
            payload.query,
            payload.limit,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return build_response(
        raw_result=raw_result,
        request=request,
    )


@router.post(
    "/image",
    response_model=SearchResponse,
)
async def search_by_image(
    request: Request,
    file: UploadFile = File(...),
    limit: int = Form(
        default=8,
        ge=1,
        le=50,
    ),
    settings: Settings = Depends(
        get_settings_from_app
    ),
    search_service: SearchService = Depends(
        get_search_service
    ),
) -> dict[str, Any]:
    """Busca imágenes similares a un archivo cargado."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Formato no permitido. "
                "Utiliza JPEG, PNG o WebP."
            ),
        )

    maximum_bytes = (
        settings.max_upload_mb * 1024 * 1024
    )

    contents = await file.read(
        maximum_bytes + 1
    )

    await file.close()

    if len(contents) > maximum_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "La imagen excede el límite de "
                f"{settings.max_upload_mb} MB."
            ),
        )

    try:
        with Image.open(
            BytesIO(contents)
        ) as verification_image:
            verification_image.verify()

        with Image.open(
            BytesIO(contents)
        ) as source_image:
            query_image = source_image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El archivo no contiene una imagen válida.",
        ) from error

    try:
        raw_result = await run_in_threadpool(
            search_service.search_image,
            query_image,
            limit,
        )

    finally:
        query_image.close()

    return build_response(
        raw_result=raw_result,
        request=request,
    )