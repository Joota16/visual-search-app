"""Servicio de búsqueda multimodal sobre Qdrant."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image
from qdrant_client import models

from src.services.embedding_service import EmbeddingService
from src.services.qdrant_service import QdrantService

import re

try:
    from nltk.corpus import wordnet as wn
except Exception:
    wn = None


WNID_PATTERN = re.compile(
    r"^[nvar]\d{8}$",
    re.IGNORECASE,
)


def humanize_label(
    label: str,
    dataset_id: str,
) -> str:
    """Convierte etiquetas técnicas en nombres legibles."""
    clean_label = label.strip()

    if dataset_id != "tiny_imagenet":
        return clean_label

    if not WNID_PATTERN.match(clean_label):
        return clean_label

    if wn is None:
        return clean_label

    try:
        pos = clean_label[0].lower()
        offset = int(clean_label[1:])

        synset = wn.synset_from_pos_and_offset(
            pos,
            offset,
        )

        names = [
            name.replace("_", " ")
            for name in synset.lemma_names()
        ]

        if names:
            return names[0].title()

    except LookupError:
        return clean_label

    except Exception:
        return clean_label

    return clean_label

DEFAULT_HNSW_EF = 128
VECTOR_DIMENSION = 512


class SearchService:
    """Realiza búsquedas por texto, imagen o vector."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
    ) -> None:
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service

    @staticmethod
    def _validate_limit(limit: int) -> None:
        """Valida la cantidad solicitada de resultados."""
        if limit <= 0:
            raise ValueError(
                "El límite debe ser mayor que cero."
            )

        if limit > 100:
            raise ValueError(
                "El límite no puede ser mayor que 100."
            )

    @staticmethod
    def _build_filter(
        exclude_ids: Sequence[str] | None = None,
    ) -> models.Filter | None:
        """Construye el filtro usado en la búsqueda."""
        if not exclude_ids:
            return None

        return models.Filter(
            must_not=[
                models.HasIdCondition(
                    has_id=list(exclude_ids),
                )
            ]
        )

    def search_vector(
        self,
        query_vector: np.ndarray,
        limit: int = 8,
        exclude_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Busca los vecinos más cercanos a un vector."""
        self._validate_limit(limit)

        vector = np.asarray(
            query_vector,
            dtype=np.float32,
        ).reshape(-1)

        if vector.shape != (VECTOR_DIMENSION,):
            raise ValueError(
                "El vector debe tener dimensión "
                f"{VECTOR_DIMENSION}, pero tiene "
                f"{vector.shape}."
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "El vector contiene NaN o infinito."
            )

        query_filter = self._build_filter(
            exclude_ids=exclude_ids,
        )

        search_start = time.perf_counter()

        response = (
            self.qdrant_service.client.query_points(
                collection_name=(
                    self.qdrant_service
                    .settings
                    .qdrant_collection
                ),
                query=vector.tolist(),
                query_filter=query_filter,
                search_params=models.SearchParams(
                    hnsw_ef=DEFAULT_HNSW_EF,
                    exact=False,
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        )

        search_ms = (
            time.perf_counter() - search_start
        ) * 1000

        results: list[dict[str, Any]] = []

        for position, point in enumerate(
            response.points,
            start=1,
        ):
            payload = point.payload or {}

            dataset_id = str(
                payload.get("dataset_id", "")
            )

            raw_label = str(
                payload.get("label", "")
            )

            display_label = humanize_label(
                label=raw_label,
                dataset_id=dataset_id,
            )

            results.append(
                {
                    "position": position,
                    "point_id": str(point.id),
                    "score": float(point.score),

                    "label": display_label,
                    "raw_label": raw_label,

                    "dataset_id": dataset_id,
                    "dataset_name": str(
                        payload.get("dataset_name", "")
                    ),
                    "domain": str(
                        payload.get("domain", "")
                    ),
                    "source_dataset": str(
                        payload.get("source_dataset", "")
                    ),

                    "label_id": int(
                        payload.get("label_id", -1)
                    ),
                    "split": str(
                        payload.get("split", "")
                    ),
                    "row_index": int(
                        payload.get("row_index", -1)
                    ),

                    "width": int(
                        payload.get("width", 0) or 0
                    ),
                    "height": int(
                        payload.get("height", 0) or 0
                    ),

                    "image_relpath": str(
                        payload.get(
                            "image_relpath",
                            "",
                        )
                    ),
                    "thumbnail_relpath": str(
                        payload.get(
                            "thumbnail_relpath",
                            "",
                        )
                    ),
                }
            )

        return {
            "search_ms": search_ms,
            "results": results,
        }

    def search_text(
        self,
        query: str,
        limit: int = 8,
    ) -> dict[str, Any]:
        """Convierte un texto en embedding y busca imágenes."""
        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "La consulta textual está vacía."
            )

        total_start = time.perf_counter()
        embedding_start = time.perf_counter()

        embedding = (
            self.embedding_service.encode_texts(
                [clean_query]
            )[0]
        )

        embedding_ms = (
            time.perf_counter() - embedding_start
        ) * 1000

        search_result = self.search_vector(
            query_vector=embedding,
            limit=limit,
        )

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        return {
            "query_type": "text",
            "query": clean_query,
            "embedding_ms": embedding_ms,
            "search_ms": search_result["search_ms"],
            "total_ms": total_ms,
            "results": search_result["results"],
        }

    def search_image(
        self,
        image: Image.Image,
        limit: int = 8,
        exclude_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Convierte una imagen en embedding y busca similares."""
        total_start = time.perf_counter()
        embedding_start = time.perf_counter()

        embedding = (
            self.embedding_service.encode_images(
                [image]
            )[0]
        )

        embedding_ms = (
            time.perf_counter() - embedding_start
        ) * 1000

        search_result = self.search_vector(
            query_vector=embedding,
            limit=limit,
            exclude_ids=exclude_ids,
        )

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        return {
            "query_type": "image",
            "embedding_ms": embedding_ms,
            "search_ms": search_result["search_ms"],
            "total_ms": total_ms,
            "results": search_result["results"],
        }