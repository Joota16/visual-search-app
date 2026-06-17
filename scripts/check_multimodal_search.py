"""Prueba las búsquedas texto-imagen e imagen-imagen."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from src.core.config import get_settings
from src.services.embedding_service import EmbeddingService
from src.services.qdrant_service import QdrantService
from src.services.search_service import SearchService


TOP_K = 8
TARGET_LABEL = "grapes"
TEXT_QUERY = "a photo of grapes"

REPORT_FILENAME = "multimodal_search_report.json"


def load_manifest(
    manifest_path: Path,
) -> list[dict[str, str]]:
    """Carga los registros del manifiesto."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No existe el manifiesto: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        records = list(csv.DictReader(csv_file))

    if not records:
        raise RuntimeError(
            "El manifiesto está vacío."
        )

    return records


def select_query_record(
    records: list[dict[str, str]],
    target_label: str,
) -> dict[str, str]:
    """Selecciona una imagen de prueba por categoría."""
    for record in records:
        if (
            record["label"] == target_label
            and record["split"] == "test"
        ):
            return record

    for record in records:
        if record["label"] == target_label:
            return record

    raise RuntimeError(
        f"No se encontró la categoría: {target_label}"
    )


def print_results(
    title: str,
    result: dict[str, Any],
) -> None:
    """Muestra los resultados de una búsqueda."""
    print("\n" + title)
    print("-" * 80)

    print(
        f"Embedding: "
        f"{result['embedding_ms']:.2f} ms"
    )

    print(
        f"Búsqueda Qdrant: "
        f"{result['search_ms']:.2f} ms"
    )

    print(
        f"Tiempo total: "
        f"{result['total_ms']:.2f} ms"
    )

    print()

    for item in result["results"]:
        print(
            f"{item['position']:>2}. "
            f"score={item['score']:.4f} | "
            f"label={item['label']:<25} | "
            f"split={item['split']:<10} | "
            f"id={item['point_id']}"
        )


def calculate_precision_at_k(
    results: list[dict[str, Any]],
    expected_label: str,
) -> float:
    """Calcula la proporción de resultados con la categoría esperada."""
    if not results:
        return 0.0

    relevant = sum(
        result["label"] == expected_label
        for result in results
    )

    return relevant / len(results)


def write_json_atomically(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """Guarda el reporte utilizando un temporal."""
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            data,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(output_path)


def main() -> None:
    """Ejecuta las dos búsquedas multimodales."""
    settings = get_settings()

    manifest_path = (
        settings.manifest_path
        / "caltech256_manifest.csv"
    )

    report_path = (
        settings.manifest_path
        / REPORT_FILENAME
    )

    records = load_manifest(manifest_path)

    query_record = select_query_record(
        records=records,
        target_label=TARGET_LABEL,
    )

    query_image_path = (
        settings.image_path
        / query_record["image_relpath"]
    )

    if not query_image_path.is_file():
        raise FileNotFoundError(
            f"No existe la imagen: {query_image_path}"
        )

    print("=" * 80)
    print("PRUEBA DE BÚSQUEDA MULTIMODAL")
    print("=" * 80)
    print(f"Colección: {settings.qdrant_collection}")
    print(f"Consulta textual: {TEXT_QUERY}")
    print(f"Imagen de consulta: {query_image_path}")
    print(f"Categoría real: {query_record['label']}")
    print(f"Point ID: {query_record['point_id']}")
    print(f"Top K: {TOP_K}")

    model_start = time.perf_counter()

    embedding_service = EmbeddingService(
        settings=settings
    )

    qdrant_service = QdrantService(
        settings=settings
    )

    try:
        qdrant_service.check_connection()

        search_service = SearchService(
            embedding_service=embedding_service,
            qdrant_service=qdrant_service,
        )

        with Image.open(
            query_image_path
        ) as source_image:
            query_image = source_image.convert(
                "RGB"
            )

        # Calentamiento para no incluir la inicialización
        # de CUDA en las mediciones principales.
        _ = embedding_service.encode_texts(
            ["warmup query"]
        )

        _ = embedding_service.encode_images(
            [query_image]
        )

        model_load_seconds = (
            time.perf_counter() - model_start
        )

        text_result = search_service.search_text(
            query=TEXT_QUERY,
            limit=TOP_K,
        )

        image_result = search_service.search_image(
            image=query_image,
            limit=TOP_K,
            exclude_ids=[
                query_record["point_id"]
            ],
        )

        query_image.close()

        if len(text_result["results"]) != TOP_K:
            raise RuntimeError(
                "La búsqueda textual no devolvió "
                f"{TOP_K} resultados."
            )

        if len(image_result["results"]) != TOP_K:
            raise RuntimeError(
                "La búsqueda por imagen no devolvió "
                f"{TOP_K} resultados."
            )

        returned_image_ids = {
            result["point_id"]
            for result in image_result["results"]
        }

        if query_record["point_id"] in returned_image_ids:
            raise RuntimeError(
                "La imagen de consulta fue devuelta "
                "entre sus propios resultados."
            )

        text_precision = calculate_precision_at_k(
            results=text_result["results"],
            expected_label=TARGET_LABEL,
        )

        image_precision = calculate_precision_at_k(
            results=image_result["results"],
            expected_label=TARGET_LABEL,
        )

        print_results(
            title="TEXTO → IMAGEN",
            result=text_result,
        )

        print_results(
            title="IMAGEN → IMAGEN",
            result=image_result,
        )

        report = {
            "collection": settings.qdrant_collection,
            "model": settings.openclip_model,
            "pretrained": settings.openclip_pretrained,
            "top_k": TOP_K,
            "model_load_and_warmup_seconds": (
                model_load_seconds
            ),
            "text_search": {
                **text_result,
                "expected_label": TARGET_LABEL,
                "precision_at_k": text_precision,
            },
            "image_search": {
                **image_result,
                "query_point_id": (
                    query_record["point_id"]
                ),
                "query_image": str(
                    query_image_path
                ),
                "expected_label": TARGET_LABEL,
                "precision_at_k": image_precision,
                "self_match_excluded": True,
            },
        }

        write_json_atomically(
            data=report,
            output_path=report_path,
        )

        print("\n" + "=" * 80)
        print("RESULTADO")
        print("=" * 80)

        print(
            f"Precision@{TOP_K} texto: "
            f"{text_precision:.4f}"
        )

        print(
            f"Precision@{TOP_K} imagen: "
            f"{image_precision:.4f}"
        )

        print(
            "Imagen original excluida: correcto"
        )

        print(
            f"Latencia Qdrant texto: "
            f"{text_result['search_ms']:.2f} ms"
        )

        print(
            f"Latencia Qdrant imagen: "
            f"{image_result['search_ms']:.2f} ms"
        )

        print(f"Reporte: {report_path}")

        print("=" * 80)
        print("BÚSQUEDA MULTIMODAL FUNCIONANDO")
        print("=" * 80)

    finally:
        qdrant_service.close()


if __name__ == "__main__":
    main()