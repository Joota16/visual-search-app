"""Carga embeddings multilingües y metadatos de Caltech-256 en Qdrant."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import models
from tqdm import tqdm

from src.core.config import get_settings
from src.services.qdrant_service import (
    QdrantService,
    VECTOR_DIMENSION,
)


EMBEDDING_FILENAME = "caltech256_multilingual_clip_vit_b32.npy"
REPORT_FILENAME = "qdrant_index_report_multilingual.json"


def parse_arguments() -> argparse.Namespace:
    """Lee argumentos de la terminal."""
    parser = argparse.ArgumentParser(
        description=(
            "Inserta los embeddings multilingües de Caltech-256 "
            "en Qdrant."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cantidad máxima de registros a insertar.",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Elimina y vuelve a crear la colección.",
    )

    return parser.parse_args()


def load_manifest(
    manifest_path: Path,
) -> list[dict[str, str]]:
    """Carga el manifiesto CSV."""
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


def build_payload(
    record: dict[str, str],
) -> dict[str, Any]:
    """Convierte una fila del manifiesto en payload de Qdrant."""
    return {
        "record_key": record["record_key"],
        "dataset_id": record["dataset_id"],
        "split": record["split"],
        "row_index": int(record["row_index"]),
        "label_id": int(record["label_id"]),
        "label": record["label"],
        "image_relpath": record["image_relpath"],
        "thumbnail_relpath": record["thumbnail_relpath"],
        "width": int(record["width"]),
        "height": int(record["height"]),
        "image_bytes": int(record["image_bytes"]),
        "thumbnail_bytes": int(record["thumbnail_bytes"]),

        # Modelo nuevo
        "embedding_type": "sentence_transformers_multilingual_clip",
        "embedding_image_model": "clip-ViT-B-32",
        "embedding_text_model": (
            "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        ),
        "embedding_version": "multilingual_v1",
    }


def write_json_atomically(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """Guarda un JSON mediante un archivo temporal."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: int,
) -> None:
    """Valida forma y tipo del archivo de embeddings."""
    expected_shape = (
        expected_rows,
        VECTOR_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise RuntimeError(
            "La forma del archivo de embeddings es incorrecta: "
            f"{embeddings.shape} != {expected_shape}"
        )

    if embeddings.dtype != np.float32:
        raise RuntimeError(
            "Los embeddings deben ser float32, "
            f"pero son {embeddings.dtype}."
        )


def main() -> None:
    """Ejecuta la indexación completa."""
    arguments = parse_arguments()
    settings = get_settings()

    manifest_path = (
        settings.manifest_path
        / "caltech256_manifest.csv"
    )

    embedding_path = (
        settings.embedding_path
        / EMBEDDING_FILENAME
    )

    report_path = (
        settings.manifest_path
        / REPORT_FILENAME
    )

    if not embedding_path.exists():
        raise FileNotFoundError(
            "No existe el archivo de embeddings multilingües: "
            f"{embedding_path}\n"
            "Primero ejecuta: python -m scripts.generate_embeddings"
        )

    records = load_manifest(manifest_path)

    embeddings = np.load(
        embedding_path,
        mmap_mode="r",
    )

    validate_embeddings(
        embeddings=embeddings,
        expected_rows=len(records),
    )

    if (
        arguments.limit is not None
        and arguments.limit <= 0
    ):
        raise ValueError(
            "--limit debe ser mayor que cero."
        )

    target_count = (
        min(arguments.limit, len(records))
        if arguments.limit is not None
        else len(records)
    )

    print("=" * 80)
    print("INDEXACIÓN EN QDRANT — CLIP MULTILINGÜE")
    print("=" * 80)
    print(f"Colección: {settings.qdrant_collection}")
    print(f"Archivo embeddings: {embedding_path}")
    print(f"Registros disponibles: {len(records):,}")
    print(f"Registros a insertar: {target_count:,}")
    print(f"Dimensión vectorial: {VECTOR_DIMENSION}")
    print(f"Batch de inserción: {settings.upsert_batch_size}")
    print(f"Recrear colección: {arguments.recreate}")
    print("=" * 80)

    service = QdrantService(
        settings=settings
    )

    try:
        service.check_connection()

        print("Conexión con Qdrant: correcta")

        service.prepare_collection(
            recreate=arguments.recreate
        )

        total_batches = math.ceil(
            target_count
            / settings.upsert_batch_size
        )

        index_start = time.perf_counter()

        for start in tqdm(
            range(
                0,
                target_count,
                settings.upsert_batch_size,
            ),
            total=total_batches,
            desc="Insertando vectores",
            unit="lote",
        ):
            end = min(
                start + settings.upsert_batch_size,
                target_count,
            )

            batch_records = records[start:end]

            batch_ids = [
                record["point_id"]
                for record in batch_records
            ]

            batch_payloads = [
                build_payload(record)
                for record in batch_records
            ]

            batch_vectors = np.asarray(
                embeddings[start:end],
                dtype=np.float32,
            ).tolist()

            service.client.upsert(
                collection_name=settings.qdrant_collection,
                points=models.Batch(
                    ids=batch_ids,
                    payloads=batch_payloads,
                    vectors=batch_vectors,
                ),
                wait=True,
            )

        elapsed_seconds = (
            time.perf_counter()
            - index_start
        )

        collection_info = (
            service.client.get_collection(
                settings.qdrant_collection
            )
        )

        points_count = int(
            collection_info.points_count or 0
        )

        indexed_vectors_count = int(
            collection_info.indexed_vectors_count
            or 0
        )

        if arguments.limit is None:
            if points_count != len(records):
                raise RuntimeError(
                    "La colección no contiene todos los puntos: "
                    f"{points_count:,} != {len(records):,}"
                )

        elif points_count < target_count:
            raise RuntimeError(
                "La colección contiene menos puntos "
                "de los insertados."
            )

        points_per_second = (
            target_count / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        )

        report = {
            "collection": settings.qdrant_collection,
            "qdrant_url": settings.qdrant_url,
            "embedding_file": str(embedding_path),
            "embedding_type": "sentence_transformers_multilingual_clip",
            "embedding_image_model": "clip-ViT-B-32",
            "embedding_text_model": (
                "sentence-transformers/clip-ViT-B-32-multilingual-v1"
            ),
            "target_count": target_count,
            "points_count": points_count,
            "indexed_vectors_count": indexed_vectors_count,
            "vector_dimension": VECTOR_DIMENSION,
            "distance": "cosine",
            "upsert_batch_size": settings.upsert_batch_size,
            "elapsed_seconds": elapsed_seconds,
            "points_per_second": points_per_second,
            "collection_status": str(collection_info.status),
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        write_json_atomically(
            report,
            report_path,
        )

        print("\n" + "=" * 80)
        print("RESULTADO")
        print("=" * 80)
        print(f"Puntos enviados: {target_count:,}")
        print(f"Puntos en colección: {points_count:,}")
        print(
            f"Vectores indexados por HNSW: "
            f"{indexed_vectors_count:,}"
        )
        print(f"Tiempo: {elapsed_seconds:.2f} s")
        print(
            f"Rendimiento: "
            f"{points_per_second:.2f} puntos/s"
        )
        print(f"Estado: {collection_info.status}")
        print(f"Reporte: {report_path}")
        print("=" * 80)
        print("INDEXACIÓN MULTILINGÜE EN QDRANT COMPLETADA")
        print("=" * 80)

    finally:
        service.close()


if __name__ == "__main__":
    main()