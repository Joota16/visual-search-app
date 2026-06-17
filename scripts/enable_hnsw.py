"""Activa la indexación HNSW y espera a que Qdrant la complete."""

from __future__ import annotations

import time

from qdrant_client import models

from src.core.config import get_settings
from src.services.qdrant_service import QdrantService


INDEXING_THRESHOLD_KB = 1000
POLL_INTERVAL_SECONDS = 2
TIMEOUT_SECONDS = 180


def status_is_green(status: object) -> bool:
    """Comprueba el estado sin depender de la representación del enum."""
    return str(status).lower().endswith("green")


def main() -> None:
    settings = get_settings()
    service = QdrantService(settings=settings)

    try:
        service.check_connection()

        collection_name = settings.qdrant_collection

        initial_info = service.client.get_collection(
            collection_name
        )

        print("=" * 72)
        print("ACTIVACIÓN DEL ÍNDICE HNSW")
        print("=" * 72)
        print(f"Colección: {collection_name}")
        print(
            f"Puntos: {int(initial_info.points_count or 0):,}"
        )
        print(
            "Vectores indexados inicialmente: "
            f"{int(initial_info.indexed_vectors_count or 0):,}"
        )
        print(
            f"Nuevo indexing_threshold: "
            f"{INDEXING_THRESHOLD_KB} kB"
        )

        service.client.update_collection(
            collection_name=collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=INDEXING_THRESHOLD_KB,
            ),
        )

        print("\nConfiguración actualizada.")
        print("Esperando la optimización de Qdrant...")

        start_time = time.perf_counter()
        last_indexed_count = -1

        while True:
            info = service.client.get_collection(
                collection_name
            )

            points_count = int(
                info.points_count or 0
            )

            indexed_count = int(
                info.indexed_vectors_count or 0
            )

            elapsed_seconds = (
                time.perf_counter() - start_time
            )

            if indexed_count != last_indexed_count:
                percentage = (
                    indexed_count / points_count * 100
                    if points_count
                    else 0
                )

                print(
                    f"Indexados: {indexed_count:,} / "
                    f"{points_count:,} "
                    f"({percentage:.2f}%) | "
                    f"estado={info.status}"
                )

                last_indexed_count = indexed_count

            if (
                indexed_count > 0
                and status_is_green(info.status)
            ):
                break

            if elapsed_seconds >= TIMEOUT_SECONDS:
                raise TimeoutError(
                    "Qdrant no construyó el índice HNSW "
                    f"en {TIMEOUT_SECONDS} segundos."
                )

            time.sleep(POLL_INTERVAL_SECONDS)

        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        final_info = service.client.get_collection(
            collection_name
        )

        print("\n" + "=" * 72)
        print("RESULTADO")
        print("=" * 72)
        print(
            f"Puntos: "
            f"{int(final_info.points_count or 0):,}"
        )
        print(
            "Vectores indexados por HNSW: "
            f"{int(final_info.indexed_vectors_count or 0):,}"
        )
        print(f"Estado: {final_info.status}")
        print(
            f"Tiempo de optimización: "
            f"{elapsed_seconds:.2f} segundos"
        )
        print("=" * 72)
        print("HNSW ACTIVADO CORRECTAMENTE")
        print("=" * 72)

    finally:
        service.close()


if __name__ == "__main__":
    main()