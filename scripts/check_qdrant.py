"""Prueba la conexión, escritura y búsqueda en Qdrant."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


TEST_COLLECTION = "visual_search_connection_test"


def create_client() -> QdrantClient:
    """Crea el cliente usando la URL definida en el archivo .env."""
    load_dotenv()

    qdrant_url = os.getenv(
        "QDRANT_URL",
        "http://localhost:6333",
    )

    return QdrantClient(
        url=qdrant_url,
        timeout=10.0,
    )


def print_results(results: list) -> None:
    """Muestra los resultados encontrados."""
    print("\nResultados de la búsqueda:")

    for position, point in enumerate(results, start=1):
        print(
            f"{position}. "
            f"id={point.id} | "
            f"score={point.score:.4f} | "
            f"payload={point.payload}"
        )


def main() -> None:
    """Ejecuta una prueba completa sobre una colección temporal."""
    client = create_client()
    collection_created = False

    print("=" * 60)
    print("VERIFICACIÓN DE QDRANT")
    print("=" * 60)

    try:
        collections = client.get_collections()

        print("Conexión con Qdrant: correcta")
        print(
            "Colecciones existentes:",
            len(collections.collections),
        )

        # Limpiar una prueba anterior que pudiera haber quedado incompleta.
        if client.collection_exists(TEST_COLLECTION):
            client.delete_collection(TEST_COLLECTION)

        # Colección temporal de dimensión 4.
        client.create_collection(
            collection_name=TEST_COLLECTION,
            vectors_config=models.VectorParams(
                size=4,
                distance=models.Distance.COSINE,
            ),
        )

        collection_created = True

        print(f"Colección temporal creada: {TEST_COLLECTION}")

        points = [
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0, 0.0, 0.0],
                payload={
                    "name": "vector_rojo",
                    "category": "test",
                },
            ),
            models.PointStruct(
                id=2,
                vector=[0.0, 1.0, 0.0, 0.0],
                payload={
                    "name": "vector_verde",
                    "category": "test",
                },
            ),
            models.PointStruct(
                id=3,
                vector=[0.9, 0.1, 0.0, 0.0],
                payload={
                    "name": "vector_rojo_similar",
                    "category": "test",
                },
            ),
        ]

        client.upsert(
            collection_name=TEST_COLLECTION,
            points=points,
            wait=True,
        )

        collection_info = client.get_collection(TEST_COLLECTION)

        print(
            "Puntos almacenados:",
            collection_info.points_count,
        )

        if collection_info.points_count != len(points):
            raise RuntimeError(
                "La cantidad de puntos almacenados no es la esperada."
            )

        response = client.query_points(
            collection_name=TEST_COLLECTION,
            query=[1.0, 0.0, 0.0, 0.0],
            limit=3,
            with_payload=True,
        )

        results = response.points

        if not results:
            raise RuntimeError(
                "Qdrant no devolvió resultados."
            )

        print_results(results)

        if results[0].id != 1:
            raise RuntimeError(
                "El resultado más similar no fue el esperado."
            )

        print("\nLa búsqueda por similitud funcionó correctamente.")

    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise

    finally:
        if collection_created and client.collection_exists(
            TEST_COLLECTION
        ):
            client.delete_collection(TEST_COLLECTION)

            print(
                "\nColección temporal eliminada correctamente."
            )

        client.close()

    print("=" * 60)
    print("QDRANT LISTO")
    print("=" * 60)


if __name__ == "__main__":
    main()