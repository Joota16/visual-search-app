"""Servicio central para comunicarse con Qdrant."""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from src.core.config import Settings, get_settings


VECTOR_DIMENSION = 512


class QdrantService:
    """Administra la conexión y colección vectorial."""

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            timeout=self.settings.qdrant_timeout_seconds,
        )

    def check_connection(self) -> None:
        """Comprueba que Qdrant esté disponible."""
        self.client.get_collections()

    def prepare_collection(
        self,
        recreate: bool = False,
    ) -> None:
        """Crea o reutiliza la colección principal."""
        collection_name = (
            self.settings.qdrant_collection
        )

        exists = self.client.collection_exists(
            collection_name
        )

        if recreate and exists:
            print(
                f"Eliminando colección existente: "
                f"{collection_name}"
            )

            self.client.delete_collection(
                collection_name=collection_name
            )

            exists = False

        if not exists:
            print(
                f"Creando colección: {collection_name}"
            )

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=models.Distance.COSINE,
                    on_disk=False,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=1000,
                ),
            )

        else:
            print(
                f"Reutilizando colección: "
                f"{collection_name}"
            )

        self.create_payload_indexes()

    def create_payload_indexes(self) -> None:
        """Crea índices para los campos usados en filtros."""
        collection_name = (
            self.settings.qdrant_collection
        )

        collection_info = self.client.get_collection(
            collection_name
        )

        existing_indexes = set(
            collection_info.payload_schema.keys()
        )

        index_definitions = {
            "dataset_id": models.PayloadSchemaType.KEYWORD,
            "dataset_name": models.PayloadSchemaType.KEYWORD,
            "domain": models.PayloadSchemaType.KEYWORD,
            "source_dataset": models.PayloadSchemaType.KEYWORD,
            "split": models.PayloadSchemaType.KEYWORD,
            "label": models.PayloadSchemaType.KEYWORD,
            "label_id": models.PayloadSchemaType.INTEGER,
            "row_index": models.PayloadSchemaType.INTEGER,
        }
        for field_name, schema_type in (
            index_definitions.items()
        ):
            if field_name in existing_indexes:
                continue

            print(
                f"Creando índice de payload: "
                f"{field_name}"
            )

            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
                wait=True,
            )

    def close(self) -> None:
        """Cierra el cliente."""
        self.client.close()