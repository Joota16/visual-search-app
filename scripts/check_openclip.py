"""Comprueba OpenCLIP mediante una imagen real de Caltech-256."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.core.config import get_settings
from src.services.embedding_service import EmbeddingService


def load_manifest(
    manifest_path: Path,
) -> list[dict[str, str]]:
    """Carga el manifiesto CSV."""
    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def build_prompts(
    records: list[dict[str, str]],
    correct_label: str,
) -> list[str]:
    """Construye una lista de textos para probar la búsqueda."""
    alternative_labels: list[str] = []

    for record in records:
        label = record["label"]

        if (
            label != correct_label
            and label not in alternative_labels
        ):
            alternative_labels.append(label)

        if len(alternative_labels) == 3:
            break

    labels = [
        correct_label,
        *alternative_labels,
    ]

    return [
        f"a photo of a {label}"
        for label in labels
    ]


def verify_embeddings(
    image_embedding: np.ndarray,
    text_embeddings: np.ndarray,
) -> None:
    """Valida dimensiones, valores y normalización."""
    if image_embedding.ndim != 2:
        raise RuntimeError(
            "El embedding de imagen no tiene dos dimensiones."
        )

    if text_embeddings.ndim != 2:
        raise RuntimeError(
            "Los embeddings textuales no tienen dos dimensiones."
        )

    if (
        image_embedding.shape[1]
        != text_embeddings.shape[1]
    ):
        raise RuntimeError(
            "Texto e imagen tienen dimensiones diferentes."
        )

    if not np.isfinite(image_embedding).all():
        raise RuntimeError(
            "El embedding de imagen contiene valores inválidos."
        )

    if not np.isfinite(text_embeddings).all():
        raise RuntimeError(
            "Los embeddings textuales contienen valores inválidos."
        )

    image_norm = np.linalg.norm(
        image_embedding,
        axis=1,
    )

    text_norms = np.linalg.norm(
        text_embeddings,
        axis=1,
    )

    if not np.allclose(
        image_norm,
        1.0,
        atol=1e-3,
    ):
        raise RuntimeError(
            "El embedding de imagen no está normalizado."
        )

    if not np.allclose(
        text_norms,
        1.0,
        atol=1e-3,
    ):
        raise RuntimeError(
            "Los embeddings textuales no están normalizados."
        )


def synchronize_cuda() -> None:
    """Espera a que terminen las operaciones de CUDA."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    """Ejecuta una prueba multimodal completa."""
    settings = get_settings()

    manifest_path = (
        settings.manifest_path
        / "caltech256_manifest.csv"
    )

    records = load_manifest(manifest_path)

    if not records:
        raise RuntimeError(
            "El manifiesto no contiene registros."
        )

    sample_record = records[0]

    image_path = (
        settings.image_path
        / sample_record["image_relpath"]
    )

    correct_label = sample_record["label"]

    prompts = build_prompts(
        records=records,
        correct_label=correct_label,
    )

    print("=" * 70)
    print("VERIFICACIÓN DE OPENCLIP")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Imagen: {image_path}")
    print(f"Categoría real: {correct_label}")

    model_start = time.perf_counter()

    service = EmbeddingService(
        settings=settings
    )

    synchronize_cuda()

    model_load_seconds = (
        time.perf_counter()
        - model_start
    )

    print(
        f"Modelo cargado en: "
        f"{model_load_seconds:.2f} segundos"
    )

    with Image.open(image_path) as source_image:
        query_image = source_image.convert("RGB")

    # Calentamiento inicial.
    _ = service.encode_images([query_image])
    _ = service.encode_texts(prompts)

    synchronize_cuda()

    torch.cuda.reset_peak_memory_stats()

    inference_start = time.perf_counter()

    image_embedding = service.encode_images(
        [query_image]
    )

    text_embeddings = service.encode_texts(
        prompts
    )

    synchronize_cuda()

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    verify_embeddings(
        image_embedding=image_embedding,
        text_embeddings=text_embeddings,
    )

    similarities = (
        text_embeddings
        @ image_embedding.T
    ).reshape(-1)

    ranking = np.argsort(
        similarities
    )[::-1]

    image_norm = float(
        np.linalg.norm(image_embedding[0])
    )

    peak_memory_gb = (
        torch.cuda.max_memory_allocated()
        / (1024**3)
    )

    print("\nEmbeddings:")
    print(
        f"- Imagen: {image_embedding.shape}"
    )
    print(
        f"- Texto: {text_embeddings.shape}"
    )
    print(
        f"- Dimensión común: "
        f"{image_embedding.shape[1]}"
    )
    print(
        f"- Norma del embedding visual: "
        f"{image_norm:.6f}"
    )

    print("\nRanking texto → imagen:")

    for position, prompt_index in enumerate(
        ranking,
        start=1,
    ):
        print(
            f"{position}. "
            f"{prompts[prompt_index]} | "
            f"similitud={similarities[prompt_index]:.4f}"
        )

    print("\nRendimiento:")
    print(
        f"- Inferencia medida: "
        f"{inference_seconds * 1000:.2f} ms"
    )
    print(
        f"- Memoria CUDA máxima asignada: "
        f"{peak_memory_gb:.2f} GB"
    )

    query_image.close()

    print("=" * 70)
    print("OPENCLIP LISTO")
    print("=" * 70)


if __name__ == "__main__":
    main()