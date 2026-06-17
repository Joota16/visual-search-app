"""Compara tamaños de lote para OpenCLIP en la GPU."""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.core.config import get_settings
from src.services.embedding_service import EmbeddingService


BATCH_SIZES = [16, 32, 64, 128, 256]
REPETITIONS = 5
WARMUP_REPETITIONS = 2


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
        return list(csv.DictReader(csv_file))


def prepare_tensors(
    records: list[dict[str, str]],
    image_root: Path,
    service: EmbeddingService,
    sample_size: int,
) -> torch.Tensor:
    """Carga y preprocesa una muestra de imágenes."""

    tensors: list[torch.Tensor] = []

    selected_records = records[:sample_size]

    for record in tqdm(
        selected_records,
        desc="Preprocesando muestra",
        unit="imagen",
    ):
        image_path = (
            image_root
            / record["image_relpath"]
        )

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            tensor = service.preprocess(rgb_image)
            tensors.append(tensor)

    batch = torch.stack(tensors)

    if torch.cuda.is_available():
        batch = batch.pin_memory()

    return batch


def synchronize_cuda() -> None:
    """Espera a que finalicen las operaciones CUDA."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: int,
) -> None:
    """Valida forma, valores y normalización."""

    if embeddings.shape[0] != expected_rows:
        raise RuntimeError(
            "Cantidad inesperada de embeddings: "
            f"{embeddings.shape[0]} != {expected_rows}"
        )

    if embeddings.shape[1] != 512:
        raise RuntimeError(
            "Dimensión inesperada del embedding: "
            f"{embeddings.shape[1]}"
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Los embeddings contienen NaN o infinito."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-3,
    ):
        raise RuntimeError(
            "Los embeddings no están normalizados."
        )


def benchmark_batch(
    service: EmbeddingService,
    all_tensors: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    """Ejecuta el benchmark para un tamaño de lote."""

    batch = all_tensors[:batch_size]

    for _ in range(WARMUP_REPETITIONS):
        _ = service.encode_image_tensors(batch)

    synchronize_cuda()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    elapsed_times_ms: list[float] = []
    last_embeddings: np.ndarray | None = None

    for _ in range(REPETITIONS):
        synchronize_cuda()

        start = time.perf_counter()

        last_embeddings = (
            service.encode_image_tensors(batch)
        )

        synchronize_cuda()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        elapsed_times_ms.append(elapsed_ms)

    if last_embeddings is None:
        raise RuntimeError(
            "No se generaron embeddings."
        )

    validate_embeddings(
        embeddings=last_embeddings,
        expected_rows=batch_size,
    )

    median_ms = statistics.median(
        elapsed_times_ms
    )

    mean_ms = statistics.mean(
        elapsed_times_ms
    )

    throughput = (
        batch_size / (median_ms / 1000)
    )

    peak_allocated_gb = (
        torch.cuda.max_memory_allocated()
        / (1024**3)
    )

    peak_reserved_gb = (
        torch.cuda.max_memory_reserved()
        / (1024**3)
    )

    return {
        "batch_size": batch_size,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min(elapsed_times_ms),
        "max_ms": max(elapsed_times_ms),
        "images_per_second": throughput,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "status": "ok",
    }


def save_results(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Guarda los resultados como CSV."""

    if not results:
        raise RuntimeError(
            "No existen resultados para guardar."
        )

    temporary_path = output_path.with_suffix(
        ".csv.part"
    )

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "batch_size",
                "mean_ms",
                "median_ms",
                "min_ms",
                "max_ms",
                "images_per_second",
                "peak_allocated_gb",
                "peak_reserved_gb",
                "status",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    temporary_path.replace(output_path)


def print_result(result: dict[str, Any]) -> None:
    """Muestra el resultado de un tamaño de lote."""

    print(
        f"Batch {result['batch_size']:>3} | "
        f"mediana={result['median_ms']:>8.2f} ms | "
        f"velocidad="
        f"{result['images_per_second']:>8.2f} img/s | "
        f"VRAM asignada="
        f"{result['peak_allocated_gb']:.2f} GB | "
        f"VRAM reservada="
        f"{result['peak_reserved_gb']:.2f} GB"
    )


def main() -> None:
    """Ejecuta el benchmark completo."""

    settings = get_settings()

    manifest_path = (
        settings.manifest_path
        / "caltech256_manifest.csv"
    )

    output_path = (
        settings.manifest_path
        / "openclip_batch_benchmark.csv"
    )

    records = load_manifest(manifest_path)

    maximum_batch_size = max(BATCH_SIZES)

    if len(records) < maximum_batch_size:
        raise RuntimeError(
            "El manifiesto no contiene suficientes imágenes."
        )

    print("=" * 90)
    print("BENCHMARK DE TAMAÑOS DE LOTE — OPENCLIP")
    print("=" * 90)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
    print(
        f"VRAM total: "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    )
    print(
        f"Modelo: {settings.openclip_model}"
    )
    print(
        f"Pesos: {settings.openclip_pretrained}"
    )
    print(
        f"Repeticiones por lote: {REPETITIONS}"
    )

    service = EmbeddingService(
        settings=settings
    )

    all_tensors = prepare_tensors(
        records=records,
        image_root=settings.image_path,
        service=service,
        sample_size=maximum_batch_size,
    )

    print(
        f"\nTensor preparado: {tuple(all_tensors.shape)}"
    )

    results: list[dict[str, Any]] = []

    print("\nResultados:")

    for batch_size in BATCH_SIZES:
        try:
            result = benchmark_batch(
                service=service,
                all_tensors=all_tensors,
                batch_size=batch_size,
            )

            results.append(result)
            print_result(result)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

            result = {
                "batch_size": batch_size,
                "mean_ms": "",
                "median_ms": "",
                "min_ms": "",
                "max_ms": "",
                "images_per_second": "",
                "peak_allocated_gb": "",
                "peak_reserved_gb": "",
                "status": "out_of_memory",
            }

            results.append(result)

            print(
                f"Batch {batch_size:>3} | "
                "sin memoria CUDA"
            )

    save_results(
        results=results,
        output_path=output_path,
    )

    successful_results = [
        result
        for result in results
        if result["status"] == "ok"
    ]

    if not successful_results:
        raise RuntimeError(
            "Ningún tamaño de lote pudo ejecutarse."
        )

    best_result = max(
        successful_results,
        key=lambda item: item[
            "images_per_second"
        ],
    )

    print("\n" + "=" * 90)
    print("RESULTADO DEL BENCHMARK")
    print("=" * 90)
    print(
        "Mayor rendimiento observado: "
        f"batch={best_result['batch_size']}"
    )
    print(
        "Velocidad: "
        f"{best_result['images_per_second']:.2f} "
        "imágenes/s"
    )
    print(
        "VRAM reservada: "
        f"{best_result['peak_reserved_gb']:.2f} GB"
    )
    print(f"Reporte: {output_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()