"""Genera embeddings OpenCLIP para todo Caltech-256."""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.core.config import get_settings
from src.data.manifest_dataset import ManifestImageDataset
from src.services.embedding_service import EmbeddingService


EMBEDDING_FILENAME = (
    "caltech256_vit_b32_laion2b_s34b_b79k.npy"
)

PROGRESS_FILENAME = "embedding_progress.json"
REPORT_FILENAME = "embedding_report.json"

VALIDATION_CHUNK_SIZE = 4096


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
            "El manifiesto no contiene registros."
        )

    return records


def write_json_atomically(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """Escribe un JSON utilizando un archivo temporal."""
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


def load_progress(
    progress_path: Path,
) -> dict[str, Any] | None:
    """Carga el progreso de una ejecución anterior."""
    if not progress_path.exists():
        return None

    with progress_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(json_file)


def create_dataloader(
    dataset: ManifestImageDataset,
    start_index: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    """Construye un DataLoader para las imágenes pendientes."""
    remaining_indices = range(
        start_index,
        len(dataset),
    )

    subset = Subset(
        dataset,
        remaining_indices,
    )

    loader_options: dict[str, Any] = {
        "dataset": subset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }

    if num_workers > 0:
        loader_options["prefetch_factor"] = (
            prefetch_factor
        )
        loader_options["persistent_workers"] = True

    return DataLoader(**loader_options)


def validate_embedding_file(
    embedding_path: Path,
    expected_rows: int,
    expected_dimension: int,
) -> tuple[float, float]:
    """Valida forma, valores y normalización."""
    embeddings = np.load(
        embedding_path,
        mmap_mode="r",
    )

    expected_shape = (
        expected_rows,
        expected_dimension,
    )

    if embeddings.shape != expected_shape:
        raise RuntimeError(
            "Forma incorrecta del archivo de embeddings: "
            f"{embeddings.shape} != {expected_shape}"
        )

    if embeddings.dtype != np.float32:
        raise RuntimeError(
            "El archivo debe utilizar float32, "
            f"pero utiliza {embeddings.dtype}."
        )

    minimum_norm = float("inf")
    maximum_norm = float("-inf")

    for start in tqdm(
        range(
            0,
            expected_rows,
            VALIDATION_CHUNK_SIZE,
        ),
        desc="Validando embeddings",
        unit="vector",
    ):
        chunk = np.asarray(
            embeddings[
                start : start + VALIDATION_CHUNK_SIZE
            ]
        )

        if not np.isfinite(chunk).all():
            raise RuntimeError(
                "Los embeddings contienen NaN o infinito."
            )

        norms = np.linalg.norm(
            chunk,
            axis=1,
        )

        minimum_norm = min(
            minimum_norm,
            float(norms.min()),
        )

        maximum_norm = max(
            maximum_norm,
            float(norms.max()),
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-3,
        ):
            raise RuntimeError(
                "Se encontraron embeddings "
                "sin normalizar."
            )

    return minimum_norm, maximum_norm


def main() -> None:
    """Genera todos los embeddings del dataset."""
    settings = get_settings()

    manifest_path = (
        settings.manifest_path
        / "caltech256_manifest.csv"
    )

    settings.embedding_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_path = (
        settings.embedding_path
        / EMBEDDING_FILENAME
    )

    progress_path = (
        settings.manifest_path
        / PROGRESS_FILENAME
    )

    report_path = (
        settings.manifest_path
        / REPORT_FILENAME
    )

    records = load_manifest(manifest_path)
    total_rows = len(records)

    print("=" * 80)
    print("GENERACIÓN DE EMBEDDINGS — OPENCLIP")
    print("=" * 80)
    print(f"Registros: {total_rows:,}")
    print(f"Batch size: {settings.batch_size}")
    print(f"Workers: {settings.num_workers}")
    print(f"Modelo: {settings.openclip_model}")
    print(f"Pesos: {settings.openclip_pretrained}")
    print(f"Salida: {embedding_path}")

    service = EmbeddingService(
        settings=settings
    )

    dataset = ManifestImageDataset(
        records=records,
        image_root=settings.image_path,
        transform=service.preprocess,
    )

    # Genera un embedding de prueba para obtener
    # automáticamente la dimensión del modelo.
    sample_tensor, _ = dataset[0]

    sample_embedding = (
        service.encode_image_tensors(
            sample_tensor.unsqueeze(0)
        )
    )

    embedding_dimension = int(
        sample_embedding.shape[1]
    )

    if embedding_dimension != 512:
        raise RuntimeError(
            "Se esperaba una dimensión de 512, "
            f"pero se obtuvo {embedding_dimension}."
        )

    previous_progress = load_progress(
        progress_path
    )

    if previous_progress is None:
        if embedding_path.exists():
            raise RuntimeError(
                "Existe un archivo de embeddings, "
                "pero no existe un archivo de progreso. "
                "No se sobrescribirá automáticamente."
            )

        embedding_matrix = (
            np.lib.format.open_memmap(
                embedding_path,
                mode="w+",
                dtype=np.float32,
                shape=(
                    total_rows,
                    embedding_dimension,
                ),
            )
        )

        start_index = 0

    else:
        if not embedding_path.exists():
            raise RuntimeError(
                "Existe progreso registrado, pero no "
                "existe el archivo de embeddings."
            )

        if int(
            previous_progress["total_rows"]
        ) != total_rows:
            raise RuntimeError(
                "El progreso pertenece a un "
                "manifiesto diferente."
            )

        if int(
            previous_progress[
                "embedding_dimension"
            ]
        ) != embedding_dimension:
            raise RuntimeError(
                "La dimensión del progreso no coincide "
                "con el modelo actual."
            )

        if (
            previous_progress["model"]
            != settings.openclip_model
            or previous_progress["pretrained"]
            != settings.openclip_pretrained
        ):
            raise RuntimeError(
                "El progreso pertenece a otro modelo."
            )

        start_index = int(
            previous_progress["next_index"]
        )

        embedding_matrix = (
            np.lib.format.open_memmap(
                embedding_path,
                mode="r+",
            )
        )

        expected_shape = (
            total_rows,
            embedding_dimension,
        )

        if embedding_matrix.shape != expected_shape:
            raise RuntimeError(
                "El archivo existente tiene una forma "
                f"incorrecta: {embedding_matrix.shape}"
            )

    if start_index < 0 or start_index > total_rows:
        raise RuntimeError(
            f"Índice de reanudación inválido: {start_index}"
        )

    print(f"Dimensión: {embedding_dimension}")
    print(f"Inicio/reanudación: {start_index:,}")
    print("=" * 80)

    if start_index < total_rows:
        data_loader = create_dataloader(
            dataset=dataset,
            start_index=start_index,
            batch_size=settings.batch_size,
            num_workers=settings.num_workers,
            prefetch_factor=settings.prefetch_factor,
        )

        total_batches = math.ceil(
            (total_rows - start_index)
            / settings.batch_size
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        generation_start = time.perf_counter()
        processed_now = 0

        for image_batch, row_indices in tqdm(
            data_loader,
            total=total_batches,
            desc="Generando embeddings",
            unit="lote",
        ):
            batch_embeddings = (
                service.encode_image_tensors(
                    image_batch
                )
            )

            indices = (
                row_indices
                .cpu()
                .numpy()
                .astype(np.int64)
            )

            if batch_embeddings.shape[0] != len(
                indices
            ):
                raise RuntimeError(
                    "La cantidad de embeddings no coincide "
                    "con los índices del lote."
                )

            embedding_matrix[indices] = (
                batch_embeddings
            )

            # Se guarda cada lote para garantizar
            # que el checkpoint sea reanudable.
            embedding_matrix.flush()

            next_index = int(indices[-1]) + 1
            processed_now += len(indices)

            write_json_atomically(
                {
                    "model": settings.openclip_model,
                    "pretrained": (
                        settings.openclip_pretrained
                    ),
                    "total_rows": total_rows,
                    "embedding_dimension": (
                        embedding_dimension
                    ),
                    "next_index": next_index,
                    "completed": False,
                    "updated_at_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                },
                progress_path,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_seconds = (
            time.perf_counter()
            - generation_start
        )

    else:
        processed_now = 0
        elapsed_seconds = 0.0

    embedding_matrix.flush()
    del embedding_matrix

    minimum_norm, maximum_norm = (
        validate_embedding_file(
            embedding_path=embedding_path,
            expected_rows=total_rows,
            expected_dimension=embedding_dimension,
        )
    )

    peak_allocated_gb = 0.0
    peak_reserved_gb = 0.0

    if torch.cuda.is_available():
        peak_allocated_gb = (
            torch.cuda.max_memory_allocated()
            / (1024**3)
        )

        peak_reserved_gb = (
            torch.cuda.max_memory_reserved()
            / (1024**3)
        )

    images_per_second = (
        processed_now / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )

    final_progress = {
        "model": settings.openclip_model,
        "pretrained": settings.openclip_pretrained,
        "total_rows": total_rows,
        "embedding_dimension": embedding_dimension,
        "next_index": total_rows,
        "completed": True,
        "updated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    write_json_atomically(
        final_progress,
        progress_path,
    )

    report = {
        **final_progress,
        "embedding_file": str(embedding_path),
        "embedding_dtype": "float32",
        "embedding_shape": [
            total_rows,
            embedding_dimension,
        ],
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "processed_in_current_run": processed_now,
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": images_per_second,
        "minimum_norm": minimum_norm,
        "maximum_norm": maximum_norm,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "file_size_mb": (
            embedding_path.stat().st_size
            / (1024**2)
        ),
    }

    write_json_atomically(
        report,
        report_path,
    )

    print("\n" + "=" * 80)
    print("RESULTADO")
    print("=" * 80)
    print(
        f"Embeddings generados: {total_rows:,}"
    )
    print(
        f"Forma: ({total_rows:,}, "
        f"{embedding_dimension})"
    )
    print("Tipo: float32")
    print(
        f"Normas: {minimum_norm:.6f} "
        f"– {maximum_norm:.6f}"
    )
    print(
        f"Procesados en esta ejecución: "
        f"{processed_now:,}"
    )

    if elapsed_seconds > 0:
        print(
            f"Tiempo: {elapsed_seconds:.2f} segundos"
        )
        print(
            f"Rendimiento integral: "
            f"{images_per_second:.2f} imágenes/s"
        )

    print(
        f"VRAM máxima asignada: "
        f"{peak_allocated_gb:.2f} GB"
    )
    print(
        f"VRAM máxima reservada: "
        f"{peak_reserved_gb:.2f} GB"
    )
    print(
        f"Tamaño del archivo: "
        f"{report['file_size_mb']:.2f} MB"
    )
    print(f"Archivo: {embedding_path}")
    print(f"Reporte: {report_path}")
    print("=" * 80)
    print("EMBEDDINGS GENERADOS Y VALIDADOS")
    print("=" * 80)


if __name__ == "__main__":
    main()