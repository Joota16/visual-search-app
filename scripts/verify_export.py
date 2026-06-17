"""Verifica la integridad de la exportación de Caltech-256."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from src.core.config import get_settings


EXPECTED_THUMBNAIL_SIZE = (256, 256)
IMAGE_SAMPLE_SIZE = 100
RANDOM_SEED = 42


def load_manifest(
    manifest_path: Path,
) -> list[dict[str, str]]:
    """Carga todos los registros del manifiesto CSV."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No se encontró el manifiesto: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def load_dataset_report(
    report_path: Path,
) -> dict[str, Any]:
    """Carga el reporte generado durante la inspección del dataset."""
    if not report_path.exists():
        raise FileNotFoundError(
            f"No se encontró dataset_report.json: {report_path}"
        )

    with report_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(json_file)


def verify_unique_values(
    records: list[dict[str, str]],
    field_name: str,
) -> None:
    """Comprueba que un campo no contenga valores duplicados."""
    values = [
        record[field_name]
        for record in records
    ]

    unique_values = set(values)

    if len(values) != len(unique_values):
        counts = Counter(values)

        duplicates = [
            value
            for value, count in counts.items()
            if count > 1
        ]

        raise RuntimeError(
            f"Se encontraron valores duplicados en "
            f"'{field_name}'. Ejemplos: {duplicates[:5]}"
        )


def verify_split_counts(
    records: list[dict[str, str]],
    report: dict[str, Any],
) -> None:
    """Compara las cantidades de cada split con el reporte original."""
    actual_counts = Counter(
        record["split"]
        for record in records
    )

    print("\nDistribución por split:")

    for split_name, split_report in report["splits"].items():
        expected_count = int(split_report["rows"])
        actual_count = actual_counts.get(split_name, 0)

        print(
            f"- {split_name}: "
            f"{actual_count:,} / {expected_count:,}"
        )

        if actual_count != expected_count:
            raise RuntimeError(
                f"Cantidad incorrecta para '{split_name}': "
                f"{actual_count:,} != {expected_count:,}"
            )

    unexpected_splits = (
        set(actual_counts)
        - set(report["splits"])
    )

    if unexpected_splits:
        raise RuntimeError(
            "El manifiesto contiene splits inesperados: "
            f"{sorted(unexpected_splits)}"
        )


def verify_all_paths(
    records: list[dict[str, str]],
    image_root: Path,
    thumbnail_root: Path,
) -> tuple[int, int]:
    """Comprueba que todos los archivos registrados existan."""
    missing_images: list[str] = []
    missing_thumbnails: list[str] = []

    total_image_bytes = 0
    total_thumbnail_bytes = 0

    for record in tqdm(
        records,
        desc="Verificando rutas",
        unit="archivo",
    ):
        image_path = (
            image_root
            / record["image_relpath"]
        )

        thumbnail_path = (
            thumbnail_root
            / record["thumbnail_relpath"]
        )

        if not image_path.is_file():
            missing_images.append(str(image_path))
        else:
            total_image_bytes += image_path.stat().st_size

        if not thumbnail_path.is_file():
            missing_thumbnails.append(
                str(thumbnail_path)
            )
        else:
            total_thumbnail_bytes += (
                thumbnail_path.stat().st_size
            )

    print(f"\nImágenes faltantes: {len(missing_images)}")
    print(
        "Miniaturas faltantes: "
        f"{len(missing_thumbnails)}"
    )

    if missing_images:
        print("\nEjemplos de imágenes faltantes:")

        for path in missing_images[:5]:
            print(f"- {path}")

    if missing_thumbnails:
        print("\nEjemplos de miniaturas faltantes:")

        for path in missing_thumbnails[:5]:
            print(f"- {path}")

    if missing_images or missing_thumbnails:
        raise RuntimeError(
            "La exportación contiene archivos faltantes."
        )

    return total_image_bytes, total_thumbnail_bytes


def verify_image_sample(
    records: list[dict[str, str]],
    image_root: Path,
    thumbnail_root: Path,
) -> None:
    """Abre una muestra reproducible de imágenes y miniaturas."""
    random_generator = random.Random(RANDOM_SEED)

    sample_size = min(
        IMAGE_SAMPLE_SIZE,
        len(records),
    )

    sample = random_generator.sample(
        records,
        sample_size,
    )

    for record in tqdm(
        sample,
        desc="Validando imágenes",
        unit="imagen",
    ):
        image_path = (
            image_root
            / record["image_relpath"]
        )

        thumbnail_path = (
            thumbnail_root
            / record["thumbnail_relpath"]
        )

        # Verifica que el JPEG sea legible.
        with Image.open(image_path) as image:
            image.verify()

        # Se vuelve a abrir porque verify() invalida
        # el objeto para operaciones posteriores.
        with Image.open(image_path) as image:
            image.load()

            expected_width = int(record["width"])
            expected_height = int(record["height"])

            if image.size != (
                expected_width,
                expected_height,
            ):
                raise RuntimeError(
                    "Dimensión original incorrecta: "
                    f"{image_path} | "
                    f"{image.size} != "
                    f"{(expected_width, expected_height)}"
                )

            if image.mode != "RGB":
                raise RuntimeError(
                    f"Imagen no convertida a RGB: "
                    f"{image_path} | mode={image.mode}"
                )

        with Image.open(thumbnail_path) as thumbnail:
            thumbnail.load()

            if (
                thumbnail.size
                != EXPECTED_THUMBNAIL_SIZE
            ):
                raise RuntimeError(
                    "Miniatura con dimensión incorrecta: "
                    f"{thumbnail_path} | "
                    f"{thumbnail.size}"
                )

            if thumbnail.format != "WEBP":
                raise RuntimeError(
                    "Miniatura con formato incorrecto: "
                    f"{thumbnail_path} | "
                    f"{thumbnail.format}"
                )


def verify_checkpoint(
    checkpoint_path: Path,
    expected_count: int,
) -> None:
    """Verifica la cantidad de líneas válidas del checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No se encontró el checkpoint: {checkpoint_path}"
        )

    valid_records = 0
    record_keys: set[str] = set()

    with checkpoint_path.open(
        "r",
        encoding="utf-8",
    ) as checkpoint_file:
        for line_number, line in enumerate(
            checkpoint_file,
            start=1,
        ):
            clean_line = line.strip()

            if not clean_line:
                continue

            try:
                record = json.loads(clean_line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "JSON inválido en el checkpoint, "
                    f"línea {line_number}."
                ) from error

            record_key = record.get("record_key")

            if not record_key:
                raise RuntimeError(
                    "Registro sin record_key en la línea "
                    f"{line_number}."
                )

            if record_key in record_keys:
                raise RuntimeError(
                    "Record key duplicado en checkpoint: "
                    f"{record_key}"
                )

            record_keys.add(record_key)
            valid_records += 1

    print(
        "Registros válidos en checkpoint: "
        f"{valid_records:,}"
    )

    if valid_records != expected_count:
        raise RuntimeError(
            "La cantidad del checkpoint no coincide con "
            f"el manifiesto: {valid_records:,} != "
            f"{expected_count:,}"
        )


def format_size(size_in_bytes: int) -> str:
    """Convierte bytes a una representación legible."""
    return f"{size_in_bytes / (1024**3):.2f} GB"


def main() -> None:
    """Ejecuta todas las verificaciones."""
    settings = get_settings()

    manifest_path = (
        settings.manifest_path
        / "caltech256_manifest.csv"
    )

    report_path = (
        settings.manifest_path
        / "dataset_report.json"
    )

    checkpoint_path = (
        settings.manifest_path
        / "caltech256_export_progress.jsonl"
    )

    print("=" * 70)
    print("VERIFICACIÓN DE LA EXPORTACIÓN")
    print("=" * 70)
    print(f"Manifiesto: {manifest_path}")
    print(f"Imágenes: {settings.image_path}")
    print(f"Miniaturas: {settings.thumbnail_path}")

    records = load_manifest(manifest_path)
    report = load_dataset_report(report_path)

    expected_total = int(report["total_rows"])
    actual_total = len(records)

    print(f"\nRegistros esperados: {expected_total:,}")
    print(f"Registros encontrados: {actual_total:,}")

    if actual_total != expected_total:
        raise RuntimeError(
            "El total del manifiesto no coincide con "
            f"el dataset: {actual_total:,} != "
            f"{expected_total:,}"
        )

    verify_unique_values(
        records,
        "record_key",
    )

    print("Record keys únicos: correcto")

    verify_unique_values(
        records,
        "point_id",
    )

    print("Point IDs únicos: correcto")

    verify_split_counts(
        records=records,
        report=report,
    )

    total_image_bytes, total_thumbnail_bytes = (
        verify_all_paths(
            records=records,
            image_root=settings.image_path,
            thumbnail_root=settings.thumbnail_path,
        )
    )

    verify_image_sample(
        records=records,
        image_root=settings.image_path,
        thumbnail_root=settings.thumbnail_path,
    )

    print(
        f"Muestra de {min(IMAGE_SAMPLE_SIZE, actual_total)} "
        "imágenes válida: correcto"
    )

    verify_checkpoint(
        checkpoint_path=checkpoint_path,
        expected_count=actual_total,
    )

    categories = {
        record["label"]
        for record in records
    }

    print(f"\nCategorías encontradas: {len(categories):,}")
    print(
        "Tamaño total de imágenes: "
        f"{format_size(total_image_bytes)}"
    )
    print(
        "Tamaño total de miniaturas: "
        f"{format_size(total_thumbnail_bytes)}"
    )

    print("=" * 70)
    print("EXPORTACIÓN VERIFICADA CORRECTAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    main()