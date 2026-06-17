"""Descarga e inspecciona la estructura del dataset Caltech-256."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Debe configurarse antes de importar datasets/huggingface_hub.
HF_ROOT = Path(r"C:\hf")
HF_DATASETS_CACHE = HF_ROOT / "datasets"
HF_HUB_CACHE = HF_ROOT / "hub"

os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)

from datasets import DatasetDict, load_dataset
from PIL import Image

from src.core.config import get_settings


EXPECTED_COLUMNS = {"image", "label", "text"}


def prepare_directories(
    cache_directory: Path,
    manifest_directory: Path,
) -> Path:
    """Crea las carpetas necesarias para la inspección."""
    cache_directory.mkdir(parents=True, exist_ok=True)
    manifest_directory.mkdir(parents=True, exist_ok=True)

    samples_directory = manifest_directory / "samples"
    samples_directory.mkdir(parents=True, exist_ok=True)

    return samples_directory


def inspect_label_mapping(
    dataset: DatasetDict,
) -> tuple[dict[int, str], list[str]]:
    """Verifica que cada label numérico corresponda a un solo texto."""
    label_text_values: dict[int, set[str]] = defaultdict(set)

    for split_name, split_dataset in dataset.items():
        labels = split_dataset["label"]
        texts = split_dataset["text"]

        if len(labels) != len(texts):
            raise RuntimeError(
                f"El split {split_name} tiene cantidades diferentes "
                "de labels y textos."
            )

        for label, text in zip(labels, texts, strict=True):
            label_text_values[int(label)].add(str(text))

    conflicts: list[str] = []
    mapping: dict[int, str] = {}

    for label, text_values in sorted(label_text_values.items()):
        if len(text_values) != 1:
            conflicts.append(
                f"Label {label}: {sorted(text_values)}"
            )
            continue

        mapping[label] = next(iter(text_values))

    return mapping, conflicts


def inspect_split(
    split_name: str,
    split_dataset: Any,
    samples_directory: Path,
) -> dict[str, Any]:
    """Inspecciona un split y guarda su primera imagen como muestra."""
    columns = set(split_dataset.column_names)
    missing_columns = EXPECTED_COLUMNS - columns

    if missing_columns:
        raise RuntimeError(
            f"El split '{split_name}' no contiene las columnas "
            f"esperadas: {sorted(missing_columns)}"
        )

    if len(split_dataset) == 0:
        raise RuntimeError(
            f"El split '{split_name}' está vacío."
        )

    # Accedemos primero a una fila y después a su imagen.
    # Así evitamos solicitar la decodificación de toda la columna.
    first_row = split_dataset[0]
    image = first_row["image"]

    if not isinstance(image, Image.Image):
        raise TypeError(
            f"La columna image de '{split_name}' no devolvió "
            "un objeto PIL.Image."
        )

    sample_path = samples_directory / f"{split_name}_sample.jpg"

    image.convert("RGB").save(
        sample_path,
        format="JPEG",
        quality=95,
    )

    return {
        "name": split_name,
        "rows": len(split_dataset),
        "columns": split_dataset.column_names,
        "features": {
            name: str(feature)
            for name, feature in split_dataset.features.items()
        },
        "sample": {
            "label": int(first_row["label"]),
            "text": str(first_row["text"]),
            "width": int(image.width),
            "height": int(image.height),
            "mode": str(image.mode),
            "saved_path": str(sample_path),
        },
    }


def save_report(
    report: dict[str, Any],
    manifest_directory: Path,
) -> Path:
    """Guarda el reporte de inspección como JSON."""
    report_path = manifest_directory / "dataset_report.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return report_path


def main() -> None:
    """Descarga e inspecciona Caltech-256."""
    settings = get_settings()

    samples_directory = prepare_directories(
        cache_directory=HF_DATASETS_CACHE,
        manifest_directory=settings.manifest_path,
    )

    print("=" * 70)
    print("DESCARGA E INSPECCIÓN DE CALTECH-256")
    print("=" * 70)
    print(f"Dataset: {settings.hf_dataset}")
    print(f"Caché: {HF_DATASETS_CACHE}")
    print()
    print("Descargando o reutilizando archivos en caché...")

    dataset = load_dataset(
        settings.hf_dataset,
        cache_dir=str(HF_DATASETS_CACHE),
    )

    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            "Se esperaba que load_dataset devolviera un DatasetDict."
        )

    print("\nDataset cargado correctamente.")
    print(f"Splits encontrados: {list(dataset.keys())}")

    split_reports: dict[str, dict[str, Any]] = {}
    total_rows = 0

    for split_name, split_dataset in dataset.items():
        split_report = inspect_split(
            split_name=split_name,
            split_dataset=split_dataset,
            samples_directory=samples_directory,
        )

        split_reports[split_name] = split_report
        total_rows += split_report["rows"]

        print("-" * 70)
        print(f"Split: {split_name}")
        print(f"Filas: {split_report['rows']:,}")
        print(f"Columnas: {split_report['columns']}")
        print(
            "Primera muestra: "
            f"label={split_report['sample']['label']} | "
            f"text={split_report['sample']['text']} | "
            f"size={split_report['sample']['width']}x"
            f"{split_report['sample']['height']} | "
            f"mode={split_report['sample']['mode']}"
        )

    print("\nVerificando la relación label ↔ text...")

    label_mapping, conflicts = inspect_label_mapping(dataset)

    if conflicts:
        print("\nSe encontraron conflictos:")

        for conflict in conflicts:
            print(f"- {conflict}")

        raise RuntimeError(
            "La relación entre label y text no es consistente."
        )

    report = {
        "dataset_id": settings.hf_dataset,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_rows": total_rows,
        "split_count": len(dataset),
        "splits": split_reports,
        "unique_labels": len(label_mapping),
        "label_mapping": {
            str(label): text
            for label, text in label_mapping.items()
        },
    }

    report_path = save_report(
        report=report,
        manifest_directory=settings.manifest_path,
    )

    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Total de imágenes: {total_rows:,}")
    print(f"Cantidad de splits: {len(dataset)}")
    print(f"Categorías únicas: {len(label_mapping)}")
    print(f"Conflictos label ↔ text: {len(conflicts)}")
    print(f"Reporte: {report_path}")
    print(f"Muestras: {samples_directory}")
    print("=" * 70)
    print("DATASET DESCARGADO E INSPECCIONADO CORRECTAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    main()