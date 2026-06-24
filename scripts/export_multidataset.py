"""Exporta múltiples datasets y construye un manifiesto unificado."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import os
from pathlib import Path

HF_ROOT = Path(r"C:\hf-cache")

os.environ.setdefault("HF_HOME", str(HF_ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(HF_ROOT / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_ROOT / "datasets"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from datasets import DatasetDict, load_dataset
from PIL import Image, ImageOps
from tqdm import tqdm

from src.core.config import get_settings


POINT_NAMESPACE = UUID("5d6d975e-9d80-46a9-9e36-35f0c9e6ef51")

THUMBNAIL_SIZE = (256, 256)
JPEG_QUALITY = 95
WEBP_QUALITY = 82

MANIFEST_FIELDS = [
    "record_key",
    "point_id",
    "dataset_id",
    "dataset_name",
    "domain",
    "source_dataset",
    "split",
    "row_index",
    "label_id",
    "label",
    "image_relpath",
    "thumbnail_relpath",
    "width",
    "height",
    "image_bytes",
    "thumbnail_bytes",
]


def slugify(value: str, fallback: str) -> str:
    """Convierte texto en un nombre seguro para carpetas."""
    normalized = unicodedata.normalize("NFKD", value)

    ascii_value = normalized.encode(
        "ascii",
        errors="ignore",
    ).decode("ascii")

    slug = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "-",
        ascii_value,
    )

    slug = slug.strip("-_.").lower()

    return slug or fallback


def create_point_id(
    dataset_id: str,
    split_name: str,
    row_index: int,
) -> str:
    """Crea un UUID determinista para Qdrant."""
    source = f"{dataset_id}:{split_name}:{row_index}"

    return str(
        uuid5(
            POINT_NAMESPACE,
            source,
        )
    )


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """Corrige orientación y convierte a RGB."""
    corrected = ImageOps.exif_transpose(image)

    if corrected.mode in {"RGBA", "LA"} or (
        "transparency" in corrected.info
    ):
        rgba_image = corrected.convert("RGBA")

        background = Image.new(
            mode="RGB",
            size=rgba_image.size,
            color="white",
        )

        background.paste(
            rgba_image,
            mask=rgba_image.getchannel("A"),
        )

        rgba_image.close()

        return background

    return corrected.convert("RGB")


def save_image_atomically(
    image: Image.Image,
    destination: Path,
    image_format: str,
    **save_options: Any,
) -> None:
    """Guarda una imagen usando archivo temporal."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    image.save(
        temporary_path,
        format=image_format,
        **save_options,
    )

    temporary_path.replace(destination)


def load_checkpoint(
    checkpoint_path: Path,
) -> dict[str, dict[str, Any]]:
    """Carga registros exportados en ejecuciones anteriores."""
    records: dict[str, dict[str, Any]] = {}

    if not checkpoint_path.exists():
        return records

    with checkpoint_path.open(
        "r",
        encoding="utf-8",
    ) as checkpoint_file:
        for line in checkpoint_file:
            clean_line = line.strip()

            if not clean_line:
                continue

            try:
                record = json.loads(clean_line)
                records[record["record_key"]] = record

            except Exception:
                continue

    return records


def append_checkpoint(
    checkpoint_file: Any,
    record: dict[str, Any],
) -> None:
    """Agrega un registro al checkpoint."""
    checkpoint_file.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )

    checkpoint_file.flush()


def write_manifest(
    records: dict[str, dict[str, Any]],
    manifest_path: Path,
) -> None:
    """Escribe el manifiesto unificado."""
    ordered_records = sorted(
        records.values(),
        key=lambda item: (
            item["dataset_id"],
            item["split"],
            int(item["row_index"]),
        ),
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = manifest_path.with_suffix(
        ".csv.part"
    )

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=MANIFEST_FIELDS,
        )

        writer.writeheader()
        writer.writerows(ordered_records)

    temporary_path.replace(manifest_path)


def get_dataset_configs() -> dict[str, dict[str, Any]]:
    """Define los datasets disponibles."""
    settings = get_settings()

    return {
        "caltech256": {
            "dataset_id": "caltech256",
            "dataset_name": "Caltech-256",
            "domain": "general_objects",
            "source_dataset": settings.caltech256_dataset,
            "limit": settings.caltech256_limit,
        },
        "food101": {
            "dataset_id": "food101",
            "dataset_name": "Food-101",
            "domain": "food",
            "source_dataset": settings.food101_dataset,
            "limit": settings.food101_limit,
        },
        "product_images": {
            "dataset_id": "product_images",
            "dataset_name": "Product Images",
            "domain": "products",
            "source_dataset": settings.product_images_dataset,
            "limit": settings.product_images_limit,
        },
        "tiny_imagenet": {
            "dataset_id": "tiny_imagenet",
            "dataset_name": "Tiny ImageNet",
            "domain": "general_images",
            "source_dataset": settings.tiny_imagenet_dataset,
            "limit": settings.tiny_imagenet_limit,
        },
    }


def find_image_column(row: dict[str, Any]) -> str:
    """Encuentra la columna que contiene la imagen."""
    if "image" in row and isinstance(row["image"], Image.Image):
        return "image"

    for column_name, value in row.items():
        if isinstance(value, Image.Image):
            return column_name

    raise RuntimeError(
        "No se encontró una columna de imagen compatible."
    )


def get_label_from_row(
    row: dict[str, Any],
    label_names: list[str] | None,
) -> tuple[int, str]:
    """Obtiene label_id y label textual de una fila."""
    label_id = -1
    label_text = ""

    if "label" in row:
        raw_label = row["label"]

        if isinstance(raw_label, int):
            label_id = int(raw_label)

            if (
                label_names is not None
                and 0 <= label_id < len(label_names)
            ):
                label_text = str(label_names[label_id])

        else:
            label_text = str(raw_label)

    if not label_text and "text" in row:
        label_text = str(row["text"])

    if not label_text and "articleType" in row:
        label_text = str(row["articleType"])

    if not label_text and "productDisplayName" in row:
        label_text = str(row["productDisplayName"])

    if not label_text and "class" in row:
        label_text = str(row["class"])

    if not label_text and "category" in row:
        label_text = str(row["category"])

    if not label_text:
        label_text = "unknown"

    return label_id, label_text


def get_label_names(split_dataset: Any) -> list[str] | None:
    """Obtiene nombres de clases cuando el dataset los expone."""
    if "label" not in split_dataset.features:
        return None

    label_feature = split_dataset.features["label"]

    names = getattr(
        label_feature,
        "names",
        None,
    )

    if names is None:
        return None

    return list(names)


def build_record(
    dataset_config: dict[str, Any],
    split_name: str,
    row_index: int,
    label_id: int,
    label: str,
    image_relpath: Path,
    thumbnail_relpath: Path,
    width: int,
    height: int,
    image_path: Path,
    thumbnail_path: Path,
) -> dict[str, Any]:
    """Construye una fila del manifiesto."""
    dataset_id = str(dataset_config["dataset_id"])

    record_key = (
        f"{dataset_id}:{split_name}:{row_index}"
    )

    return {
        "record_key": record_key,
        "point_id": create_point_id(
            dataset_id=dataset_id,
            split_name=split_name,
            row_index=row_index,
        ),
        "dataset_id": dataset_id,
        "dataset_name": dataset_config["dataset_name"],
        "domain": dataset_config["domain"],
        "source_dataset": dataset_config["source_dataset"],
        "split": split_name,
        "row_index": row_index,
        "label_id": label_id,
        "label": label,
        "image_relpath": image_relpath.as_posix(),
        "thumbnail_relpath": thumbnail_relpath.as_posix(),
        "width": width,
        "height": height,
        "image_bytes": image_path.stat().st_size,
        "thumbnail_bytes": thumbnail_path.stat().st_size,
    }


def export_one_dataset(
    dataset_key: str,
    dataset_config: dict[str, Any],
    records: dict[str, dict[str, Any]],
    checkpoint_file: Any,
) -> tuple[int, int, int]:
    """Exporta un dataset específico."""
    settings = get_settings()

    source_dataset = str(
        dataset_config["source_dataset"]
    )

    dataset_id = str(
        dataset_config["dataset_id"]
    )

    limit = int(
        dataset_config.get("limit", 0) or 0
    )

    print("\n" + "=" * 80)
    print(f"EXPORTANDO DATASET: {dataset_config['dataset_name']}")
    print("=" * 80)
    print(f"Clave: {dataset_key}")
    print(f"HF dataset: {source_dataset}")
    print(f"Dataset ID interno: {dataset_id}")
    print(f"Dominio: {dataset_config['domain']}")

    dataset = load_dataset(
        source_dataset,
        cache_dir=str(settings.hf_cache_path),
    )

    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"El dataset {source_dataset} no devolvió un DatasetDict."
        )

    total_rows = sum(
        len(split_dataset)
        for split_dataset in dataset.values()
    )

    target_rows = (
        min(limit, total_rows)
        if limit > 0
        else total_rows
    )

    print(f"Filas disponibles: {total_rows:,}")
    print(f"Filas objetivo: {target_rows:,}")

    processed = 0
    newly_exported = 0
    reused = 0

    with tqdm(
        total=target_rows,
        desc=f"Exportando {dataset_id}",
        unit="img",
    ) as progress_bar:
        for split_name, split_dataset in dataset.items():
            label_names = get_label_names(
                split_dataset
            )

            for row_index in range(len(split_dataset)):
                if processed >= target_rows:
                    break

                record_key = (
                    f"{dataset_id}:{split_name}:{row_index}"
                )

                previous_record = records.get(
                    record_key
                )

                if previous_record is not None:
                    image_path = (
                        settings.image_path
                        / previous_record["image_relpath"]
                    )

                    thumbnail_path = (
                        settings.thumbnail_path
                        / previous_record["thumbnail_relpath"]
                    )

                    if (
                        image_path.exists()
                        and thumbnail_path.exists()
                    ):
                        reused += 1
                        processed += 1
                        progress_bar.update(1)
                        continue

                row = split_dataset[row_index]

                image_column = find_image_column(row)
                source_image = row[image_column]

                label_id, label = get_label_from_row(
                    row=row,
                    label_names=label_names,
                )

                label_slug = slugify(
                    label,
                    fallback=f"label-{label_id}",
                )

                filename = f"{row_index:06d}"

                image_relpath = (
                    Path(dataset_id)
                    / split_name
                    / label_slug
                    / f"{filename}.jpg"
                )

                thumbnail_relpath = (
                    Path(dataset_id)
                    / split_name
                    / label_slug
                    / f"{filename}.webp"
                )

                image_path = (
                    settings.image_path
                    / image_relpath
                )

                thumbnail_path = (
                    settings.thumbnail_path
                    / thumbnail_relpath
                )

                rgb_image: Image.Image | None = None
                thumbnail: Image.Image | None = None

                try:
                    rgb_image = convert_to_rgb(
                        source_image
                    )

                    width, height = rgb_image.size

                    save_image_atomically(
                        image=rgb_image,
                        destination=image_path,
                        image_format="JPEG",
                        quality=JPEG_QUALITY,
                        optimize=False,
                    )

                    thumbnail = ImageOps.fit(
                        rgb_image,
                        THUMBNAIL_SIZE,
                        method=Image.Resampling.LANCZOS,
                    )

                    save_image_atomically(
                        image=thumbnail,
                        destination=thumbnail_path,
                        image_format="WEBP",
                        quality=WEBP_QUALITY,
                        method=4,
                    )

                    record = build_record(
                        dataset_config=dataset_config,
                        split_name=split_name,
                        row_index=row_index,
                        label_id=label_id,
                        label=label,
                        image_relpath=image_relpath,
                        thumbnail_relpath=thumbnail_relpath,
                        width=width,
                        height=height,
                        image_path=image_path,
                        thumbnail_path=thumbnail_path,
                    )

                    records[record_key] = record

                    append_checkpoint(
                        checkpoint_file=checkpoint_file,
                        record=record,
                    )

                    newly_exported += 1

                except Exception:
                    image_path.unlink(
                        missing_ok=True
                    )

                    thumbnail_path.unlink(
                        missing_ok=True
                    )

                    raise

                finally:
                    if thumbnail is not None:
                        thumbnail.close()

                    if rgb_image is not None:
                        rgb_image.close()

                    if isinstance(
                        source_image,
                        Image.Image,
                    ):
                        source_image.close()

                processed += 1
                progress_bar.update(1)

            if processed >= target_rows:
                break

    return processed, newly_exported, reused


def main() -> None:
    """Ejecuta la exportación multidataset."""
    settings = get_settings()

    settings.hf_cache_path.mkdir(
    parents=True,
    exist_ok=True,
    )

    settings.hf_home_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.hf_hub_cache_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.image_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.thumbnail_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.manifest_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        settings.manifest_path
        / settings.multidataset_manifest_filename
    )

    checkpoint_path = (
        settings.manifest_path
        / "visual_search_multidataset_export_progress.jsonl"
    )

    dataset_configs = get_dataset_configs()

    selected_sources = settings.dataset_source_list

    print("=" * 80)
    print("EXPORTACIÓN MULTI-DATASET")
    print("=" * 80)
    print(f"Modo: {settings.dataset_mode}")
    print(f"Datasets seleccionados: {selected_sources}")
    print(f"Imágenes: {settings.image_path}")
    print(f"Miniaturas: {settings.thumbnail_path}")
    print(f"Manifiesto: {manifest_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)

    records = load_checkpoint(
        checkpoint_path
    )

    total_processed = 0
    total_new = 0
    total_reused = 0

    with checkpoint_path.open(
        "a",
        encoding="utf-8",
    ) as checkpoint_file:
        for dataset_key in selected_sources:
            if dataset_key not in dataset_configs:
                raise ValueError(
                    f"Dataset no reconocido en DATASET_SOURCES: {dataset_key}"
                )

            processed, newly_exported, reused = (
                export_one_dataset(
                    dataset_key=dataset_key,
                    dataset_config=dataset_configs[dataset_key],
                    records=records,
                    checkpoint_file=checkpoint_file,
                )
            )

            total_processed += processed
            total_new += newly_exported
            total_reused += reused

    write_manifest(
        records=records,
        manifest_path=manifest_path,
    )

    print("\n" + "=" * 80)
    print("RESULTADO DE EXPORTACIÓN MULTI-DATASET")
    print("=" * 80)
    print(f"Procesadas en esta ejecución: {total_processed:,}")
    print(f"Nuevas exportaciones: {total_new:,}")
    print(f"Reutilizadas: {total_reused:,}")
    print(f"Registros totales en manifiesto: {len(records):,}")
    print(f"Manifiesto: {manifest_path}")
    print("=" * 80)
    print("EXPORTACIÓN MULTI-DATASET COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()