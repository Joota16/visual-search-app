"""Exporta Caltech-256, genera miniaturas y construye el manifiesto."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from datasets import DatasetDict, load_dataset
from PIL import Image, ImageOps
from tqdm import tqdm

from src.core.config import get_settings


POINT_NAMESPACE = UUID("d7f1869c-a13a-4ccb-ae16-5a38bf9b9d77")

THUMBNAIL_SIZE = (256, 256)
JPEG_QUALITY = 95
WEBP_QUALITY = 82

MANIFEST_FIELDS = [
    "record_key",
    "point_id",
    "dataset_id",
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


def parse_arguments() -> argparse.Namespace:
    """Lee los argumentos enviados desde la terminal."""
    parser = argparse.ArgumentParser(
        description=(
            "Exporta imágenes de Caltech-256 y genera miniaturas."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cantidad máxima de imágenes que se procesarán. "
            "Si se omite, procesa el dataset completo."
        ),
    )

    return parser.parse_args()


def slugify(value: str, fallback: str) -> str:
    """Convierte el nombre de una categoría en un nombre de carpeta seguro."""
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
    """Genera un UUID determinista para utilizarlo posteriormente en Qdrant."""
    source = f"{dataset_id}:{split_name}:{row_index}"

    return str(
        uuid5(
            POINT_NAMESPACE,
            source,
        )
    )


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """Corrige la orientación y convierte una imagen a RGB."""
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
    """Guarda primero en un archivo temporal y luego crea el archivo final."""
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
    """Recupera las imágenes procesadas en ejecuciones anteriores."""
    records: dict[str, dict[str, Any]] = {}

    if not checkpoint_path.exists():
        return records

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
                records[record["record_key"]] = record

            except (json.JSONDecodeError, KeyError):
                print(
                    "Advertencia: se ignoró una línea inválida "
                    f"del checkpoint: {line_number}"
                )

    return records


def append_checkpoint(
    checkpoint_file: Any,
    record: dict[str, Any],
) -> None:
    """Añade inmediatamente un registro al archivo de progreso."""
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
    """Construye el manifiesto CSV final."""
    ordered_records = sorted(
        records.values(),
        key=lambda item: (
            item["split"],
            int(item["row_index"]),
        ),
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


def build_record(
    dataset_id: str,
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
    """Construye un registro del manifiesto."""
    return {
        "record_key": f"{split_name}:{row_index}",
        "point_id": create_point_id(
            dataset_id,
            split_name,
            row_index,
        ),
        "dataset_id": dataset_id,
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


def process_dataset(
    dataset: DatasetDict,
    limit: int | None,
) -> None:
    """Exporta las imágenes, miniaturas y metadatos."""
    settings = get_settings()

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
        / "caltech256_manifest.csv"
    )

    checkpoint_path = (
        settings.manifest_path
        / "caltech256_export_progress.jsonl"
    )

    records = load_checkpoint(checkpoint_path)

    total_dataset_rows = sum(
        len(split_dataset)
        for split_dataset in dataset.values()
    )

    if limit is not None and limit <= 0:
        raise ValueError(
            "--limit debe ser mayor que cero."
        )

    target_rows = (
        min(limit, total_dataset_rows)
        if limit is not None
        else total_dataset_rows
    )

    processed_in_current_run = 0
    newly_exported = 0
    reused = 0

    print("=" * 70)
    print("EXPORTACIÓN DE CALTECH-256")
    print("=" * 70)
    print(f"Dataset: {settings.hf_dataset}")
    print(f"Imágenes: {settings.image_path}")
    print(f"Miniaturas: {settings.thumbnail_path}")
    print(f"Manifiesto: {manifest_path}")
    print(f"Registros recuperados: {len(records):,}")
    print(f"Objetivo de esta ejecución: {target_rows:,}")
    print("=" * 70)

    with checkpoint_path.open(
        "a",
        encoding="utf-8",
    ) as checkpoint_file:
        with tqdm(
            total=target_rows,
            desc="Exportando",
            unit="img",
        ) as progress_bar:
            for split_name, split_dataset in dataset.items():
                metadata_dataset = split_dataset.remove_columns(
                    ["image"]
                )

                for row_index in range(len(split_dataset)):
                    if processed_in_current_run >= target_rows:
                        break

                    metadata_row = metadata_dataset[row_index]

                    label_id = int(metadata_row["label"])
                    label = str(metadata_row["text"])

                    label_slug = slugify(
                        label,
                        fallback=f"label-{label_id:03d}",
                    )

                    filename = f"{row_index:06d}"

                    image_relpath = (
                        Path(split_name)
                        / label_slug
                        / f"{filename}.jpg"
                    )

                    thumbnail_relpath = (
                        Path(split_name)
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

                    record_key = (
                        f"{split_name}:{row_index}"
                    )

                    previous_record = records.get(
                        record_key
                    )

                    if (
                        previous_record is not None
                        and image_path.exists()
                        and thumbnail_path.exists()
                    ):
                        reused += 1
                        processed_in_current_run += 1
                        progress_bar.update(1)
                        continue

                    source_image = split_dataset[
                        row_index
                    ]["image"]

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

                        source_image.close()

                    record = build_record(
                        dataset_id=settings.hf_dataset,
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
                        checkpoint_file,
                        record,
                    )

                    newly_exported += 1
                    processed_in_current_run += 1
                    progress_bar.update(1)

                if processed_in_current_run >= target_rows:
                    break

    write_manifest(
        records=records,
        manifest_path=manifest_path,
    )

    print()
    print("=" * 70)
    print("RESULTADO DE LA EXPORTACIÓN")
    print("=" * 70)
    print(
        "Procesadas en esta ejecución: "
        f"{processed_in_current_run:,}"
    )
    print(
        f"Nuevas exportaciones: {newly_exported:,}"
    )
    print(f"Reutilizadas: {reused:,}")
    print(
        "Registros totales en manifiesto: "
        f"{len(records):,}"
    )
    print(f"Manifiesto: {manifest_path}")
    print("=" * 70)
    print("EXPORTACIÓN COMPLETADA CORRECTAMENTE")
    print("=" * 70)


def main() -> None:
    arguments = parse_arguments()
    settings = get_settings()

    # Garantiza que el directorio del caché exista antes de cargar el dataset.
    settings.hf_cache_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Cargando dataset desde Hugging Face...")
    print(f"Caché de Hugging Face: {settings.hf_cache_path}")

    dataset = load_dataset(
        settings.hf_dataset,
        cache_dir=str(settings.hf_cache_path),
    )

    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            "Se esperaba que load_dataset devolviera "
            "un DatasetDict."
        )

    process_dataset(
        dataset=dataset,
        limit=arguments.limit,
    )


if __name__ == "__main__":
    main()