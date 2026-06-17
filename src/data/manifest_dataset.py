"""Dataset de PyTorch basado en el manifiesto de imágenes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class ManifestImageDataset(Dataset):
    """Carga las imágenes registradas en el manifiesto."""

    def __init__(
        self,
        records: list[dict[str, str]],
        image_root: Path,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.records = records
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        """Devuelve el número de imágenes."""
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        """Carga y preprocesa una imagen."""
        record = self.records[index]

        image_path = (
            self.image_root
            / record["image_relpath"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"No existe la imagen: {image_path}"
            )

        with Image.open(image_path) as source_image:
            rgb_image = source_image.convert("RGB")

            try:
                tensor = self.transform(rgb_image)
            finally:
                rgb_image.close()

        return tensor, index