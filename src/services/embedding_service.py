"""Servicio para generar embeddings multimodales con OpenCLIP."""

from __future__ import annotations

from threading import Lock
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.core.config import Settings, get_settings


class EmbeddingService:
    """Carga OpenCLIP y genera embeddings de imágenes y textos."""

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._inference_lock = Lock()
        # Deben configurarse antes de importar OpenCLIP,
        # porque OpenCLIP utiliza Hugging Face Hub.
        self.settings.hf_home_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.settings.hf_hub_cache_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.environ["HF_HOME"] = str(
            self.settings.hf_home_path
        )

        os.environ["HF_HUB_CACHE"] = str(
            self.settings.hf_hub_cache_path
        )

        # Importación diferida para garantizar que el caché
        # esté configurado antes de cargar Hugging Face Hub.
        import open_clip

        self.open_clip = open_clip

        if (
            self.settings.device.lower() == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "DEVICE=cuda, pero PyTorch no detecta CUDA."
            )

        self.device = torch.device(
            self.settings.device.lower()
        )

        print(
            "Cargando OpenCLIP "
            f"{self.settings.openclip_model} "
            f"({self.settings.openclip_pretrained})..."
        )

        model, _, preprocess = (
            open_clip.create_model_and_transforms(
                self.settings.openclip_model,
                pretrained=self.settings.openclip_pretrained,
            )
        )

        self.model: Any = model.to(self.device)
        self.model.eval()

        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(
            self.settings.openclip_model
        )

    def _use_autocast(self) -> bool:
        """Indica si debe emplearse precisión mixta."""
        return (
            self.device.type == "cuda"
            and self.settings.model_precision.lower()
            == "float16"
        )
    def encode_image_tensors(
        self,
        batch: torch.Tensor,
    ) -> np.ndarray:
        """Genera embeddings a partir de un lote preprocesado."""

        if batch.ndim != 4:
            raise ValueError(
                "El lote debe tener forma "
                "(batch, channels, height, width)."
            )

        if batch.shape[0] == 0:
            raise ValueError(
                "El lote de imágenes está vacío."
            )

        with self._inference_lock:
            batch = batch.to(
                self.device,
                non_blocking=True,
            )

            with torch.inference_mode():
                if self._use_autocast():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        features = self.model.encode_image(
                            batch,
                            normalize=True,
                        )
                else:
                    features = self.model.encode_image(
                        batch,
                        normalize=True,
                    )

            embeddings = (
                features
                .float()
                .cpu()
                .numpy()
                .astype("float32")
            )

        return np.ascontiguousarray(embeddings)
    
    def encode_images(
        self,
        images: Sequence[Image.Image],
    ) -> np.ndarray:
        """Genera embeddings normalizados para imágenes."""

        if not images:
            raise ValueError(
                "Debe proporcionarse al menos una imagen."
            )

        image_tensors = [
            self.preprocess(
                image.convert("RGB")
            )
            for image in images
        ]

        batch = torch.stack(image_tensors)

        return self.encode_image_tensors(batch)

    def encode_texts(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Genera embeddings normalizados para textos."""
        clean_texts = [
            text.strip()
            for text in texts
            if text.strip()
        ]

        if not clean_texts:
            raise ValueError(
                "Debe proporcionarse al menos un texto válido."
            )

        tokens = self.tokenizer(
            clean_texts
        ).to(
            self.device,
            non_blocking=True,
        )

        with torch.inference_mode():
            if self._use_autocast():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    features = self.model.encode_text(
                        tokens,
                        normalize=True,
                    )
            else:
                features = self.model.encode_text(
                    tokens,
                    normalize=True,
                )

        embeddings = (
            features
            .float()
            .cpu()
            .numpy()
            .astype("float32")
        )

        return np.ascontiguousarray(embeddings)