"""Servicio de embeddings usando SentenceTransformers CLIP multilingüe."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision.transforms import functional as TF


class EmbeddingService:
    """Genera embeddings de texto e imagen en el mismo espacio vectorial."""

    def __init__(self, settings):
        self.settings = settings
        self.device = settings.device

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE=cuda, pero PyTorch no detecta CUDA."
            )

        self.image_model_name = getattr(
            settings,
            "image_model",
            "clip-ViT-B-32",
        )

        self.text_model_name = getattr(
            settings,
            "text_model",
            "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        )

        print("=" * 70)
        print("CARGANDO CLIP MULTILINGÜE CON SENTENCE-TRANSFORMERS")
        print("=" * 70)
        print(f"Modelo imagen: {self.image_model_name}")
        print(f"Modelo texto: {self.text_model_name}")
        print(f"Device: {self.device}")

        self.image_model = SentenceTransformer(
            self.image_model_name,
            device=self.device,
        )

        self.text_model = SentenceTransformer(
            self.text_model_name,
            device=self.device,
        )

        self.embedding_dim = 512

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """ Compatibilidad con ManifestImageDataset.
        Convierte todas las imágenes a RGB y las redimensiona a 224x224
        para que el DataLoader pueda formar batches sin error.
        """
        image = image.convert("RGB")
        image = image.resize((224, 224))

        return TF.to_tensor(image)

    def encode_image_tensors(
        self,
        image_batch: torch.Tensor,
    ) -> np.ndarray:
        """
        Genera embeddings desde un batch de tensores.

        Este método mantiene compatibilidad con generate_embeddings.py.
        """
        images = [
            TF.to_pil_image(image.cpu()).convert("RGB")
            for image in image_batch
        ]

        embeddings = self.image_model.encode(
            images,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)

    def encode_image(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        """Genera embedding para una sola imagen PIL."""
        image = image.convert("RGB")

        embedding = self.image_model.encode(
            image,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.astype(np.float32)

    def encode_images(
        self,
        images: Iterable[Image.Image],
    ) -> np.ndarray:
        """Genera embeddings para varias imágenes PIL."""
        rgb_images = [
            image.convert("RGB")
            for image in images
        ]

        embeddings = self.image_model.encode(
            rgb_images,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)

    def encode_text(
        self,
        text: str,
    ) -> np.ndarray:
        """Genera embedding para una consulta textual."""
        embedding = self.text_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.astype(np.float32)

    def encode_texts(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Genera embeddings para varias consultas textuales."""
        embeddings = self.text_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)

    def get_embedding_dim(self) -> int:
        """Devuelve la dimensión del embedding."""
        return self.embedding_dim