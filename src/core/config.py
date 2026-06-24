"""Configuración central de Visual Search App."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Variables de configuración cargadas desde el archivo .env."""
    
    frontend_api_url: str = "http://127.0.0.1:8000"
    frontend_api_timeout_seconds: float = 60.0
    
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    default_top_k: int = 8
    max_top_k: int = 50
    max_upload_mb: int = 10
    
    
    batch_size: int = 64

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Visual Search API"
    app_env: str = "development"

    hf_dataset: str = "ilee0022/Caltech-256"
    hf_cache_directory: Path = Path("data/cache/huggingface")
    
    hf_home: Path = Path("C:/hf-cache")
    hf_hub_cache: Path = Path("C:/hf-cache/hub")

    openclip_model: str = "ViT-B-32"
    openclip_pretrained: str = "laion2b_s34b_b79k"

    device: str = "cuda"
    model_precision: str = "float16"

    image_directory: Path = Path("data/images")
    thumbnail_directory: Path = Path("data/thumbnails")
    manifest_directory: Path = Path("data/manifests")
    
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "caltech256_multilingual_v1"
    qdrant_timeout_seconds: float = 120.0

    upsert_batch_size: int = 256

    embedding_filename: str = ("caltech256_multilingual_clip_vit_b32.npy")

    image_model: str = "clip-ViT-B-32"
    text_model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    embedding_directory: Path = Path("data/embeddings")
    num_workers: int = 4
    prefetch_factor: int = 2

    def resolve_path(self, path: Path) -> Path:
        """Convierte una ruta relativa en una ruta absoluta del proyecto."""
        if path.is_absolute():
            return path

        return PROJECT_ROOT / path

    @property
    def hf_cache_path(self) -> Path:
        """Ruta absoluta del caché de Hugging Face."""
        return self.resolve_path(self.hf_cache_directory)

    @property
    def image_path(self) -> Path:
        """Ruta absoluta de las imágenes exportadas."""
        return self.resolve_path(self.image_directory)

    @property
    def thumbnail_path(self) -> Path:
        """Ruta absoluta de las miniaturas."""
        return self.resolve_path(self.thumbnail_directory)

    @property
    def manifest_path(self) -> Path:
        """Ruta absoluta de los manifiestos."""
        return self.resolve_path(self.manifest_directory)
    
    @property
    def hf_home_path(self) -> Path:
        """Ruta principal del caché de Hugging Face."""
        return self.resolve_path(self.hf_home)

    @property
    def hf_hub_cache_path(self) -> Path:
        """Ruta del caché de modelos de Hugging Face."""
        return self.resolve_path(self.hf_hub_cache)
    
    @property
    def embedding_path(self) -> Path:
        """Ruta absoluta donde se almacenarán los embeddings."""
        return self.resolve_path(self.embedding_directory)
    
    @property
    def embedding_file_path(self) -> Path:
        """Ruta completa del archivo de embeddings."""
        return self.embedding_path / self.embedding_filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve una única instancia reutilizable de la configuración."""
    return Settings()