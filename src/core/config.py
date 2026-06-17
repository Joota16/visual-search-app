"""Configuración central de Visual Search App."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Variables de configuración cargadas desde el archivo .env."""

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

    image_directory: Path = Path("data/images")
    thumbnail_directory: Path = Path("data/thumbnails")
    manifest_directory: Path = Path("data/manifests")

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve una única instancia reutilizable de la configuración."""
    return Settings()