"""Cliente HTTP para comunicarse con Visual Search API."""

from __future__ import annotations

from typing import Any

import httpx


class VisualSearchAPIError(RuntimeError):
    """Error controlado al comunicarse con la API."""


class VisualSearchAPIClient:
    """Cliente síncrono para Visual Search API."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")

        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
        )

    def _raise_api_error(
        self,
        response: httpx.Response,
    ) -> None:
        """Convierte una respuesta HTTP inválida en un error legible."""
        try:
            payload = response.json()
            detail = payload.get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        raise VisualSearchAPIError(
            f"API respondió HTTP {response.status_code}: "
            f"{detail}"
        )

    def _parse_response(
        self,
        response: httpx.Response,
    ) -> dict[str, Any]:
        """Valida y convierte una respuesta JSON."""
        if response.is_error:
            self._raise_api_error(response)

        try:
            return response.json()
        except ValueError as error:
            raise VisualSearchAPIError(
                "La API devolvió una respuesta que no es JSON."
            ) from error

    def health(self) -> dict[str, Any]:
        """Comprueba si el proceso de FastAPI responde."""
        try:
            response = self.client.get("/health")
        except httpx.RequestError as error:
            raise VisualSearchAPIError(
                "No se pudo conectar con FastAPI. "
                "Comprueba que Uvicorn esté ejecutándose."
            ) from error

        return self._parse_response(response)

    def ready(self) -> dict[str, Any]:
        """Comprueba modelo, Qdrant y colección."""
        try:
            response = self.client.get("/ready")
        except httpx.RequestError as error:
            raise VisualSearchAPIError(
                "No se pudo verificar el estado de la API."
            ) from error

        return self._parse_response(response)

    def stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del sistema."""
        try:
            response = self.client.get(
                "/api/v1/stats"
            )
        except httpx.RequestError as error:
            raise VisualSearchAPIError(
                "No se pudieron recuperar las estadísticas."
            ) from error

        return self._parse_response(response)

    def search_text(
        self,
        query: str,
        limit: int,
    ) -> dict[str, Any]:
        """Realiza una búsqueda textual."""
        try:
            response = self.client.post(
                "/api/v1/search/text",
                json={
                    "query": query,
                    "limit": limit,
                },
            )
        except httpx.RequestError as error:
            raise VisualSearchAPIError(
                "No se pudo realizar la búsqueda textual."
            ) from error

        return self._parse_response(response)

    def search_image(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        limit: int,
    ) -> dict[str, Any]:
        """Realiza una búsqueda por imagen."""
        try:
            response = self.client.post(
                "/api/v1/search/image",
                files={
                    "file": (
                        filename,
                        file_bytes,
                        content_type,
                    )
                },
                data={
                    "limit": str(limit),
                },
            )
        except httpx.RequestError as error:
            raise VisualSearchAPIError(
                "No se pudo realizar la búsqueda por imagen."
            ) from error

        return self._parse_response(response)

    def close(self) -> None:
        """Cierra las conexiones HTTP."""
        self.client.close()