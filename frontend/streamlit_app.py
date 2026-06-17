"""Interfaz web del sistema multimodal de búsqueda."""

from __future__ import annotations

from typing import Any

import streamlit as st
from PIL import Image

from frontend.api_client import (
    VisualSearchAPIClient,
    VisualSearchAPIError,
)
from src.core.config import get_settings


PAGE_TITLE = "Visual Search"
DEFAULT_QUERY = "a photo of grapes"
RESULT_COLUMNS = 4


def configure_page() -> None:
    """Configura la página antes de mostrar componentes."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_resource
def get_api_client() -> VisualSearchAPIClient:
    """Crea un único cliente HTTP reutilizable."""
    settings = get_settings()

    return VisualSearchAPIClient(
        base_url=settings.frontend_api_url,
        timeout_seconds=(
            settings.frontend_api_timeout_seconds
        ),
    )


@st.cache_data(ttl=30)
def get_system_stats(
    api_url: str,
) -> dict[str, Any]:
    """Recupera estadísticas con una caché breve."""
    settings = get_settings()

    client = VisualSearchAPIClient(
        base_url=api_url,
        timeout_seconds=(
            settings.frontend_api_timeout_seconds
        ),
    )

    try:
        return client.stats()
    finally:
        client.close()


def initialize_state() -> None:
    """Inicializa los resultados persistentes entre reruns."""
    defaults = {
        "text_results": None,
        "image_results": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar(
    client: VisualSearchAPIClient,
) -> None:
    """Muestra el estado y características del sistema."""
    with st.sidebar:
        st.header("Estado del sistema")

        try:
            ready = client.ready()

            st.success("API y buscador disponibles")

            st.metric(
                "Imágenes indexadas",
                f"{ready['points_count']:,}",
            )

            st.metric(
                "Vectores HNSW",
                f"{ready['indexed_vectors_count']:,}",
            )

            st.write(
                f"**Modelo:** {ready['model']}"
            )

            st.write(
                f"**Dispositivo:** {ready['device']}"
            )

            st.write(
                f"**Colección:** "
                f"`{ready['collection']}`"
            )

        except VisualSearchAPIError as error:
            st.error(str(error))

        st.divider()

        st.caption(
            "Caltech-256 · OpenCLIP · "
            "Qdrant · FastAPI · Streamlit"
        )


def format_label(label: str) -> str:
    """Convierte una etiqueta técnica en texto legible."""
    return (
        label
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()
    )


def render_metrics(
    response: dict[str, Any],
) -> None:
    """Muestra las métricas de la consulta."""
    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Resultados",
        len(response["results"]),
    )

    metric_columns[1].metric(
        "Embedding",
        f"{response['embedding_ms']:.2f} ms",
    )

    metric_columns[2].metric(
        "Búsqueda vectorial",
        f"{response['search_ms']:.2f} ms",
    )

    metric_columns[3].metric(
        "Tiempo total",
        f"{response['total_ms']:.2f} ms",
    )


def render_result_card(
    result: dict[str, Any],
) -> None:
    """Muestra una imagen recuperada y sus metadatos."""
    st.image(
        result["thumbnail_url"],
        width="stretch",
    )

    st.markdown(
        f"**{result['position']}. "
        f"{format_label(result['label'])}**"
    )

    st.caption(
        f"Similitud: {result['score']:.4f} · "
        f"Split: {result['split']}"
    )

    st.link_button(
        "Ver imagen original",
        result["image_url"],
        width="stretch",
    )


def render_results(
    response: dict[str, Any] | None,
) -> None:
    """Muestra una cuadrícula de resultados."""
    if response is None:
        st.info(
            "Realiza una búsqueda para visualizar resultados."
        )
        return

    results = response.get("results", [])

    if not results:
        st.warning("No se encontraron imágenes.")
        return

    render_metrics(response)

    st.subheader("Resultados")

    columns = st.columns(
        RESULT_COLUMNS,
        gap="medium",
    )

    for index, result in enumerate(results):
        column = columns[index % RESULT_COLUMNS]

        with column:
            with st.container(border=True):
                render_result_card(result)


def render_text_search(
    client: VisualSearchAPIClient,
) -> None:
    """Interfaz para buscar imágenes mediante texto."""
    st.subheader("Buscar mediante texto")

    st.write(
        "Describe el contenido visual que deseas encontrar."
    )

    with st.form(
        key="text_search_form",
        clear_on_submit=False,
    ):
        query = st.text_input(
            "Consulta",
            value=DEFAULT_QUERY,
            placeholder=(
                "Ejemplo: a red motorcycle"
            ),
        )

        limit = st.slider(
            "Cantidad de resultados",
            min_value=4,
            max_value=24,
            value=8,
            step=4,
            key="text_limit",
        )

        submitted = st.form_submit_button(
            "Buscar imágenes",
            type="primary",
            width="stretch",
        )

    if submitted:
        if not query.strip():
            st.warning(
                "Escribe una descripción antes de buscar."
            )
        else:
            try:
                with st.spinner(
                    "Generando embedding y consultando Qdrant..."
                ):
                    response = client.search_text(
                        query=query.strip(),
                        limit=limit,
                    )

                st.session_state.text_results = (
                    response
                )

            except VisualSearchAPIError as error:
                st.error(str(error))

    render_results(
        st.session_state.text_results
    )


def render_image_search(
    client: VisualSearchAPIClient,
) -> None:
    """Interfaz para buscar mediante una imagen."""
    st.subheader("Buscar mediante imagen")

    st.write(
        "Carga una imagen para recuperar elementos "
        "visualmente similares."
    )

    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        max_upload_size=10,
        key="query_image",
    )

    limit = st.slider(
        "Cantidad de resultados",
        min_value=4,
        max_value=24,
        value=8,
        step=4,
        key="image_limit",
    )

    if uploaded_file is not None:
        preview_column, action_column = st.columns(
            [1, 2],
            vertical_alignment="center",
        )

        with preview_column:
            preview_image = Image.open(
                uploaded_file
            )

            st.image(
                preview_image,
                caption="Imagen de consulta",
                width="stretch",
            )

        with action_column:
            st.write(
                f"**Archivo:** {uploaded_file.name}"
            )

            st.write(
                f"**Tipo:** {uploaded_file.type}"
            )

            search_clicked = st.button(
                "Buscar imágenes similares",
                type="primary",
                width="stretch",
            )

        if search_clicked:
            try:
                file_bytes = uploaded_file.getvalue()

                with st.spinner(
                    "Procesando imagen y consultando Qdrant..."
                ):
                    response = client.search_image(
                        file_bytes=file_bytes,
                        filename=uploaded_file.name,
                        content_type=(
                            uploaded_file.type
                            or "application/octet-stream"
                        ),
                        limit=limit,
                    )

                st.session_state.image_results = (
                    response
                )

            except VisualSearchAPIError as error:
                st.error(str(error))

            except Exception as error:
                st.error(
                    "No se pudo procesar la imagen: "
                    f"{error}"
                )

    render_results(
        st.session_state.image_results
    )


def main() -> None:
    """Ejecuta la interfaz Streamlit."""
    configure_page()
    initialize_state()

    st.title("🔎 Visual Search")

    st.write(
        "Sistema multimodal para buscar imágenes "
        "mediante texto o una imagen de referencia."
    )

    client = get_api_client()

    try:
        client.health()
    except VisualSearchAPIError as error:
        st.error(str(error))
        st.warning(
            "Inicia FastAPI antes de utilizar la interfaz."
        )
        st.stop()

    render_sidebar(client)

    text_tab, image_tab = st.tabs(
        [
            "💬 Texto → imagen",
            "🖼️ Imagen → imagen",
        ]
    )

    with text_tab:
        render_text_search(client)

    with image_tab:
        render_image_search(client)


if __name__ == "__main__":
    main()