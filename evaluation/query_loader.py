"""Carga y validacion de queries de evaluacion."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from evaluation.constants import (
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_QUERIES_PATH,
    DEFAULT_TOP_K,
)


@dataclass(frozen=True)
class EvaluationQuery:
    """Representa una query evaluable."""

    query_id: str
    query: str
    language: str
    category: str
    query_type: str
    relevant_labels: tuple[str, ...]
    graded_labels: dict[str, int]


def parse_arguments() -> argparse.Namespace:
    """Define y parsea los argumentos CLI de la evaluacion.

    Centraliza todas las opciones de entrada para que `evaluate.py`
    solo se encargue de orquestar el flujo principal.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evalua busqueda texto-imagen sobre un conjunto "
            "curado de queries."
        )
    )
    parser.add_argument(
        "--queries-file",
        default=str(DEFAULT_QUERIES_PATH),
        help=(
            "Archivo JSON con las queries de evaluacion. "
            "Por defecto usa evaluation/queries/sample.json."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Cantidad de resultados a evaluar por query."
        ),
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help=(
            "Limita la cantidad de queries ejecutadas. "
            "Util para pruebas rapidas."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Ruta del reporte JSON de salida. "
            "Si se omite, se guarda en evaluation/reports/evaluation_report.json."
        ),
    )
    parser.add_argument(
        "--models",
        default="current",
        help=(
            "Configuraciones textuales a evaluar, separadas por coma. "
            "Ejemplos: current o base,multilingual."
        ),
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Exporta CSV resumen, por query y comparativo.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Fuerza la generacion de una comparacion final "
            "entre configuraciones."
        ),
    )
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help=(
            "Guarda archivos JSON separados con queries problematicas."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="No genera graficos PNG al finalizar.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help=(
            "Directorio de salida para graficos. "
            "Si se omite, usa evaluation/plots o un sibling plots."
        ),
    )
    return parser.parse_args()


def load_queries(
    queries_path: Path,
    query_limit: int | None = None,
) -> list[EvaluationQuery]:
    """Carga queries desde JSON y las normaliza al dataclass interno.

    Valida estructura, campos obligatorios y relevancias declaradas.
    Si `query_limit` viene informado, recorta la lista final para
    facilitar pruebas rapidas.
    """
    if not queries_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de queries: {queries_path}"
        )

    with queries_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        payload = json.load(json_file)

    if not isinstance(payload, list) or not payload:
        raise RuntimeError(
            "El archivo de queries debe ser una lista no vacia."
        )

    loaded_queries: list[EvaluationQuery] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"La query #{index} no es un objeto JSON."
            )

        query_id = str(item.get("query_id") or f"q{index:03d}")
        query = str(item.get("query") or "").strip()
        language = str(
            item.get("language") or "unknown"
        ).strip()
        category = str(
            item.get("category") or "general"
        ).strip()
        query_type = str(
            item.get("query_type") or "general"
        ).strip()
        relevant_labels = tuple(
            str(label).strip().casefold()
            for label in item.get("relevant_labels", [])
            if str(label).strip()
        )
        graded_labels = {
            str(label).strip().casefold(): int(score)
            for label, score in (
                item.get("graded_labels", {}) or {}
            ).items()
            if str(label).strip()
        }

        if not query:
            raise RuntimeError(
                f"La query #{index} esta vacia."
            )
        if not relevant_labels and not graded_labels:
            raise RuntimeError(
                f"La query {query_id} no define relevancias."
            )

        if not graded_labels:
            graded_labels = {
                label: 1 for label in relevant_labels
            }

        loaded_queries.append(
            EvaluationQuery(
                query_id=query_id,
                query=query,
                language=language,
                category=category,
                query_type=query_type,
                relevant_labels=relevant_labels
                or tuple(graded_labels.keys()),
                graded_labels=graded_labels,
            )
        )

    if query_limit is not None:
        return loaded_queries[:query_limit]
    return loaded_queries


def list_manifest_labels(
    manifest_path: Path,
) -> list[str]:
    """Lee el manifiesto CSV y devuelve los labels disponibles.

    Se usa para verificar que las relevancias declaradas en las
    queries apunten a clases realmente presentes en el dataset.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No existe el manifiesto: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        labels = {
            row["label"].strip()
            for row in csv.DictReader(csv_file)
            if row.get("label", "").strip()
        }

    return sorted(labels)


def validate_queries_against_labels(
    queries: list[EvaluationQuery],
    available_labels: set[str],
) -> None:
    """Valida que cada label referenciado exista en el manifiesto.

    Si alguna query apunta a labels inexistentes, detiene la ejecucion
    con un error detallado para evitar evaluar contra ground truth
    inconsistente.
    """
    invalid_references: list[str] = []

    for evaluation_query in queries:
        referenced_labels = set(
            evaluation_query.relevant_labels
        ).union(evaluation_query.graded_labels.keys())
        missing_labels = sorted(
            label
            for label in referenced_labels
            if label not in available_labels
        )
        if missing_labels:
            invalid_references.append(
                f"{evaluation_query.query_id}: {missing_labels}"
            )

    if invalid_references:
        raise RuntimeError(
            "Hay labels de evaluacion que no existen en el manifiesto:\n"
            + "\n".join(invalid_references)
        )


def resolve_output_path(raw_output: str | None) -> Path:
    """Convierte la salida pedida por CLI en una ruta absoluta final.

    Si el usuario no especifica `--output`, usa la ruta por defecto
    dentro de `evaluation/reports`.
    """
    output_path = (
        Path(raw_output)
        if raw_output
        else Path("evaluation/reports") / DEFAULT_OUTPUT_FILENAME
    )
    if output_path.is_absolute():
        return output_path
    return (Path.cwd() / output_path).resolve()


def resolve_queries_path(raw_queries_path: str) -> Path:
    """Convierte la ruta del archivo de queries en una ruta absoluta.

    Permite que la CLI acepte tanto rutas absolutas como relativas al
    directorio actual desde donde se ejecuta el modulo.
    """
    queries_path = Path(raw_queries_path)
    if queries_path.is_absolute():
        return queries_path
    return (Path.cwd() / queries_path).resolve()
