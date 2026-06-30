"""Evalua recuperacion texto-imagen sobre embeddings ya indexados.

Esta evaluacion no entrena el modelo.
Solo ejecuta busquedas sobre embeddings ya indexados en Qdrant.
Aumentar la cantidad de queries incrementa el numero de busquedas,
no el entrenamiento.
La evaluacion valida recuperacion texto-imagen, no clasificacion supervisada.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev
from typing import Any

import numpy as np

from src.core.config import Settings, get_settings
from src.services.embedding_service import EmbeddingService
from src.services.qdrant_service import QdrantService
from src.services.search_service import SearchService


DEFAULT_TOP_K = 10
DEFAULT_OUTPUT_FILENAME = "evaluation_report.json"
DEFAULT_QUERIES_PATH = Path(
    "evaluation/queries/sample.json"
)
DEFAULT_EXPANDED_QUERIES_PATH = Path(
    "evaluation/queries/expanded.json"
)
MODEL_REGISTRY = {
    "current": {
        "name": "current",
        "text_model": None,
        "description": (
            "Usa la configuracion textual actual del proyecto."
        ),
    },
    # Ambos modelos producen embeddings de 512 dimensiones
    # alineados con CLIP ViT-B-32, por lo que esta evaluacion
    # puede reutilizar la misma coleccion de embeddings visuales.
    "base": {
        "name": "base",
        "text_model": "sentence-transformers/clip-ViT-B-32",
        "description": (
            "Encoder textual CLIP base alineado con CLIP ViT-B-32."
        ),
    },
    "multilingual": {
        "name": "multilingual",
        "text_model": (
            "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        ),
        "description": (
            "Encoder textual CLIP multilingue alineado con CLIP ViT-B-32."
        ),
    },
}
METRICS = (
    "precision_at_1",
    "precision_at_5",
    "hit_at_1",
    "hit_at_5",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
    "embedding_ms",
    "search_ms",
    "total_ms",
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
    """Lee argumentos enviados por terminal."""
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
        help="No genera gráficos PNG al finalizar.",
    )

    parser.add_argument(
        "--plots-dir",
        default=None,
        help=(
            "Directorio de salida para gráficos. "
            "Si se omite, usa evaluation/plots o un sibling plots."
        ),
    )

    return parser.parse_args()


def load_queries(
    queries_path: Path,
    query_limit: int | None = None,
) -> list[EvaluationQuery]:
    """Carga queries desde un archivo JSON."""
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
    """Extrae labels disponibles en el manifiesto."""
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
    """Verifica que las relevancias declaradas existan en el manifiesto."""
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


def resolve_model_keys(
    raw_models: str,
) -> list[str]:
    """Normaliza y valida modelos solicitados."""
    model_keys = [
        model.strip().casefold()
        for model in raw_models.split(",")
        if model.strip()
    ]

    if not model_keys:
        raise RuntimeError(
            "Debes indicar al menos un modelo en --models."
        )

    invalid_models = [
        model
        for model in model_keys
        if model not in MODEL_REGISTRY
    ]

    if invalid_models:
        raise RuntimeError(
            "Modelos no reconocidos: "
            + ", ".join(invalid_models)
        )

    return model_keys


def build_model_settings(
    settings: Settings,
    model_key: str,
) -> Settings:
    """Crea una configuracion con el encoder textual deseado."""
    text_model = MODEL_REGISTRY[model_key]["text_model"]
    if text_model is None:
        return settings

    if not hasattr(settings, "text_model"):
        raise RuntimeError(
            "Esta rama no soporta --models base,multilingual. "
            "Usa --models current o una rama con soporte "
            "sentence-transformers."
        )

    return settings.model_copy(
        update={
            "text_model": text_model,
            "openclip_pretrained": text_model,
        }
    )


def precision_at_k(
    labels: list[str],
    relevant_labels: set[str],
    k: int,
) -> float:
    """Calcula precision@k."""
    if k <= 0:
        return 0.0

    inspected = labels[:k]
    if not inspected:
        return 0.0

    relevant_count = sum(
        label in relevant_labels for label in inspected
    )

    return relevant_count / len(inspected)


def hit_at_k(
    labels: list[str],
    relevant_labels: set[str],
    k: int,
) -> float:
    """Calcula hit@k."""
    if k <= 0:
        return 0.0

    return float(
        any(
            label in relevant_labels
            for label in labels[:k]
        )
    )


def recall_at_k(
    labels: list[str],
    relevant_labels: set[str],
    k: int,
) -> float:
    """Calcula recall@k por cobertura de labels relevantes."""
    if not relevant_labels or k <= 0:
        return 0.0

    retrieved_relevant = {
        label
        for label in labels[:k]
        if label in relevant_labels
    }

    return len(retrieved_relevant) / len(relevant_labels)


def reciprocal_rank(
    labels: list[str],
    relevant_labels: set[str],
) -> float:
    """Calcula reciprocal rank del primer acierto."""
    for index, label in enumerate(labels, start=1):
        if label in relevant_labels:
            return 1.0 / index

    return 0.0


def dcg_at_k(
    labels: list[str],
    graded_labels: dict[str, int],
    k: int,
) -> float:
    """Calcula discounted cumulative gain."""
    score = 0.0
    seen_labels: set[str] = set()

    for index, label in enumerate(labels[:k], start=1):
        if label in seen_labels:
            continue
        seen_labels.add(label)

        relevance = graded_labels.get(label, 0)
        if relevance <= 0:
            continue

        score += (
            (2**relevance - 1)
            / math.log2(index + 1)
        )

    return score


def ndcg_at_k(
    labels: list[str],
    graded_labels: dict[str, int],
    k: int,
) -> float:
    """Calcula normalized DCG."""
    actual_dcg = dcg_at_k(
        labels=labels,
        graded_labels=graded_labels,
        k=k,
    )

    ideal_labels = sorted(
        graded_labels,
        key=lambda label: graded_labels[label],
        reverse=True,
    )
    ideal_dcg = dcg_at_k(
        labels=ideal_labels,
        graded_labels=graded_labels,
        k=k,
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def summarize_values(
    values: list[float],
) -> dict[str, float]:
    """Resume una lista con promedio y dispersion."""
    if not values:
        return {
            "avg": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "avg": sum(values) / len(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_bucket(
    query_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Agrega metricas por grupo."""
    summary = {
        "query_count": len(query_metrics),
    }

    for metric in METRICS:
        summary[metric] = summarize_values(
            [float(item[metric]) for item in query_metrics]
        )

    return summary


def sort_query_reports(
    query_reports: list[dict[str, Any]],
    metric: str,
    reverse: bool,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Ordena y recorta queries segun una metrica."""
    sorted_reports = sorted(
        query_reports,
        key=lambda item: float(item[metric]),
        reverse=reverse,
    )
    return sorted_reports[:limit]


def build_error_report(
    query_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construye un reporte de errores y casos extremos."""
    no_hit_top_10 = [
        report
        for report in query_reports
        if float(report["hit_at_10"]) == 0.0
    ]

    incorrect_top_1 = [
        report
        for report in query_reports
        if float(report["hit_at_1"]) == 0.0
    ]

    return {
        "queries_without_hit_top_10": no_hit_top_10,
        "queries_with_incorrect_top_1": incorrect_top_1,
        "best_queries_by_ndcg_at_10": sort_query_reports(
            query_reports,
            metric="ndcg_at_10",
            reverse=True,
        ),
        "worst_queries_by_ndcg_at_10": sort_query_reports(
            query_reports,
            metric="ndcg_at_10",
            reverse=False,
        ),
    }


def write_json_atomically(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """Guarda el reporte en JSON usando un temporal."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            data,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(output_path)


def write_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Escribe un CSV de manera simple."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def export_csv_reports(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Exporta CSV global, por query y comparativo."""
    stem_directory = output_path.parent
    stem_name = output_path.stem

    global_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for model_key, model_report in report["models"].items():
        overall = model_report["overall"]
        global_row = {
            "model": model_key,
            "query_count": overall["query_count"],
        }
        for metric in METRICS:
            metric_summary = overall[metric]
            global_row[f"{metric}_avg"] = metric_summary["avg"]
            global_row[f"{metric}_std"] = metric_summary["std"]
            global_row[f"{metric}_min"] = metric_summary["min"]
            global_row[f"{metric}_max"] = metric_summary["max"]
        global_rows.append(global_row)

        for query_report in model_report["queries"]:
            row = {
                "model": model_key,
                "query_id": query_report["query_id"],
                "query": query_report["query"],
                "language": query_report["language"],
                "category": query_report["category"],
                "query_type": query_report["query_type"],
            }
            for metric in METRICS:
                row[metric] = query_report[metric]
            per_query_rows.append(row)

    comparison = report.get("comparison", {})
    for metric, metric_report in comparison.get(
        "metrics", {}
    ).items():
        row = {"metric": metric}
        row.update(metric_report["by_model"])
        row["winner"] = metric_report["winner"]
        comparison_rows.append(row)

    write_csv(
        output_path=stem_directory / f"{stem_name}_global.csv",
        fieldnames=list(global_rows[0].keys()) if global_rows else ["model"],
        rows=global_rows,
    )
    write_csv(
        output_path=stem_directory / f"{stem_name}_per_query.csv",
        fieldnames=list(per_query_rows[0].keys()) if per_query_rows else ["model"],
        rows=per_query_rows,
    )
    if comparison_rows:
        write_csv(
            output_path=stem_directory / f"{stem_name}_comparison.csv",
            fieldnames=list(comparison_rows[0].keys()),
            rows=comparison_rows,
        )


def save_failure_reports(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Guarda JSONs separados con fallos por modelo."""
    stem_directory = output_path.parent
    stem_name = output_path.stem

    for model_key, model_report in report["models"].items():
        failure_path = (
            stem_directory
            / f"{stem_name}_{model_key}_failures.json"
        )
        write_json_atomically(
            data=model_report["error_report"],
            output_path=failure_path,
        )


def build_comparison_report(
    model_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Construye comparacion lado a lado entre modelos."""
    comparison_metrics: dict[str, Any] = {}
    maximize_metrics = {
        "precision_at_1",
        "precision_at_5",
        "hit_at_1",
        "hit_at_5",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
    }

    for metric in METRICS:
        by_model = {
            model_key: float(
                model_report["overall"][metric]["avg"]
            )
            for model_key, model_report in model_reports.items()
        }
        if metric in maximize_metrics:
            winner = max(by_model, key=by_model.get)
        else:
            winner = min(by_model, key=by_model.get)

        comparison_metrics[metric] = {
            "by_model": by_model,
            "winner": winner,
        }

    return {
        "metrics": comparison_metrics,
    }


def evaluate_configuration(
    settings: Settings,
    queries: list[EvaluationQuery],
    top_k: int,
    model_key: str,
) -> dict[str, Any]:
    """Evalua una configuracion textual concreta."""
    model_settings = build_model_settings(
        settings=settings,
        model_key=model_key,
    )

    print("\n" + "=" * 80)
    print(f"CONFIGURACION: {model_key}")
    print("=" * 80)
    text_model_name = getattr(
        model_settings,
        "text_model",
        getattr(model_settings, "openclip_pretrained", "unknown"),
    )
    image_model_name = getattr(
        model_settings,
        "image_model",
        getattr(model_settings, "openclip_model", "unknown"),
    )
    print(
        "Text model: "
        f"{text_model_name}"
    )
    print(
        "Coleccion visual reutilizada: "
        f"{model_settings.qdrant_collection}"
    )

    embedding_service = EmbeddingService(
        settings=model_settings
    )
    qdrant_service = QdrantService(
        settings=model_settings
    )
    qdrant_service.check_connection()

    search_service = SearchService(
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
    )

    _ = embedding_service.encode_texts(
        ["warmup query"]
    )

    query_reports: list[dict[str, Any]] = []
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_query_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for evaluation_query in queries:
        response = search_service.search_text(
            evaluation_query.query,
            limit=top_k,
        )

        labels = [
            result["label"].casefold()
            for result in response["results"]
        ]
        relevant_labels = set(
            evaluation_query.relevant_labels
        )

        query_report = {
            "query_id": evaluation_query.query_id,
            "query": evaluation_query.query,
            "language": evaluation_query.language,
            "category": evaluation_query.category,
            "query_type": evaluation_query.query_type,
            "relevant_labels": list(
                evaluation_query.relevant_labels
            ),
            "precision_at_1": precision_at_k(
                labels,
                relevant_labels,
                1,
            ),
            "precision_at_5": precision_at_k(
                labels,
                relevant_labels,
                5,
            ),
            "hit_at_1": hit_at_k(
                labels,
                relevant_labels,
                1,
            ),
            "hit_at_5": hit_at_k(
                labels,
                relevant_labels,
                5,
            ),
            "hit_at_10": hit_at_k(
                labels,
                relevant_labels,
                10,
            ),
            "recall_at_1": recall_at_k(
                labels,
                relevant_labels,
                1,
            ),
            "recall_at_5": recall_at_k(
                labels,
                relevant_labels,
                5,
            ),
            "recall_at_10": recall_at_k(
                labels,
                relevant_labels,
                10,
            ),
            "mrr": reciprocal_rank(
                labels,
                relevant_labels,
            ),
            "ndcg_at_10": ndcg_at_k(
                labels,
                evaluation_query.graded_labels,
                10,
            ),
            "embedding_ms": response["embedding_ms"],
            "search_ms": response["search_ms"],
            "total_ms": response["total_ms"],
            "optimized_query": response.get(
                "optimized_query"
            ),
            "query_variants": response.get(
                "query_variants",
                [],
            ),
            "query_aspects": response.get(
                "query_aspects",
                [],
            ),
            "top_results": [
                {
                    "position": result["position"],
                    "label": result["label"],
                    "score": result["score"],
                    "is_relevant": (
                        result["label"].casefold()
                        in relevant_labels
                    ),
                }
                for result in response["results"]
            ],
        }

        query_reports.append(query_report)
        by_language[evaluation_query.language].append(
            query_report
        )
        by_category[evaluation_query.category].append(
            query_report
        )
        by_query_type[evaluation_query.query_type].append(
            query_report
        )

    model_report = {
        "model_key": model_key,
        "description": MODEL_REGISTRY[model_key]["description"],
        "text_model": text_model_name,
        "image_model": image_model_name,
        "collection": model_settings.qdrant_collection,
        "overall": summarize_bucket(query_reports),
        "by_language": {
            key: summarize_bucket(value)
            for key, value in sorted(by_language.items())
        },
        "by_category": {
            key: summarize_bucket(value)
            for key, value in sorted(by_category.items())
        },
        "by_query_type": {
            key: summarize_bucket(value)
            for key, value in sorted(by_query_type.items())
        },
        "error_report": build_error_report(query_reports),
        "queries": query_reports,
    }

    qdrant_service.close()

    overall = model_report["overall"]
    print(f"Precision@5: {overall['precision_at_5']['avg']:.4f}")
    print(f"Recall@10: {overall['recall_at_10']['avg']:.4f}")
    print(f"MRR: {overall['mrr']['avg']:.4f}")
    print(f"nDCG@10: {overall['ndcg_at_10']['avg']:.4f}")
    print(
        "Latencia media total: "
        f"{overall['total_ms']['avg']:.2f} ms"
    )

    return model_report


def print_comparison(
    comparison: dict[str, Any],
) -> None:
    """Imprime comparacion resumida en consola."""
    print("\n" + "=" * 80)
    print("COMPARACION")
    print("=" * 80)

    for metric in (
        "precision_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "total_ms",
    ):
        metric_report = comparison["metrics"][metric]
        by_model = metric_report["by_model"]
        winner = metric_report["winner"]
        formatted_values = ", ".join(
            f"{model}={value:.4f}"
            if "ms" not in metric
            else f"{model}={value:.2f}"
            for model, value in by_model.items()
        )
        print(f"{metric}: {formatted_values}")
        print(f"Mejor {metric}: {winner}")


def main() -> None:
    """Ejecuta la evaluacion de queries."""
    arguments = parse_arguments()
    settings = get_settings()
    model_keys = resolve_model_keys(arguments.models)

    queries_path = Path(arguments.queries_file)
    if not queries_path.is_absolute():
        queries_path = (
            Path.cwd() / queries_path
        ).resolve()

    queries = load_queries(
        queries_path=queries_path,
        query_limit=arguments.query_limit,
    )

    output_path = (
        Path(arguments.output)
        if arguments.output
        else (
            Path("evaluation/reports")
            / DEFAULT_OUTPUT_FILENAME
        )
    )
    if not output_path.is_absolute():
        output_path = (
            Path.cwd() / output_path
        ).resolve()

    manifest_path = getattr(
        settings,
        "manifest_file_path",
        settings.manifest_path / "caltech256_manifest.csv",
    )
    available_labels = list_manifest_labels(
        manifest_path=manifest_path,
    )
    validate_queries_against_labels(
        queries=queries,
        available_labels={
            label.casefold()
            for label in available_labels
        },
    )

    print("=" * 80)
    print("EVALUACION DE BUSQUEDA TEXTO-IMAGEN")
    print("=" * 80)
    print(f"Queries: {len(queries)}")
    print(f"Top K: {arguments.top_k}")
    print(f"Archivo queries: {queries_path}")
    print(f"Reporte: {output_path}")
    print(
        f"Labels disponibles detectados: {len(available_labels)}"
    )
    print(f"Modelos: {', '.join(model_keys)}")
    print("=" * 80)

    model_reports = {
        model_key: evaluate_configuration(
            settings=settings,
            queries=queries,
            top_k=arguments.top_k,
            model_key=model_key,
        )
        for model_key in model_keys
    }

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "top_k": arguments.top_k,
        "query_count": len(queries),
        "queries_file": str(queries_path),
        "available_labels_count": len(available_labels),
        "models": model_reports,
        "comparison": (
            build_comparison_report(model_reports)
            if arguments.compare or len(model_reports) > 1
            else {}
        ),
    }

    write_json_atomically(
        data=report,
        output_path=output_path,
    )

    if arguments.export_csv:
        export_csv_reports(
            report=report,
            output_path=output_path,
        )

    if arguments.save_failures:
        save_failure_reports(
            report=report,
            output_path=output_path,
        )

    created_plots: list[Path] = []
    if not arguments.skip_plots:
        try:
            from evaluation.plot_results import generate_plots
        except ModuleNotFoundError as error:
            if error.name == "matplotlib":
                raise RuntimeError(
                    "Falta matplotlib. Instala dependencias o "
                    "usa --skip-plots."
                ) from error
            raise

        plots_dir = (
            Path(arguments.plots_dir)
            if arguments.plots_dir
            else None
        )
        if plots_dir is not None and not plots_dir.is_absolute():
            plots_dir = (
                Path.cwd() / plots_dir
            ).resolve()
        created_plots = generate_plots(
            report_path=output_path,
            output_dir=plots_dir,
        )

    if report["comparison"]:
        print_comparison(report["comparison"])

    print("\nRESULTADO")
    print("-" * 80)
    for model_key, model_report in model_reports.items():
        overall = model_report["overall"]
        print(f"CONFIGURACION: {model_key}")
        print(
            f"Precision@5: {overall['precision_at_5']['avg']:.4f}"
        )
        print(
            f"Recall@10: {overall['recall_at_10']['avg']:.4f}"
        )
        print(f"MRR: {overall['mrr']['avg']:.4f}")
        print(
            f"nDCG@10: {overall['ndcg_at_10']['avg']:.4f}"
        )
        print(
            "Latencia media total: "
            f"{overall['total_ms']['avg']:.2f} ms"
        )
        print()

    print(f"Reporte: {output_path}")
    if created_plots:
        print("Graficos:")
        for plot_path in created_plots:
            print(f"- {plot_path}")


if __name__ == "__main__":
    main()
