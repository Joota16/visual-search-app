"""Utilidades de resumen y exportacion de reportes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import pstdev
from typing import Any

from evaluation.constants import METRICS


def summarize_values(
    values: list[float],
) -> dict[str, float]:
    """Resume una lista numerica con estadisticos basicos.

    Devuelve promedio, desviacion estandar poblacional, minimo y maximo.
    Si la lista esta vacia, devuelve ceros para mantener el formato.
    """
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
    """Agrega metricas de varias queries dentro de un mismo grupo.

    Se usa para el resumen global y para desgloses por idioma,
    categoria y tipo de query.
    """
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
    """Ordena queries por una metrica y devuelve solo las primeras.

    Permite construir listados de mejores o peores casos segun la
    metrica solicitada.
    """
    sorted_reports = sorted(
        query_reports,
        key=lambda item: float(item[metric]),
        reverse=reverse,
    )
    return sorted_reports[:limit]


def build_error_report(
    query_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construye un resumen de fallos y casos extremos.

    Incluye queries sin aciertos en top-10, queries con top-1
    incorrecto y rankings de mejores/peores resultados por nDCG@10.
    """
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
    """Escribe JSON de forma atomica usando un archivo temporal.

    Evita dejar archivos finales incompletos si la escritura se
    interrumpe a mitad del proceso.
    """
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
    """Escribe un CSV tabular con cabecera y filas provistas."""
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
    """Exporta el reporte en CSVs derivados.

    Genera tres vistas: resumen global por modelo, detalle por query
    y comparacion entre modelos si existe.
    """
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
    """Guarda un JSON de fallos por cada modelo evaluado.

    Esto facilita inspeccionar errores sin abrir el reporte completo.
    """
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
    """Construye una comparacion lado a lado entre modelos.

    Para metricas de calidad gana el valor mas alto; para latencia
    gana el valor mas bajo.
    """
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


def print_comparison(
    comparison: dict[str, Any],
) -> None:
    """Imprime en consola una comparacion resumida entre modelos.

    Muestra las metricas principales y el ganador por cada una.
    """
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
