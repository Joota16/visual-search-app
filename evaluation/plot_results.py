"""Genera gráficos a partir de un reporte de evaluación."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PRIMARY_METRICS = (
    "precision_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
)
LATENCY_METRICS = (
    "embedding_ms",
    "search_ms",
    "total_ms",
)


def load_pyplot() -> Any:
    """Carga matplotlib en modo no interactivo."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as pyplot

    return pyplot


def parse_arguments() -> argparse.Namespace:
    """Lee argumentos enviados por terminal."""
    parser = argparse.ArgumentParser(
        description=(
            "Genera gráficos PNG a partir de un reporte JSON "
            "de evaluación."
        )
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Ruta al reporte JSON generado por evaluation.evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directorio de salida para los PNG. "
            "Si se omite, usa evaluation/plots o un sibling plots."
        ),
    )
    return parser.parse_args()


def resolve_plots_dir(
    report_path: Path,
    output_dir: str | None = None,
) -> Path:
    """Resuelve dónde guardar los gráficos."""
    if output_dir:
        path = Path(output_dir)
        return (
            path if path.is_absolute() else (Path.cwd() / path).resolve()
        )

    if report_path.parent.name == "reports":
        return report_path.parent.parent / "plots"

    return report_path.parent / "plots"


def load_report(
    report_path: Path,
) -> dict[str, Any]:
    """Carga un reporte JSON."""
    with report_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(json_file)


def save_grouped_bar_chart(
    output_path: Path,
    title: str,
    labels: list[str],
    series: dict[str, list[float]],
    ylabel: str,
) -> None:
    """Guarda un gráfico de barras agrupadas."""
    plt = load_pyplot()
    figure, axis = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
    )

    model_names = list(series.keys())
    bar_width = 0.8 / max(len(model_names), 1)
    positions = list(range(len(labels)))

    for offset, model_name in enumerate(model_names):
        axis.bar(
            [
                position + (offset - (len(model_names) - 1) / 2) * bar_width
                for position in positions
            ],
            series[model_name],
            width=bar_width,
            label=model_name,
        )

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, labels)
    axis.legend()
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_breakdown_chart(
    output_path: Path,
    title: str,
    breakdown: dict[str, Any],
    metric: str,
    ylabel: str,
) -> None:
    """Guarda gráfico por breakdown para todos los modelos."""
    if not breakdown:
        return

    groups = sorted(
        {
            group_name
            for model_breakdown in breakdown.values()
            for group_name in model_breakdown.keys()
        }
    )
    if not groups:
        return

    series = {
        model_name: [
            model_breakdown.get(group_name, {})
            .get(metric, {})
            .get("avg", 0.0)
            for group_name in groups
        ]
        for model_name, model_breakdown in breakdown.items()
    }

    save_grouped_bar_chart(
        output_path=output_path,
        title=title,
        labels=groups,
        series=series,
        ylabel=ylabel,
    )


def generate_plots(
    report_path: Path,
    output_dir: Path | None = None,
) -> list[Path]:
    """Genera gráficos principales a partir de un reporte."""
    report = load_report(report_path)
    plots_dir = output_dir or resolve_plots_dir(report_path)
    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = report["models"]
    metric_series = {
        model_key: [
            model_report["overall"][metric]["avg"]
            for metric in PRIMARY_METRICS
        ]
        for model_key, model_report in models.items()
    }
    latency_series = {
        model_key: [
            model_report["overall"][metric]["avg"]
            for metric in LATENCY_METRICS
        ]
        for model_key, model_report in models.items()
    }

    created_paths = [
        plots_dir / "overall_quality.png",
        plots_dir / "overall_latency.png",
        plots_dir / "by_language_ndcg_at_10.png",
        plots_dir / "by_query_type_ndcg_at_10.png",
    ]

    save_grouped_bar_chart(
        output_path=created_paths[0],
        title="Comparación general de calidad",
        labels=[
            "Precision@5",
            "Recall@10",
            "MRR",
            "nDCG@10",
        ],
        series=metric_series,
        ylabel="Score",
    )
    save_grouped_bar_chart(
        output_path=created_paths[1],
        title="Comparación general de latencia",
        labels=[
            "Embedding ms",
            "Search ms",
            "Total ms",
        ],
        series=latency_series,
        ylabel="Milisegundos",
    )
    save_breakdown_chart(
        output_path=created_paths[2],
        title="nDCG@10 por idioma",
        breakdown={
            model_key: model_report.get("by_language", {})
            for model_key, model_report in models.items()
        },
        metric="ndcg_at_10",
        ylabel="nDCG@10",
    )
    save_breakdown_chart(
        output_path=created_paths[3],
        title="nDCG@10 por tipo de query",
        breakdown={
            model_key: model_report.get("by_query_type", {})
            for model_key, model_report in models.items()
        },
        metric="ndcg_at_10",
        ylabel="nDCG@10",
    )

    return created_paths


def main() -> None:
    """Punto de entrada CLI."""
    arguments = parse_arguments()
    report_path = Path(arguments.report)
    if not report_path.is_absolute():
        report_path = (Path.cwd() / report_path).resolve()

    output_dir = (
        resolve_plots_dir(report_path, arguments.output_dir)
        if arguments.output_dir is not None
        else resolve_plots_dir(report_path)
    )
    created_paths = generate_plots(
        report_path=report_path,
        output_dir=output_dir,
    )
    print("Graficos generados:")
    for path in created_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
