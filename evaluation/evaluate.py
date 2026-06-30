"""Evalua recuperacion texto-imagen sobre embeddings ya indexados.

Esta evaluacion no entrena el modelo.
Solo ejecuta busquedas sobre embeddings ya indexados en Qdrant.
Aumentar la cantidad de queries incrementa el numero de busquedas,
no el entrenamiento.
La evaluacion valida recuperacion texto-imagen, no clasificacion supervisada.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from evaluation.evaluator import (
    evaluate_configuration,
    resolve_model_keys,
)
from evaluation.query_loader import (
    list_manifest_labels,
    load_queries,
    parse_arguments,
    resolve_output_path,
    resolve_queries_path,
    validate_queries_against_labels,
)
from evaluation.reporting import (
    build_comparison_report,
    export_csv_reports,
    print_comparison,
    save_failure_reports,
    write_json_atomically,
)
from src.core.config import get_settings


def main() -> None:
    """Orquesta la evaluacion completa desde la CLI.

    1. Lee argumentos de terminal.
    2. Carga configuracion global del proyecto.
    3. Resuelve y valida queries y labels disponibles.
    4. Ejecuta la evaluacion por cada modelo solicitado.
    5. Construye el reporte final JSON.
    6. Exporta artefactos opcionales: CSV, fallos y plots.

    Coordina llamadas a modulos especializados.
    """
    arguments = parse_arguments()
    settings = get_settings()
    model_keys = resolve_model_keys(arguments.models)

    queries_path = resolve_queries_path(arguments.queries_file)
    queries = load_queries(
        queries_path=queries_path,
        query_limit=arguments.query_limit,
    )

    output_path = resolve_output_path(arguments.output)
    manifest_path = (
        settings.manifest_path
        / "caltech256_manifest.csv"
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
