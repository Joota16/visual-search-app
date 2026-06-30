"""Ejecucion principal de la evaluacion por configuracion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluation.constants import MODEL_REGISTRY
from evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.query_loader import EvaluationQuery
from evaluation.reporting import (
    build_error_report,
    summarize_bucket,
)
from src.core.config import Settings
from src.services.embedding_service import EmbeddingService
from src.services.qdrant_service import QdrantService
from src.services.search_service import SearchService


def resolve_model_keys(
    raw_models: str,
) -> list[str]:
    """Convierte --models en una lista valida de claves de modelo.

    Recibe el string separado por comas que llega desde la CLI,
    normaliza espacios y mayusculas/minusculas, y verifica que cada
    modelo exista en `MODEL_REGISTRY`.
    """
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
    """Construye la configuracion efectiva para un modelo textual.

    Para `current` reutiliza la configuracion actual.
    Para los otros modelos crea una copia de `settings` cambiando
    el encoder textual, sin tocar la coleccion visual ya indexada.
    """
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


def evaluate_configuration(
    settings: Settings,
    queries: list[EvaluationQuery],
    top_k: int,
    model_key: str,
) -> dict[str, Any]:
    """Evalua una configuracion textual concreta de punta a punta.

    Inicializa servicios, ejecuta cada query contra el indice ya
    existente, calcula metricas de recuperacion y agrupa resultados
    globales y por segmento (idioma, categoria y tipo de query).

    Devuelve un diccionario serializable con el reporte completo del
    modelo evaluado.
    """
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
