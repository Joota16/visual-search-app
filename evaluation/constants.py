"""Constantes compartidas para evaluacion."""

from __future__ import annotations

from pathlib import Path

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
