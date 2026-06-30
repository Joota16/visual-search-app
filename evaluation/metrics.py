"""Metricas de evaluacion de recuperacion."""

from __future__ import annotations

import math


def precision_at_k(
    labels: list[str],
    relevant_labels: set[str],
    k: int,
) -> float:
    """Calcula precision@k sobre los primeros `k` resultados.

    Mide que proporcion de los elementos inspeccionados pertenece al
    conjunto de labels relevantes.
    """
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
    """Calcula hit@k.

    Devuelve 1.0 si aparece al menos un resultado relevante dentro de
    los primeros `k`; en caso contrario devuelve 0.0.
    """
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
    """Calcula recall@k por cobertura de labels relevantes.

    Mide cuantas clases relevantes distintas fueron recuperadas dentro
    de los primeros `k`, respecto del total esperado.
    """
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
    """Calcula reciprocal rank del primer acierto relevante.

    Premia que el primer resultado correcto aparezca lo mas arriba
    posible en el ranking.
    """
    for index, label in enumerate(labels, start=1):
        if label in relevant_labels:
            return 1.0 / index
    return 0.0


def dcg_at_k(
    labels: list[str],
    graded_labels: dict[str, int],
    k: int,
) -> float:
    """Calcula DCG@k usando relevancias graduadas por label.

    Aplica descuento logaritmico por posicion y evita contar una misma
    clase mas de una vez dentro del ranking inspeccionado.
    """
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
    """Calcula nDCG@k normalizando el DCG contra el ranking ideal.

    Produce un score entre 0 y 1 para comparar calidad de ranking
    incluso cuando hay distintas relevancias por clase.
    """
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
