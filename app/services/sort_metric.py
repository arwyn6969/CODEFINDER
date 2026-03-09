"""Shared score computation for cross-source character-sort comparison."""

from __future__ import annotations

from typing import Mapping


SORT_METRIC_WEIGHTS = {
    "cosine_similarity": 0.30,
    "avg_fingerprint_score": 0.40,
    "dimension_similarity": 0.30,
}

SORT_METRIC_VERSION = "sort_metric_v1_0_30_40_30"


def score_components(
    cosine_similarity: float,
    avg_fingerprint_score: float,
    dimension_similarity: float,
    weights: Mapping[str, float] | None = None,
) -> float:
    metric_weights = weights or SORT_METRIC_WEIGHTS
    return float(
        metric_weights["cosine_similarity"] * cosine_similarity
        + metric_weights["avg_fingerprint_score"] * avg_fingerprint_score
        + metric_weights["dimension_similarity"] * dimension_similarity
    )


def score_metrics(
    metrics: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    return score_components(
        cosine_similarity=float(metrics.get("cosine_similarity", 0.0)),
        avg_fingerprint_score=float(metrics.get("avg_fingerprint_score", 0.0)),
        dimension_similarity=float(metrics.get("dimension_similarity", 0.0)),
        weights=weights,
    )


def metric_descriptor(weights: Mapping[str, float] | None = None) -> str:
    metric_weights = weights or SORT_METRIC_WEIGHTS
    return (
        f"cosine={metric_weights['cosine_similarity']:.2f}, "
        f"avg_fingerprint={metric_weights['avg_fingerprint_score']:.2f}, "
        f"dimension={metric_weights['dimension_similarity']:.2f}"
    )
