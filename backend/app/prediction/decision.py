from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations

from app.core.config import settings


class PredictionDecisionPolicy:
    """Separate a probability forecast from a decision-grade recommendation."""

    OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")

    @classmethod
    def _normalize(cls, value: object) -> tuple[float, float, float] | None:
        if not isinstance(value, Mapping):
            return None
        try:
            probabilities = tuple(float(value[outcome]) for outcome in cls.OUTCOMES)
        except (KeyError, TypeError, ValueError):
            return None
        if any(not math.isfinite(item) or item < 0 for item in probabilities):
            return None
        total = math.fsum(probabilities)
        if total <= 0:
            return None
        return (
            probabilities[0] / total,
            probabilities[1] / total,
            probabilities[2] / total,
        )

    @staticmethod
    def _normalized_entropy(probabilities: tuple[float, float, float]) -> float:
        entropy = -math.fsum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0
        )
        return entropy / math.log(len(probabilities))

    @staticmethod
    def _js_divergence(
        left: tuple[float, float, float], right: tuple[float, float, float]
    ) -> float:
        midpoint = tuple((a + b) / 2.0 for a, b in zip(left, right))

        def divergence(
            source: tuple[float, float, float], reference: tuple[float, ...]
        ) -> float:
            return math.fsum(
                probability * math.log(probability / reference[index])
                for index, probability in enumerate(source)
                if probability > 0 and reference[index] > 0
            )

        return (divergence(left, midpoint) + divergence(right, midpoint)) / (
            2.0 * math.log(2.0)
        )

    @classmethod
    def evaluate(cls, analysis: Mapping[str, object]) -> dict[str, object]:
        probabilities = cls._normalize(analysis.get("all_probabilities"))
        if probabilities is None:
            return {
                "status": "abstain",
                "reasons": ["invalid_probabilities"],
                "confidence_tier": "insufficient",
            }

        ordered = sorted(probabilities, reverse=True)
        top_probability_pct = ordered[0] * 100.0
        margin_pct = (ordered[0] - ordered[1]) * 100.0
        entropy = cls._normalized_entropy(probabilities)

        raw_ensemble = analysis.get("ensemble")
        components = (
            raw_ensemble.get("components")
            if isinstance(raw_ensemble, Mapping)
            else None
        )
        normalized_sources: list[tuple[float, float, float]] = []
        if isinstance(components, Mapping):
            normalized_sources = [
                normalized
                for component in components.values()
                if (normalized := cls._normalize(component)) is not None
            ]
        max_source_jsd = max(
            (
                cls._js_divergence(left, right)
                for left, right in combinations(normalized_sources, 2)
            ),
            default=0.0,
        )

        reasons: list[str] = []
        if top_probability_pct < settings.DECISION_MIN_TOP_PROBABILITY_PCT:
            reasons.append("top_probability_too_low")
        if margin_pct < settings.DECISION_MIN_MARGIN_PCT:
            reasons.append("probability_margin_too_low")
        if entropy > settings.DECISION_MAX_NORMALIZED_ENTROPY:
            reasons.append("predictive_entropy_too_high")
        if (
            max_source_jsd > settings.DECISION_MAX_SOURCE_JSD
            and margin_pct < settings.DECISION_SOURCE_DIVERGENCE_MAX_MARGIN_PCT
        ):
            reasons.append("prediction_sources_diverge")

        status = "abstain" if reasons else "eligible"
        confidence_tier = (
            "low"
            if reasons
            else (
                "high"
                if top_probability_pct >= 60.0 and margin_pct >= 15.0
                else "medium"
            )
        )
        return {
            "status": status,
            "reasons": reasons,
            "confidence_tier": confidence_tier,
            "top_probability_pct": round(top_probability_pct, 2),
            "probability_margin_pct": round(margin_pct, 2),
            "normalized_entropy": round(entropy, 6),
            "max_source_js_divergence": round(max_source_jsd, 6),
            "source_count": len(normalized_sources),
        }
