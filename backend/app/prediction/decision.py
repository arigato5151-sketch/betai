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
    def _effective_thresholds(
        cls,
        *,
        data_quality_score: float | None = None,
        market_confirmed: bool = False,
    ) -> tuple[float, float]:
        """Return (top_prob, margin) thresholds for the conditional tier.

        The conditional tier sits between abstain and eligible.  Contextual
        signals (market confirmation, high data quality) relax the conditional
        thresholds further, letting more borderline predictions reach
        "conditional" instead of "abstain".
        """
        cond_top = settings.DECISION_CONDITIONAL_TOP_PROBABILITY_PCT
        cond_margin = settings.DECISION_CONDITIONAL_MARGIN_PCT

        if (
            market_confirmed
            and data_quality_score is not None
            and data_quality_score >= 70
        ):
            return (max(25.0, cond_top * 0.85), max(1.0, cond_margin * 0.6))
        if market_confirmed:
            return (max(27.0, cond_top * 0.90), max(1.0, cond_margin * 0.7))
        if data_quality_score is not None and data_quality_score >= 70:
            return (max(28.0, cond_top * 0.90), max(1.2, cond_margin * 0.8))
        return cond_top, cond_margin

    @classmethod
    def evaluate(
        cls,
        analysis: Mapping[str, object],
        *,
        market_edge_pct: float | None = None,
        market_implied_pct: float | None = None,
        market_min_edge_pct: float | None = None,
        require_market: bool = False,
        market_confirmed: bool = False,
        data_quality_score: float | None = None,
    ) -> dict[str, object]:
        """Separate a probability forecast from a decision-grade recommendation.

        ``market_edge_pct`` ties the recommendation to an actual price. When
        ``require_market`` is set the forecast may only count as a decision once
        a live market clears the requested edge; otherwise it is demoted to
        research-only. Callers that only need a forecast grade skip the market
        requirement and keep the pure probability decision.

        ``market_confirmed`` and ``data_quality_score`` enable contextual
        threshold relaxation: when the market agrees with the model or data
        quality is high, borderline predictions reach "conditional" instead of
        "abstain".
        """
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

        cond_top, cond_margin = cls._effective_thresholds(
            data_quality_score=data_quality_score,
            market_confirmed=market_confirmed,
        )

        reasons: list[str] = []
        if top_probability_pct < cond_top:
            reasons.append("top_probability_too_low")
        if margin_pct < cond_margin:
            reasons.append("probability_margin_too_low")
        if (
            max_source_jsd > settings.DECISION_MAX_SOURCE_JSD
            and margin_pct < settings.DECISION_SOURCE_DIVERGENCE_MAX_MARGIN_PCT
        ):
            reasons.append("prediction_sources_diverge")

        market_validation = cls._validate_market(
            market_edge_pct=market_edge_pct,
            market_implied_pct=market_implied_pct,
            market_min_edge_pct=market_min_edge_pct,
            top_probability_pct=top_probability_pct,
        )

        base_top = settings.DECISION_MIN_TOP_PROBABILITY_PCT
        base_margin = settings.DECISION_MIN_MARGIN_PCT

        eligible_prob = top_probability_pct >= base_top and margin_pct >= base_margin
        in_conditional_band = (
            top_probability_pct >= cond_top
            and margin_pct >= cond_margin
            and not eligible_prob
        )

        if reasons:
            status = "abstain"
        elif in_conditional_band:
            status = "conditional"
        elif (
            require_market
            and market_validation["present"]
            and market_validation["passed"]
        ):
            status = "eligible"
        elif require_market:
            reasons.append(
                "market_unavailable"
                if not market_validation["present"]
                else "market_edge_insufficient"
            )
            status = "research"
        else:
            status = "eligible"

        high_entropy = entropy > settings.DECISION_MAX_NORMALIZED_ENTROPY
        if reasons:
            confidence_tier = "low"
        elif status == "conditional" or high_entropy:
            confidence_tier = "conditional"
        elif top_probability_pct >= 60.0 and margin_pct >= 15.0:
            confidence_tier = "high"
        else:
            confidence_tier = "medium"

        return {
            "status": status,
            "reasons": reasons,
            "confidence_tier": confidence_tier,
            "top_probability_pct": round(top_probability_pct, 2),
            "probability_margin_pct": round(margin_pct, 2),
            "normalized_entropy": round(entropy, 6),
            "max_source_js_divergence": round(max_source_jsd, 6),
            "source_count": len(normalized_sources),
            "market_validation": market_validation,
        }

    @classmethod
    def _validate_market(
        cls,
        *,
        market_edge_pct: float | None,
        market_implied_pct: float | None,
        market_min_edge_pct: float | None,
        top_probability_pct: float,
    ) -> dict[str, object]:
        if market_edge_pct is None:
            return {
                "present": False,
                "edge_pct": None,
                "implied_pct": market_implied_pct,
                "min_edge_pct": None,
                "passed": None,
            }
        minimum_edge = (
            market_min_edge_pct
            if market_min_edge_pct is not None
            else settings.DECISION_MIN_MARKET_EDGE_PCT
        )
        passed = market_edge_pct >= minimum_edge and (
            market_implied_pct is None
            or top_probability_pct
            > market_implied_pct + settings.DECISION_EDGE_MARGIN_PCT
        )
        return {
            "present": True,
            "edge_pct": round(float(market_edge_pct), 2),
            "implied_pct": market_implied_pct,
            "min_edge_pct": minimum_edge,
            "cleared_vs_implied": (
                top_probability_pct
                > (float(market_implied_pct) + settings.DECISION_EDGE_MARGIN_PCT)
                if market_implied_pct is not None
                else True
            ),
            "passed": bool(passed),
        }
