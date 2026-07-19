import math
from collections.abc import Mapping
from typing import Any

from app.core.config import settings
from app.prediction.stats_engine import StatsEngine


class ProbabilityEnsembler:
    VERSION = "weighted_probability_ensemble_v1"
    OUTCOME_KEYS = ("HOME_WIN", "DRAW", "AWAY_WIN")

    @classmethod
    def _normalize(
        cls, probabilities: Mapping[str, Any] | None
    ) -> dict[str, float] | None:
        if not isinstance(probabilities, Mapping):
            return None
        try:
            values = {
                outcome: float(probabilities[outcome]) for outcome in cls.OUTCOME_KEYS
            }
        except (KeyError, TypeError, ValueError):
            return None
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            return None
        total = sum(values.values())
        if total <= 0:
            return None
        return {outcome: value / total for outcome, value in values.items()}

    @classmethod
    def _rounded_percentages(
        cls, probabilities: Mapping[str, float]
    ) -> dict[str, float]:
        rounded = {
            outcome: round(probabilities[outcome] * 100.0, 2)
            for outcome in cls.OUTCOME_KEYS
        }
        residual = round(100.0 - sum(rounded.values()), 2)
        if residual:
            leading_outcome = max(
                cls.OUTCOME_KEYS, key=lambda outcome: probabilities[outcome]
            )
            rounded[leading_outcome] = round(rounded[leading_outcome] + residual, 2)
        return rounded

    @classmethod
    def apply(
        cls,
        stats_analysis: dict[str, Any],
        ml_result: Mapping[str, Any] | None = None,
        market: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        stats_probabilities = cls._normalize(stats_analysis.get("all_probabilities"))
        if stats_probabilities is None:
            raise ValueError("Stats analysis must contain valid 1X2 probabilities")

        sources: dict[str, dict[str, float]] = {"stats": stats_probabilities}
        configured_weights = {
            "stats": settings.ENSEMBLE_STATS_WEIGHT,
            "ml": settings.ENSEMBLE_ML_WEIGHT,
            "market": settings.ENSEMBLE_MARKET_WEIGHT,
        }

        if ml_result and ml_result.get("ready"):
            ml_probabilities = cls._normalize(ml_result.get("all_probabilities"))
            if ml_probabilities is not None and configured_weights["ml"] > 0:
                sources["ml"] = ml_probabilities

        market_probabilities = cls._normalize(
            market.get("fair_probability") if market else None
        )
        if market_probabilities is not None and configured_weights["market"] > 0:
            sources["market"] = market_probabilities

        active_weight_total = sum(configured_weights[source] for source in sources)
        if active_weight_total <= 0:
            raise ValueError("At least one ensemble source must have a positive weight")
        effective_weights = {
            source: configured_weights[source] / active_weight_total
            for source in sources
        }
        component_snapshot = {
            source: cls._rounded_percentages(probabilities)
            for source, probabilities in sources.items()
        }

        if len(sources) == 1:
            return {
                **stats_analysis,
                "ensemble": {
                    "version": cls.VERSION,
                    "applied": False,
                    "weights": {"stats": 1.0},
                    "components": component_snapshot,
                },
            }

        blended = {
            outcome: sum(
                sources[source][outcome] * effective_weights[source]
                for source in sources
            )
            for outcome in cls.OUTCOME_KEYS
        }
        final_probabilities = cls._rounded_percentages(blended)
        sorted_outcomes = sorted(
            final_probabilities.items(), key=lambda item: item[1], reverse=True
        )
        prediction, probability = sorted_outcomes[0]
        confidence_gap = round(probability - sorted_outcomes[1][1], 2)

        return {
            **stats_analysis,
            "base_model": stats_analysis.get("model"),
            "model": cls.VERSION,
            "prediction": prediction,
            "probability": probability,
            "all_probabilities": final_probabilities,
            "confidence_gap": confidence_gap,
            "confidence_tier": StatsEngine._confidence_tier(confidence_gap),
            "ensemble": {
                "version": cls.VERSION,
                "applied": True,
                "weights": {
                    source: round(weight, 6)
                    for source, weight in effective_weights.items()
                },
                "components": component_snapshot,
            },
        }
