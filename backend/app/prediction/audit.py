from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from app.core.allowed_leagues import ALLOWED_LEAGUES
from app.core.config import settings
from app.db.models import MatchPrediction

OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class PredictionAuditor:
    @staticmethod
    def wilson_interval(
        successes: int,
        samples: int,
        *,
        z_score: float = 1.959963984540054,
    ) -> dict[str, float] | None:
        if samples <= 0 or successes < 0 or successes > samples:
            return None
        proportion = successes / samples
        denominator = 1.0 + (z_score**2 / samples)
        center = (proportion + z_score**2 / (2.0 * samples)) / denominator
        margin = (
            z_score
            * math.sqrt(
                proportion * (1.0 - proportion) / samples
                + z_score**2 / (4.0 * samples**2)
            )
            / denominator
        )
        return {
            "lower_pct": round(max(0.0, center - margin) * 100.0, 2),
            "upper_pct": round(min(1.0, center + margin) * 100.0, 2),
        }

    @staticmethod
    def bootstrap_mean_interval(
        values: list[float],
        *,
        iterations: int | None = None,
    ) -> dict[str, float] | None:
        if not values:
            return None
        sample_count = len(values)
        iteration_count = iterations or settings.AUDIT_BOOTSTRAP_ITERATIONS
        generator = random.Random(20260803)
        means = sorted(
            sum(generator.choice(values) for _ in range(sample_count)) / sample_count
            for _ in range(iteration_count)
        )
        lower_index = max(0, int(iteration_count * 0.025))
        upper_index = min(iteration_count - 1, int(iteration_count * 0.975))
        return {
            "lower_pct": round(means[lower_index] * 100.0, 2),
            "upper_pct": round(means[upper_index] * 100.0, 2),
        }

    @staticmethod
    def calculate_bet_roi(
        prediction: str | None,
        actual_result: str | None,
        odd: float | None,
    ) -> float:
        """Return unit-stake profit/loss for a valid settled 1X2 bet."""
        if (
            prediction not in OUTCOMES
            or actual_result not in OUTCOMES
            or odd is None
            or not math.isfinite(odd)
            or odd <= 1.0
        ):
            return 0.0
        return round(odd - 1.0, 4) if prediction == actual_result else -1.0

    @staticmethod
    def calculate_clv(placed_odds: float | None, closing_odds: float | None) -> float:
        """Return decimal-odds CLV: placed / closing - 1."""
        if (
            placed_odds is None
            or closing_odds is None
            or not math.isfinite(placed_odds)
            or not math.isfinite(closing_odds)
            or placed_odds <= 1.0
            or closing_odds <= 1.0
        ):
            return 0.0
        return round((placed_odds / closing_odds) - 1.0, 4)

    @staticmethod
    def select_closing_odd(
        market: Mapping[str, object] | None,
        prediction: str | None,
    ) -> float | None:
        """Select the closing price for the predicted outcome, never another market."""
        if prediction not in OUTCOMES or not isinstance(market, Mapping):
            return None
        raw_odds = market.get("raw_odds")
        if not isinstance(raw_odds, Mapping):
            return None
        try:
            closing_odd = float(raw_odds[prediction])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(closing_odd) or closing_odd <= 1.0:
            return None
        return closing_odd

    @staticmethod
    def normalized_probabilities(
        values: Sequence[float | None],
    ) -> tuple[float, float, float] | None:
        """Normalize stored percentages/fractions without inventing legacy forecasts."""
        if len(values) != 3:
            return None
        if any(value is None for value in values):
            return None
        try:
            probabilities = tuple(float(cast(float, value)) for value in values)
        except (TypeError, ValueError):
            return None
        if any(not math.isfinite(value) or value < 0 for value in probabilities):
            return None
        total = sum(probabilities)
        if total <= 0:
            return None
        normalized = tuple(value / total for value in probabilities)
        return normalized[0], normalized[1], normalized[2]

    @classmethod
    def calculate_brier_score(cls, prediction: MatchPrediction) -> float | None:
        if prediction.actual_result not in OUTCOMES:
            return None
        probabilities = cls.normalized_probabilities(
            (prediction.prob_home, prediction.prob_draw, prediction.prob_away)
        )
        if probabilities is None:
            return None
        return sum(
            (forecast - float(prediction.actual_result == outcome)) ** 2
            for outcome, forecast in zip(OUTCOMES, probabilities, strict=True)
        )

    @classmethod
    def closing_snapshot_age_hours(cls, prediction: MatchPrediction) -> float | None:
        """Hours between the closing snapshot and kickoff.

        Positive values mean the snapshot pre-dated kickoff (genuine closing
        evidence); None when provenance is absent (pre-2.2 rows).
        """
        kickoff = prediction.kickoff
        captured = prediction.closing_odds_snapshot_at
        if kickoff is None or captured is None:
            return None
        kickoff_utc = _utc(kickoff)
        captured_utc = _utc(captured)
        return (kickoff_utc - captured_utc).total_seconds() / 3600.0

    @classmethod
    def _is_fresh_closing_evidence(cls, prediction: MatchPrediction) -> bool:
        age_hours = cls.closing_snapshot_age_hours(prediction)
        if age_hours is None:
            return True
        return 0.0 <= age_hours <= settings.AUDIT_MAX_CLOSING_ODDS_AGE_HOURS

    @classmethod
    def audit_predictions(
        cls,
        predictions: list[MatchPrediction],
    ) -> dict[str, Any]:
        """Compute accuracy, unit-stake ROI, multiclass Brier and genuine CLV."""
        resolved = [
            prediction
            for prediction in predictions
            if prediction.actual_result in OUTCOMES
        ]
        valid_bets = [
            prediction
            for prediction in resolved
            if prediction.prediction in OUTCOMES
            and prediction.odd is not None
            and math.isfinite(prediction.odd)
            and prediction.odd > 1.0
        ]
        correct = sum(
            prediction.prediction == prediction.actual_result
            for prediction in resolved
            if prediction.prediction in OUTCOMES
        )
        classified = sum(prediction.prediction in OUTCOMES for prediction in resolved)
        profits = [
            cls.calculate_bet_roi(
                prediction.prediction,
                prediction.actual_result,
                prediction.odd,
            )
            for prediction in valid_bets
        ]
        brier_scores = [
            score
            for prediction in resolved
            if (score := cls.calculate_brier_score(prediction)) is not None
        ]
        clv_values = [
            cls.calculate_clv(prediction.odd, prediction.closing_odds)
            for prediction in valid_bets
            if prediction.closing_odds is not None
            and math.isfinite(prediction.closing_odds)
            and prediction.closing_odds > 1.0
            and cls._is_fresh_closing_evidence(prediction)
        ]
        # Closing is the only honest reference: realized unit-stake profit when
        # every bet is settled at its closing price instead of the stale price
        # that triggered the (often illusionary) edge.
        closing_candidates = [
            prediction
            for prediction in valid_bets
            if prediction.closing_odds is not None
            and math.isfinite(prediction.closing_odds)
            and prediction.closing_odds > 1.0
        ]
        stale_closing = [
            prediction
            for prediction in closing_candidates
            if not cls._is_fresh_closing_evidence(prediction)
        ]
        closing_provenance_missing = [
            prediction
            for prediction in closing_candidates
            if cls.closing_snapshot_age_hours(prediction) is None
        ]
        closing_profits = [
            cls.calculate_bet_roi(
                prediction.prediction,
                prediction.actual_result,
                prediction.closing_odds,
            )
            for prediction in closing_candidates
            if cls._is_fresh_closing_evidence(prediction)
        ]
        total_profit = sum(profits)
        reliable_sample = len(resolved) >= settings.AUDIT_MIN_RELIABLE_SAMPLES
        closing_roi_lower = 0.0
        if closing_profits:
            closing_roi_interval = cls.bootstrap_mean_interval(closing_profits)
            if isinstance(closing_roi_interval, Mapping):
                closing_roi_lower = float(closing_roi_interval.get("lower_pct", 0.0))
        return {
            "total_predictions": len(resolved),
            "classified_predictions": classified,
            "correct_predictions": correct,
            "total_bets": len(valid_bets),
            "win_rate_pct": (
                round((correct / classified) * 100.0, 2) if classified else 0.0
            ),
            "win_rate_confidence_interval_95": cls.wilson_interval(correct, classified),
            "total_profit_units": round(total_profit, 4),
            "total_roi_pct": (
                round((total_profit / len(valid_bets)) * 100.0, 2)
                if valid_bets
                else 0.0
            ),
            "roi_confidence_interval_95_pct": cls.bootstrap_mean_interval(profits),
            "brier_score": (
                round(sum(brier_scores) / len(brier_scores), 4)
                if brier_scores
                else None
            ),
            "brier_samples": len(brier_scores),
            "avg_clv_pct": (
                round(sum(clv_values) / len(clv_values) * 100.0, 2)
                if clv_values
                else None
            ),
            "clv_samples": len(clv_values),
            "closing_bets": len(closing_profits),
            "closing_roi_pct": (
                round(sum(closing_profits) / len(closing_profits) * 100.0, 2)
                if closing_profits
                else 0.0
            ),
            "opening_roi_pct": (
                round((total_profit / len(valid_bets)) * 100.0, 2)
                if valid_bets
                else 0.0
            ),
            "closing_vs_opening_roi_delta_pct": (
                round(
                    (sum(closing_profits) / len(closing_profits) * 100.0)
                    - ((total_profit / len(valid_bets)) * 100.0),
                    2,
                )
                if closing_profits and valid_bets
                else 0.0
            ),
            "closing_roi_confidence_interval_95_pct": (
                round(closing_roi_lower, 2) if closing_profits else None
            ),
            "minimum_reliable_samples": settings.AUDIT_MIN_RELIABLE_SAMPLES,
            "closing_gate_samples": settings.AUDIT_MIN_CLOSING_SAMPLES,
            "closing_gate_min_roi_pct": settings.AUDIT_MIN_CLOSING_ROI_PCT,
            "closing_max_age_hours": settings.AUDIT_MAX_CLOSING_ODDS_AGE_HOURS,
            "stale_closing_odds_bets": len(stale_closing),
            "closing_provenance_missing_bets": len(closing_provenance_missing),
            "sample_status": "reliable" if reliable_sample else "insufficient",
            "decision_grade": reliable_sample,
        }

    @classmethod
    def audit_by_league(
        cls,
        predictions: list[MatchPrediction],
    ) -> dict[str, object]:
        """Return comparable per-league metrics with the same audit semantics."""
        league_names = {
            cast(int, league["id"]): str(league["name"]) for league in ALLOWED_LEAGUES
        }
        grouped: dict[int, list[MatchPrediction]] = {}
        unassigned = 0
        for prediction in predictions:
            if prediction.actual_result not in OUTCOMES:
                continue
            if prediction.league_id is None:
                unassigned += 1
                continue
            grouped.setdefault(prediction.league_id, []).append(prediction)

        leagues = []
        for league_id, rows in grouped.items():
            metrics = cls.audit_predictions(rows)
            leagues.append(
                {
                    "league_id": league_id,
                    "league_name": league_names.get(league_id, f"Lig {league_id}"),
                    **metrics,
                }
            )
        leagues.sort(
            key=lambda row: (-int(row["total_predictions"]), str(row["league_name"]))
        )
        return {
            "overall": cls.audit_predictions(predictions),
            "leagues": leagues,
            "unassigned_predictions": unassigned,
        }


class FinancialRecommendationPolicy:
    """Require statistically positive realized ROI and CLV before activation."""

    @staticmethod
    def evaluate(audit: Mapping[str, object]) -> dict[str, object]:
        minimum = settings.AUDIT_MIN_RELIABLE_SAMPLES
        closing_minimum = settings.AUDIT_MIN_CLOSING_SAMPLES
        closing_min_roi = settings.AUDIT_MIN_CLOSING_ROI_PCT
        reasons: list[str] = []
        total_bets_value = audit.get("total_bets")
        clv_samples_value = audit.get("clv_samples")
        closing_bets_value = audit.get("closing_bets")
        total_bets = (
            int(total_bets_value) if isinstance(total_bets_value, (int, float)) else 0
        )
        clv_samples = (
            int(clv_samples_value) if isinstance(clv_samples_value, (int, float)) else 0
        )
        closing_bets = (
            int(closing_bets_value)
            if isinstance(closing_bets_value, (int, float))
            else 0
        )
        roi_interval = audit.get("roi_confidence_interval_95_pct")
        roi_lower = (
            float(roi_interval.get("lower_pct", 0.0))
            if isinstance(roi_interval, Mapping)
            else 0.0
        )
        avg_clv = audit.get("avg_clv_pct")
        avg_clv_value = float(avg_clv) if isinstance(avg_clv, (int, float)) else 0.0
        closing_roi = audit.get("closing_roi_confidence_interval_95_pct")
        closing_roi_lower = (
            float(closing_roi) if isinstance(closing_roi, (int, float)) else 0.0
        )
        opening_roi = audit.get("opening_roi_pct")
        closing_roi_point = audit.get("closing_roi_pct")
        opening_roi_point = (
            float(opening_roi) if isinstance(opening_roi, (int, float)) else None
        )
        erosion_delta = settings.AUDIT_CLOSING_ROI_EROSION_DELTA_PCT

        if audit.get("decision_grade") is not True or total_bets < minimum:
            reasons.append("insufficient_verified_bets")
        if roi_lower <= 0:
            reasons.append("roi_confidence_interval_not_positive")
        if clv_samples < minimum:
            reasons.append("insufficient_closing_odds_samples")
        if avg_clv_value <= 0:
            reasons.append("average_clv_not_positive")
        if closing_bets < closing_minimum:
            reasons.append("insufficient_closing_odds_bets")
        if closing_roi_lower < closing_min_roi:
            reasons.append("closing_roi_below_threshold")
        if (
            closing_bets > 0
            and opening_roi_point is not None
            and isinstance(closing_roi_point, (int, float))
            and opening_roi_point - float(closing_roi_point) >= erosion_delta
        ):
            reasons.append("closing_roi_erosion")

        return {
            "eligible": not reasons,
            "reasons": reasons,
            "minimum_samples": minimum,
            "total_bets": total_bets,
            "clv_samples": clv_samples,
            "roi_lower_95_pct": roi_lower,
            "avg_clv_pct": avg_clv_value,
            "closing_gate": {
                "closing_bets": closing_bets,
                "closing_roi_lower_95_pct": closing_roi_lower,
                "minimum_samples": closing_minimum,
                "minimum_roi_pct": closing_min_roi,
            },
        }
