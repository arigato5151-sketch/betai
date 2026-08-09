from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import MatchPrediction

_OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")


def _ml_component_probabilities(row: MatchPrediction) -> tuple[object, object, object]:
    snapshot = row.probability_components
    components = snapshot.get("components") if isinstance(snapshot, dict) else None
    ml = components.get("ml") if isinstance(components, dict) else None
    if not isinstance(ml, dict):
        return None, None, None
    return ml.get("HOME_WIN"), ml.get("DRAW"), ml.get("AWAY_WIN")


def _row_brier(row: MatchPrediction) -> float | None:
    values = _ml_component_probabilities(row)
    if row.actual_result not in _OUTCOMES or any(value is None for value in values):
        return None
    probabilities: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        probability = float(value)
        if not isfinite(probability) or probability < 0:
            return None
        probabilities.append(probability)
    total = sum(probabilities)
    if total <= 0:
        return None
    normalized = [value / total for value in probabilities]
    target = _OUTCOMES.index(row.actual_result)
    return sum(
        (probability - (1.0 if index == target else 0.0)) ** 2
        for index, probability in enumerate(normalized)
    )


class ModelMonitoringService:
    def __init__(self, db: Session) -> None:
        self.db = db

    @staticmethod
    def _bootstrap_delta_lower_bound(
        recent: list[float], baseline: list[float]
    ) -> float:
        rng = np.random.default_rng(42)
        iterations = settings.MODEL_DRIFT_BOOTSTRAP_SAMPLES
        recent_values = np.asarray(recent, dtype=float)
        baseline_values = np.asarray(baseline, dtype=float)
        recent_indices = rng.integers(
            0, len(recent_values), size=(iterations, len(recent_values))
        )
        baseline_indices = rng.integers(
            0, len(baseline_values), size=(iterations, len(baseline_values))
        )
        deltas = recent_values[recent_indices].mean(axis=1) - baseline_values[
            baseline_indices
        ].mean(axis=1)
        percentile = (1.0 - settings.MODEL_DRIFT_CONFIDENCE) * 100.0
        return float(np.percentile(deltas, percentile))

    def snapshot(self, active_artifact_version: str | None = None) -> dict[str, Any]:
        required = settings.MODEL_DRIFT_MIN_SAMPLES * 2
        common: dict[str, Any] = {
            "drift_detected": False,
            "samples": 0,
            "required_samples": required,
            "window_size": 0,
            "recent_brier": None,
            "baseline_brier": None,
            "brier_delta": None,
            "brier_delta_lower_bound": None,
            "threshold": settings.MODEL_DRIFT_BRIER_THRESHOLD,
            "confidence": settings.MODEL_DRIFT_CONFIDENCE,
            "artifact_version": active_artifact_version,
            "metric_source": "ml_component",
            "ordering": "kickoff_desc",
            "evaluated_at": datetime.now(UTC).isoformat(),
        }
        if not active_artifact_version:
            return {
                **common,
                "status": "model_unavailable",
            }
        rows = (
            self.db.query(MatchPrediction)
            .filter(
                MatchPrediction.actual_result.isnot(None),
                MatchPrediction.training_eligible.is_(True),
                MatchPrediction.result_verification_status == "verified",
                MatchPrediction.model_artifact_version == active_artifact_version,
            )
            .order_by(
                MatchPrediction.kickoff.desc(),
                MatchPrediction.analyzed_at.desc(),
                MatchPrediction.id.desc(),
            )
            .limit(max(settings.MODEL_DRIFT_WINDOW_SIZE, required) * 4)
            .all()
        )
        scores = [score for row in rows if (score := _row_brier(row)) is not None][
            : settings.MODEL_DRIFT_WINDOW_SIZE * 2
        ]
        if len(scores) < required:
            return {
                **common,
                "status": "insufficient_data",
                "samples": len(scores),
            }

        window_size = min(settings.MODEL_DRIFT_WINDOW_SIZE, len(scores) // 2)
        recent = scores[:window_size]
        baseline = scores[window_size : window_size * 2]
        recent_brier = sum(recent) / len(recent)
        baseline_brier = sum(baseline) / len(baseline)
        delta = recent_brier - baseline_brier
        delta_lower_bound = self._bootstrap_delta_lower_bound(recent, baseline)
        drift_detected = (
            delta >= settings.MODEL_DRIFT_BRIER_THRESHOLD
            and delta_lower_bound >= settings.MODEL_DRIFT_BRIER_THRESHOLD
        )
        return {
            **common,
            "status": "drift" if drift_detected else "stable",
            "drift_detected": drift_detected,
            "samples": len(scores),
            "window_size": window_size,
            "recent_brier": round(recent_brier, 6),
            "baseline_brier": round(baseline_brier, 6),
            "brier_delta": round(delta, 6),
            "brier_delta_lower_bound": round(delta_lower_bound, 6),
        }
