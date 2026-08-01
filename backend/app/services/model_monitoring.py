from __future__ import annotations

from math import isfinite
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import MatchPrediction

_OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")


def _row_brier(row: MatchPrediction) -> float | None:
    values = (row.prob_home, row.prob_draw, row.prob_away)
    if row.actual_result not in _OUTCOMES or any(value is None for value in values):
        return None
    probabilities = [float(value) for value in values if value is not None]
    if not all(isfinite(value) and value >= 0 for value in probabilities):
        return None
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

    def snapshot(self) -> dict[str, Any]:
        rows = (
            self.db.query(MatchPrediction)
            .filter(MatchPrediction.actual_result.isnot(None))
            .order_by(MatchPrediction.id.desc())
            .limit(settings.MODEL_DRIFT_WINDOW_SIZE * 2)
            .all()
        )
        scores = [score for row in rows if (score := _row_brier(row)) is not None]
        required = settings.MODEL_DRIFT_MIN_SAMPLES * 2
        if len(scores) < required:
            return {
                "status": "insufficient_data",
                "drift_detected": False,
                "samples": len(scores),
                "required_samples": required,
                "recent_brier": None,
                "baseline_brier": None,
                "brier_delta": None,
            }

        window_size = min(settings.MODEL_DRIFT_WINDOW_SIZE, len(scores) // 2)
        recent = scores[:window_size]
        baseline = scores[window_size : window_size * 2]
        recent_brier = sum(recent) / len(recent)
        baseline_brier = sum(baseline) / len(baseline)
        delta = recent_brier - baseline_brier
        drift_detected = delta >= settings.MODEL_DRIFT_BRIER_THRESHOLD
        return {
            "status": "drift" if drift_detected else "stable",
            "drift_detected": drift_detected,
            "samples": len(scores),
            "required_samples": required,
            "window_size": window_size,
            "recent_brier": round(recent_brier, 6),
            "baseline_brier": round(baseline_brier, 6),
            "brier_delta": round(delta, 6),
            "threshold": settings.MODEL_DRIFT_BRIER_THRESHOLD,
        }
