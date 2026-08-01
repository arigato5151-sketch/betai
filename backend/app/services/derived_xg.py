from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from app.core.config import settings
from app.db.models import HistoricalFixture

MODEL_VERSION = "shot_xg_hgbr_v1"


@dataclass(frozen=True, slots=True)
class DerivedXGResult:
    status: str
    training_matches: int
    holdout_mae: float | None
    baseline_mae: float | None
    holdout_r2: float | None
    updates: tuple[dict[str, object], ...]


class DerivedXGService:
    """Estimate auditable xG only where observed provider xG is unavailable."""

    def __init__(
        self,
        *,
        min_training_matches: int | None = None,
        max_holdout_mae: float | None = None,
        min_baseline_improvement: float | None = None,
        confidence: float | None = None,
    ) -> None:
        self.min_training_matches = (
            min_training_matches or settings.DERIVED_XG_MIN_TRAINING_MATCHES
        )
        self.max_holdout_mae = max_holdout_mae or settings.DERIVED_XG_MAX_HOLDOUT_MAE
        self.min_baseline_improvement = (
            min_baseline_improvement
            if min_baseline_improvement is not None
            else settings.DERIVED_XG_MIN_BASELINE_IMPROVEMENT
        )
        self.confidence = confidence or settings.DERIVED_XG_CONFIDENCE

    def build_updates(self, fixtures: Iterable[HistoricalFixture]) -> DerivedXGResult:
        ordered = sorted(fixtures, key=lambda row: (row.kickoff, row.fixture_id))
        training = [
            row
            for row in ordered
            if row.xg_source == "understat"
            and self._valid_xg(row.home_xg)
            and self._valid_xg(row.away_xg)
            and self._match_features(row) is not None
        ]
        if len(training) < self.min_training_matches:
            return DerivedXGResult(
                status="insufficient_training_data",
                training_matches=len(training),
                holdout_mae=None,
                baseline_mae=None,
                holdout_r2=None,
                updates=(),
            )

        split = max(1, min(len(training) - 1, int(len(training) * 0.8)))
        train_x, train_y = self._samples(training[:split])
        test_x, test_y = self._samples(training[split:])
        candidate = self._new_model().fit(train_x, train_y)
        predictions = np.clip(candidate.predict(test_x), 0.0, 6.0)
        holdout_mae = float(mean_absolute_error(test_y, predictions))
        baseline = np.full_like(test_y, float(np.mean(train_y)))
        baseline_mae = float(mean_absolute_error(test_y, baseline))
        holdout_r2 = float(r2_score(test_y, predictions))
        if (
            not math.isfinite(holdout_mae)
            or holdout_mae > self.max_holdout_mae
            or baseline_mae - holdout_mae < self.min_baseline_improvement
        ):
            return DerivedXGResult(
                status="quality_gate_failed",
                training_matches=len(training),
                holdout_mae=round(holdout_mae, 6),
                baseline_mae=round(baseline_mae, 6),
                holdout_r2=round(holdout_r2, 6),
                updates=(),
            )

        all_x, all_y = self._samples(training)
        model = self._new_model().fit(all_x, all_y)
        targets = [
            row
            for row in ordered
            if row.xg_source != "understat"
            and (row.home_xg is None or row.xg_source == "derived_shot_model")
            and (row.away_xg is None or row.xg_source == "derived_shot_model")
            and self._match_features(row) is not None
        ]
        updates: list[dict[str, object]] = []
        for row in targets:
            match_features = self._match_features(row)
            if match_features is None:  # pragma: no cover - filtered above
                continue
            predicted = np.clip(model.predict(np.asarray(match_features)), 0.0, 6.0)
            updates.append(
                {
                    "fixture_id": row.fixture_id,
                    "home_xg": round(float(predicted[0]), 6),
                    "away_xg": round(float(predicted[1]), 6),
                    "xg_source": "derived_shot_model",
                    "xg_provider_match_id": MODEL_VERSION,
                    "xg_confidence": self.confidence,
                }
            )
        return DerivedXGResult(
            status="ready",
            training_matches=len(training),
            holdout_mae=round(holdout_mae, 6),
            baseline_mae=round(baseline_mae, 6),
            holdout_r2=round(holdout_r2, 6),
            updates=tuple(updates),
        )

    @staticmethod
    def _new_model() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=3,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        )

    @classmethod
    def _samples(
        cls, fixtures: list[HistoricalFixture]
    ) -> tuple[np.ndarray, np.ndarray]:
        features: list[list[float]] = []
        labels: list[float] = []
        for row in fixtures:
            match_features = cls._match_features(row)
            if (
                match_features is None or row.home_xg is None or row.away_xg is None
            ):  # pragma: no cover - prefiltered
                continue
            features.extend(match_features)
            labels.extend((float(row.home_xg), float(row.away_xg)))
        return np.asarray(features, dtype=float), np.asarray(labels, dtype=float)

    @classmethod
    def _match_features(cls, row: HistoricalFixture) -> list[list[float]] | None:
        home = cls._side_features(row, "home")
        away = cls._side_features(row, "away")
        return [home, away] if home is not None and away is not None else None

    @staticmethod
    def _side_features(row: HistoricalFixture, side: str) -> list[float] | None:
        values = [
            getattr(row, f"{side}_shots"),
            getattr(row, f"{side}_shots_on_target"),
            getattr(row, f"{side}_corners"),
            getattr(row, f"{side}_yellow_cards"),
            getattr(row, f"{side}_red_cards"),
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            return None
        shots, on_target, corners, yellow, red = (float(value) for value in values)
        if not (
            0 <= on_target <= shots <= 100
            and 0 <= corners <= 50
            and 0 <= yellow <= 20
            and 0 <= red <= 5
        ):
            return None
        return [shots, on_target, corners, yellow, red]

    @staticmethod
    def _valid_xg(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 15.0
        )
