from __future__ import annotations

import math
from typing import Any

from app.core.config import settings
from app.db.models import MatchPrediction


class ActiveLearningSelector:
    """Ranks unlabeled predictions by normalized entropy and class margin."""

    @staticmethod
    def should_retrain(labeled_count: int, model_ready: bool) -> bool:
        if labeled_count < settings.MIN_TRAINING_SAMPLES:
            return False
        if not model_ready:
            return True
        return labeled_count % settings.RETRAIN_EVERY_N_NEW == 0

    @staticmethod
    def _uncertainty(probabilities: list[float]) -> tuple[float, float, float]:
        total = sum(probabilities)
        if total <= 0:
            return 0.0, 0.0, 0.0

        normalized = [probability / total for probability in probabilities]
        entropy = -sum(
            probability * math.log(probability)
            for probability in normalized
            if probability > 0
        ) / math.log(len(normalized))
        ordered = sorted(normalized, reverse=True)
        margin = ordered[0] - ordered[1]
        score = (0.7 * entropy) + (0.3 * (1.0 - margin))
        return round(score, 6), round(entropy, 6), round(margin, 6)

    @classmethod
    def rank(
        cls, predictions: list[MatchPrediction], limit: int = 20
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for prediction in predictions:
            if prediction.actual_result is not None:
                continue
            probabilities = (
                prediction.prob_home,
                prediction.prob_draw,
                prediction.prob_away,
            )
            if any(
                probability is None or probability < 0 for probability in probabilities
            ):
                continue
            if (
                sum(
                    float(probability)
                    for probability in probabilities
                    if probability is not None
                )
                <= 0
            ):
                continue

            score, entropy, margin = cls._uncertainty(
                [
                    float(probability)
                    for probability in probabilities
                    if probability is not None
                ]
            )
            candidates.append(
                {
                    "id": prediction.id,
                    "fixture_id": prediction.fixture_id,
                    "home_team": prediction.home_team,
                    "away_team": prediction.away_team,
                    "prediction": prediction.prediction,
                    "probabilities": {
                        "HOME_WIN": prediction.prob_home,
                        "DRAW": prediction.prob_draw,
                        "AWAY_WIN": prediction.prob_away,
                    },
                    "uncertainty_score": score,
                    "normalized_entropy": entropy,
                    "class_margin": margin,
                    "created_at": prediction.created_at,
                }
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate["uncertainty_score"],
                candidate["id"] or 0,
            )
        )
        return candidates[:limit]
