from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from math import isfinite
from typing import Any, TypedDict

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import MatchPrediction

_OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")
_PSI_THRESHOLD = 0.2


class _FeatureSnapshotValidation(TypedDict):
    issues: list[str]
    valid: bool


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


def _row_total_deviation(row: MatchPrediction) -> float | None:
    """Signed deviation of the raw ML probability mass from 100%.

    ML component probabilities are stored on a 0-100 scale. Normalizing before
    the Brier score masks a model whose aggregate forecast mass drifts (e.g.
    increasingly over- or under-confident totals). Reporting the raw sum minus
    100 surfaces that calibration collapse as a fraction (0.1 == 10 points).
    """
    values = _ml_component_probabilities(row)
    if row.actual_result not in _OUTCOMES or any(value is None for value in values):
        return None
    total = 0.0
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        probability = float(value)
        if not isfinite(probability) or probability < 0:
            return None
        total += probability
    return (total - 100.0) / 100.0


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

    @staticmethod
    def _validate_feature_snapshot(snapshot: dict) -> _FeatureSnapshotValidation:
        issues: list[str] = []
        schema_version = snapshot.get("feature_schema_version")
        if not schema_version:
            issues.append("missing_feature_schema_version")
        numeric_keys_with_range = {
            "home_form": (-10.0, 100.0),
            "away_form": (-10.0, 100.0),
            "home_attack": (0.0, 10.0),
            "away_attack": (0.0, 10.0),
            "home_defense": (0.0, 10.0),
            "away_defense": (0.0, 10.0),
        }
        for key, (lo, hi) in numeric_keys_with_range.items():
            val = snapshot.get(key)
            if val is not None and isinstance(val, (int, float)):
                if not (lo <= float(val) <= hi):
                    issues.append(f"{key}_out_of_range")
        return {"issues": issues, "valid": len(issues) == 0}

    @staticmethod
    def _compute_feature_psi(
        training_stats: dict[str, tuple[float, float]],
        current_values: dict[str, list[float]],
    ) -> dict[str, object]:
        per_feature: dict[str, float] = {}
        for feature, (train_mean, train_std) in training_stats.items():
            values = current_values.get(feature, [])
            if len(values) < 10 or train_std <= 0:
                continue
            current_mean = float(np.mean(values))
            current_std = float(np.std(values))
            psi = ((current_mean - train_mean) / train_std) ** 2 + (
                current_std / train_std - 1.0
            ) ** 2
            per_feature[feature] = round(psi, 6)
        total_psi = sum(per_feature.values()) / max(1, len(per_feature))
        flagged = [f for f, v in per_feature.items() if v > _PSI_THRESHOLD]
        return {
            "total_psi": round(total_psi, 6),
            "per_feature": per_feature,
            "flagged_features": flagged,
            "drift_detected": total_psi > _PSI_THRESHOLD,
        }

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
            "recent_total_deviation": None,
            "baseline_total_deviation": None,
            "recent_abs_total_deviation": None,
            "threshold": settings.MODEL_DRIFT_BRIER_THRESHOLD,
            "total_deviation_threshold": settings.MODEL_DRIFT_TOTAL_DEVIATION_THRESHOLD,
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
        scored: list[tuple[float, float]] = []
        for row in rows:
            brier = _row_brier(row)
            deviation = _row_total_deviation(row)
            if brier is None or deviation is None:
                continue
            scored.append((brier, deviation))
            if len(scored) >= settings.MODEL_DRIFT_WINDOW_SIZE * 2:
                break
        if len(scored) < required:
            return {
                **common,
                "status": "insufficient_data",
                "samples": len(scored),
            }

        window_size = min(settings.MODEL_DRIFT_WINDOW_SIZE, len(scored) // 2)
        recent_scores = scored[:window_size]
        baseline_scores = scored[window_size : window_size * 2]
        recent_briers = [brier for brier, _ in recent_scores]
        baseline_briers = [brier for brier, _ in baseline_scores]
        recent_deviations = [deviation for _, deviation in recent_scores]
        baseline_deviations = [deviation for _, deviation in baseline_scores]
        recent_brier = sum(recent_briers) / len(recent_briers)
        baseline_brier = sum(baseline_briers) / len(baseline_briers)
        delta = recent_brier - baseline_brier
        delta_lower_bound = self._bootstrap_delta_lower_bound(
            recent_briers, baseline_briers
        )
        recent_total_deviation = sum(recent_deviations) / len(recent_deviations)
        baseline_total_deviation = sum(baseline_deviations) / len(baseline_deviations)
        recent_abs_total_deviation = sum(
            abs(deviation) for deviation in recent_deviations
        ) / len(recent_deviations)
        drift_detected = (
            (
                delta >= settings.MODEL_DRIFT_BRIER_THRESHOLD
                and delta_lower_bound >= settings.MODEL_DRIFT_BRIER_THRESHOLD
            )
            or recent_abs_total_deviation
            >= settings.MODEL_DRIFT_TOTAL_DEVIATION_THRESHOLD
        )

        per_league: dict[str, dict[str, object]] = {}
        league_scores: dict[int, list[float]] = {}
        for row in rows[: len(scored)]:
            brier = _row_brier(row)
            if brier is None:
                continue
            lid = getattr(row, "league_id", None)
            if lid is not None:
                league_scores.setdefault(int(lid), []).append(brier)
        for lid, scores in league_scores.items():
            if len(scores) < 5:
                continue
            league_brier = sum(scores) / len(scores)
            flagged = league_brier > recent_brier + 0.05 if recent_brier else False
            per_league[str(lid)] = {
                "brier_score": round(league_brier, 6),
                "samples": len(scores),
                "drift_from_global": (
                    round(league_brier - recent_brier, 6) if recent_brier else None
                ),
                "flagged": flagged,
            }

        feature_values: dict[str, list[float]] = defaultdict(list)
        for row in rows[: len(scored)]:
            snapshot = row.feature_snapshot
            if not isinstance(snapshot, dict):
                continue
            for key, val in snapshot.items():
                if isinstance(val, (int, float)) and isfinite(val):
                    feature_values[key].append(float(val))
        training_stats: dict[str, tuple[float, float]] = {}
        for feature, values in feature_values.items():
            if len(values) >= 20:
                training_stats[feature] = (
                    float(np.mean(values)),
                    float(np.std(values)),
                )
        psi_result = (
            self._compute_feature_psi(training_stats, feature_values)
            if training_stats
            else {
                "total_psi": 0.0,
                "per_feature": {},
                "flagged_features": [],
                "drift_detected": False,
            }
        )

        snapshot_issues: dict[str, int] = defaultdict(int)
        total_snapshots_checked = 0
        for row in rows[: len(scored)]:
            snapshot = row.feature_snapshot
            if not isinstance(snapshot, dict):
                continue
            total_snapshots_checked += 1
            result = self._validate_feature_snapshot(snapshot)
            for issue in result["issues"]:
                snapshot_issues[issue] += 1
        feature_snapshot_health = {
            "total_checked": total_snapshots_checked,
            "invalid_count": sum(
                1
                for row in rows[: len(scored)]
                if isinstance(row.feature_snapshot, dict)
                and not self._validate_feature_snapshot(row.feature_snapshot)["valid"]
            ),
            "issues": dict(snapshot_issues),
        }

        return {
            **common,
            "status": "drift" if drift_detected else "stable",
            "drift_detected": drift_detected,
            "samples": len(scored),
            "window_size": window_size,
            "recent_brier": round(recent_brier, 6),
            "baseline_brier": round(baseline_brier, 6),
            "brier_delta": round(delta, 6),
            "brier_delta_lower_bound": round(delta_lower_bound, 6),
            "recent_total_deviation": round(recent_total_deviation, 6),
            "baseline_total_deviation": round(baseline_total_deviation, 6),
            "recent_abs_total_deviation": round(recent_abs_total_deviation, 6),
            "per_league": per_league,
            "feature_drift": psi_result,
            "feature_snapshot_health": feature_snapshot_health,
        }
