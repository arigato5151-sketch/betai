from __future__ import annotations

import json
import math
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize

from app.core.config import settings
from app.core.logging_config import logger


class EnsembleWeightManager:
    SOURCES = ("stats", "ml", "market")
    OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")
    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cached_path: Path | None = None
        self._cached_mtime_ns: int | None = None
        self._cached_artifact: dict[str, Any] | None = None

    @staticmethod
    def _configured_weights() -> dict[str, float]:
        return {
            "stats": settings.ENSEMBLE_STATS_WEIGHT,
            "ml": settings.ENSEMBLE_ML_WEIGHT,
            "market": settings.ENSEMBLE_MARKET_WEIGHT,
        }

    @classmethod
    def _normalized_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        values = {source: float(weights[source]) for source in cls.SOURCES}
        total = sum(values.values())
        if total <= 0 or any(
            not math.isfinite(value) or value < 0 for value in values.values()
        ):
            raise ValueError(
                "Ensemble weights must be finite, non-negative, and non-zero"
            )
        return {source: value / total for source, value in values.items()}

    @classmethod
    def _normalize_probabilities(cls, raw: Any) -> np.ndarray | None:
        if not isinstance(raw, dict):
            return None
        try:
            values = np.asarray([float(raw[outcome]) for outcome in cls.OUTCOMES])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
            return None
        return values / values.sum()

    @classmethod
    def _extract_samples(cls, rows: Iterable[Any]) -> list[tuple[np.ndarray, int]]:
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                str(getattr(row, "created_at", "") or ""),
                int(getattr(row, "id", 0) or 0),
            ),
        )
        outcome_indices = {outcome: index for index, outcome in enumerate(cls.OUTCOMES)}
        samples: list[tuple[np.ndarray, int]] = []
        for row in ordered_rows:
            actual_result = getattr(row, "actual_result", None)
            if not isinstance(actual_result, str):
                continue
            actual_index = outcome_indices.get(actual_result)
            snapshot = getattr(row, "probability_components", None)
            components = (
                snapshot.get("components") if isinstance(snapshot, dict) else None
            )
            if actual_index is None or not isinstance(components, dict):
                continue
            source_probabilities = [
                cls._normalize_probabilities(components.get(source))
                for source in cls.SOURCES
            ]
            if any(probabilities is None for probabilities in source_probabilities):
                continue
            valid_probabilities = [
                probabilities
                for probabilities in source_probabilities
                if probabilities is not None
            ]
            samples.append((np.stack(valid_probabilities), actual_index))
        return samples

    @staticmethod
    def _log_loss(samples: list[tuple[np.ndarray, int]], weights: np.ndarray) -> float:
        losses = []
        for probabilities, actual_index in samples:
            blended = weights @ probabilities
            losses.append(-math.log(float(np.clip(blended[actual_index], 1e-12, 1.0))))
        return float(np.mean(losses)) if losses else float("inf")

    def optimize_and_activate(self, rows: Iterable[Any]) -> dict[str, Any]:
        samples = self._extract_samples(rows)
        minimum_samples = settings.MIN_ENSEMBLE_CALIBRATION_SAMPLES
        if len(samples) < minimum_samples:
            return {
                "status": "insufficient_data",
                "samples": len(samples),
                "required_samples": minimum_samples,
            }

        split_index = int(len(samples) * (1.0 - settings.ENSEMBLE_HOLDOUT_FRACTION))
        split_index = min(max(split_index, 1), len(samples) - 1)
        train_samples = samples[:split_index]
        validation_samples = samples[split_index:]
        configured = self._normalized_weights(self._configured_weights())
        initial = np.asarray([configured[source] for source in self.SOURCES])
        minimum_weight = settings.ENSEMBLE_MIN_SOURCE_WEIGHT

        result = minimize(
            lambda candidate: self._log_loss(train_samples, candidate),
            initial,
            method="SLSQP",
            bounds=[(minimum_weight, 1.0) for _ in self.SOURCES],
            constraints={"type": "eq", "fun": lambda candidate: candidate.sum() - 1.0},
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            logger.warning("Ensemble weight optimization failed: %s", result.message)
            return {"status": "optimization_failed", "samples": len(samples)}

        candidate = np.asarray(result.x, dtype=float)
        candidate /= candidate.sum()
        baseline_loss = self._log_loss(validation_samples, initial)
        candidate_loss = self._log_loss(validation_samples, candidate)
        improvement = baseline_loss - candidate_loss
        if improvement < settings.ENSEMBLE_MIN_LOG_LOSS_IMPROVEMENT:
            return {
                "status": "rejected",
                "samples": len(samples),
                "baseline_log_loss": baseline_loss,
                "candidate_log_loss": candidate_loss,
                "improvement": improvement,
            }

        weights = {
            source: float(candidate[index]) for index, source in enumerate(self.SOURCES)
        }
        artifact = {
            "schema_version": self.SCHEMA_VERSION,
            "artifact_version": datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"),
            "trained_at": datetime.now(UTC).isoformat(),
            "samples": len(samples),
            "training_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "baseline_log_loss": baseline_loss,
            "validation_log_loss": candidate_loss,
            "weights": weights,
        }
        try:
            self._write_artifact(artifact)
        except OSError as exc:
            logger.exception("Could not persist ensemble weight artifact")
            return {
                "status": "artifact_write_failed",
                "samples": len(samples),
                "error": type(exc).__name__,
            }
        logger.info(
            "Activated ensemble weight artifact %s with validation log-loss %.6f",
            artifact["artifact_version"],
            candidate_loss,
        )
        return {"status": "activated", **artifact}

    def _write_artifact(self, artifact: dict[str, Any]) -> None:
        path = Path(settings.ENSEMBLE_WEIGHTS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(
            f".{path.name}.{artifact['artifact_version']}.tmp"
        )
        temporary_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
        with self._lock:
            self._cached_path = path
            self._cached_mtime_ns = path.stat().st_mtime_ns
            self._cached_artifact = artifact

    def get_active_weights(self) -> tuple[dict[str, float], dict[str, Any]]:
        configured = self._configured_weights()
        path = Path(settings.ENSEMBLE_WEIGHTS_PATH)
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return configured, {"source": "configured", "artifact_version": None}

        with self._lock:
            if (
                self._cached_path != path
                or self._cached_mtime_ns != mtime_ns
                or self._cached_artifact is None
            ):
                try:
                    artifact = json.loads(path.read_text(encoding="utf-8"))
                    if artifact.get("schema_version") != self.SCHEMA_VERSION:
                        raise ValueError("Unsupported ensemble weight schema")
                    self._normalized_weights(artifact["weights"])
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    json.JSONDecodeError,
                ) as exc:
                    logger.warning("Ignoring invalid ensemble weight artifact: %s", exc)
                    return configured, {
                        "source": "configured",
                        "artifact_version": None,
                    }
                self._cached_path = path
                self._cached_mtime_ns = mtime_ns
                self._cached_artifact = artifact

            artifact = self._cached_artifact
            if artifact is None:
                return configured, {"source": "configured", "artifact_version": None}
            weights = self._normalized_weights(artifact["weights"])
            return weights, {
                "source": "learned",
                "artifact_version": artifact.get("artifact_version"),
                "validation_log_loss": artifact.get("validation_log_loss"),
                "samples": artifact.get("samples"),
            }


ensemble_weight_manager = EnsembleWeightManager()
