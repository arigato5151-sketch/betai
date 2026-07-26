from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from app.core.config import settings
from app.core.logging_config import logger


@dataclass(frozen=True)
class EnsembleSample:
    timestamp: datetime
    league_id: int | None
    probabilities: dict[str, np.ndarray]
    actual_index: int
    quality: float

    @property
    def source_key(self) -> str:
        return "+".join(
            source
            for source in EnsembleWeightManager.SOURCES
            if source in self.probabilities
        )


class EnsembleWeightManager:
    """Learn time-decayed Bayesian source weights globally and per league."""

    SOURCES = ("stats", "ml", "market")
    OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")
    SCHEMA_VERSION = 2
    METHOD = "league_prequential_bma"
    SIGNATURE_DOMAIN = b"bet-ai:ensemble-bma:v2\0"

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
    def _normalized_weights(
        cls,
        weights: dict[str, float],
        sources: Sequence[str] | None = None,
    ) -> dict[str, float]:
        selected_sources = tuple(sources or cls.SOURCES)
        if (
            not selected_sources
            or len(selected_sources) != len(set(selected_sources))
            or any(source not in cls.SOURCES for source in selected_sources)
        ):
            raise ValueError("Ensemble sources are invalid")
        try:
            values = {source: float(weights[source]) for source in selected_sources}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Ensemble weights are incomplete") from exc
        total = sum(values.values())
        if total <= 0 or any(
            not math.isfinite(value) or value < 0 for value in values.values()
        ):
            raise ValueError(
                "Ensemble weights must be finite, non-negative, and non-zero"
            )
        return {source: value / total for source, value in values.items()}

    @classmethod
    def _project_weight_floor(cls, weights: dict[str, float]) -> dict[str, float]:
        normalized = cls._normalized_weights(weights, tuple(weights))
        source_count = len(normalized)
        floor = settings.ENSEMBLE_MIN_SOURCE_WEIGHT
        if floor * source_count >= 1:
            raise ValueError("Ensemble source floor leaves no posterior mass")
        residual = 1.0 - floor * source_count
        return {source: floor + residual * normalized[source] for source in normalized}

    @classmethod
    def _normalize_probabilities(cls, raw: Any) -> np.ndarray | None:
        if not isinstance(raw, dict):
            return None
        try:
            values = np.asarray(
                [float(raw[outcome]) for outcome in cls.OUTCOMES],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
            return None
        return values / values.sum()

    @staticmethod
    def _timestamp(row: Any) -> datetime:
        for attribute in (
            "kickoff",
            "feature_snapshot_at",
            "analyzed_at",
            "created_at",
        ):
            value = getattr(row, attribute, None)
            if isinstance(value, datetime):
                return value.replace(tzinfo=value.tzinfo or UTC).astimezone(UTC)
            if isinstance(value, str) and value.strip():
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    continue
                return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)
        return datetime(1970, 1, 1, tzinfo=UTC)

    @staticmethod
    def _quality(row: Any) -> float:
        snapshot = getattr(row, "data_quality", None)
        raw_score = snapshot.get("score") if isinstance(snapshot, dict) else None
        if raw_score is None:
            return 1.0
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(score):
            return 0.0
        return min(max(score / 100.0, 0.0), 1.0)

    @classmethod
    def _is_point_in_time_valid(
        cls, row: Any, probabilities: dict[str, np.ndarray]
    ) -> bool:
        lead_minutes = getattr(row, "analysis_lead_minutes", None)
        if lead_minutes is not None:
            try:
                if float(lead_minutes) <= 0:
                    return False
            except (TypeError, ValueError):
                return False

        kickoff = getattr(row, "kickoff", None)
        analyzed_at = getattr(row, "analyzed_at", None)
        market_at = getattr(row, "market_snapshot_at", None)
        if isinstance(kickoff, datetime):
            kickoff = kickoff.replace(tzinfo=kickoff.tzinfo or UTC).astimezone(UTC)
            for timestamp in (
                analyzed_at,
                market_at if "market" in probabilities else None,
            ):
                if isinstance(timestamp, datetime):
                    normalized = timestamp.replace(
                        tzinfo=timestamp.tzinfo or UTC
                    ).astimezone(UTC)
                    if normalized >= kickoff:
                        return False
        return True

    @classmethod
    def _extract_samples(cls, rows: Iterable[Any]) -> list[EnsembleSample]:
        outcome_indices = {outcome: index for index, outcome in enumerate(cls.OUTCOMES)}
        samples: list[EnsembleSample] = []
        for row in rows:
            actual_result = getattr(row, "actual_result", None)
            actual_index = (
                outcome_indices.get(actual_result)
                if isinstance(actual_result, str)
                else None
            )
            snapshot = getattr(row, "probability_components", None)
            components = (
                snapshot.get("components") if isinstance(snapshot, dict) else None
            )
            if actual_index is None or not isinstance(components, dict):
                continue

            probabilities = {
                source: normalized
                for source in cls.SOURCES
                if (normalized := cls._normalize_probabilities(components.get(source)))
                is not None
            }
            if "stats" not in probabilities or len(probabilities) < 2:
                continue
            if not cls._is_point_in_time_valid(row, probabilities):
                continue

            quality = cls._quality(row)
            if quality * 100 < settings.ENSEMBLE_BMA_MIN_DATA_QUALITY_SCORE:
                continue
            league_id = getattr(row, "league_id", None)
            samples.append(
                EnsembleSample(
                    timestamp=cls._timestamp(row),
                    league_id=(
                        league_id
                        if isinstance(league_id, int) and league_id > 0
                        else None
                    ),
                    probabilities=probabilities,
                    actual_index=actual_index,
                    quality=quality,
                )
            )
        return sorted(
            samples,
            key=lambda sample: (sample.timestamp, sample.league_id or 0),
        )

    @classmethod
    def _scope_sources(cls, source_key: str) -> tuple[str, ...]:
        requested = set(source_key.split("+"))
        sources = tuple(source for source in cls.SOURCES if source in requested)
        if "stats" not in sources or len(sources) < 2:
            raise ValueError("BMA source scope must include stats and another source")
        return sources

    @staticmethod
    def _softmax(log_weights: np.ndarray) -> np.ndarray:
        shifted = log_weights - float(np.max(log_weights))
        values = np.exp(shifted)
        total = float(values.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("BMA posterior normalization failed")
        return values / total

    @classmethod
    def _quality_adjusted_prior(
        cls,
        sources: tuple[str, ...],
        *,
        effective_samples: float,
        surprise_rate: float,
    ) -> tuple[dict[str, float], float]:
        prior = cls._normalized_weights(cls._configured_weights(), sources)
        confidence = effective_samples / (
            effective_samples + settings.ENSEMBLE_BMA_PRIOR_STRENGTH
        )
        quality_score = confidence * (1.0 - min(max(surprise_rate, 0.0), 1.0))

        adjusted = dict(prior)
        if "ml" in adjusted:
            adjusted["ml"] *= 1.0 + quality_score * (
                settings.ENSEMBLE_BMA_ML_HIGH_QUALITY_BOOST - 1.0
            )
        adjusted["stats"] *= 1.0 + (1.0 - quality_score) * (
            settings.ENSEMBLE_BMA_STATS_LOW_DATA_BOOST - 1.0
        )
        return cls._project_weight_floor(adjusted), quality_score

    @classmethod
    def _posterior_profile(
        cls, samples: Sequence[EnsembleSample]
    ) -> dict[str, Any] | None:
        if not samples:
            return None
        source_key = samples[0].source_key
        if any(sample.source_key != source_key for sample in samples):
            raise ValueError("BMA profile samples must share an exact source scope")
        if {sample.actual_index for sample in samples} != set(range(len(cls.OUTCOMES))):
            return None

        sources = cls._scope_sources(source_key)
        effective_samples = float(sum(sample.quality for sample in samples))
        reference_source = "ml" if "ml" in sources else "stats"
        surprises = [
            float(
                int(np.argmax(sample.probabilities[reference_source]))
                != sample.actual_index
            )
            for sample in samples
        ]
        surprise_rate = float(np.mean(surprises))
        prior, quality_score = cls._quality_adjusted_prior(
            sources,
            effective_samples=effective_samples,
            surprise_rate=surprise_rate,
        )
        log_prior = np.log(np.asarray([prior[source] for source in sources]))
        log_posterior = log_prior.copy()
        last_timestamp: datetime | None = None

        for sample in samples:
            if last_timestamp is not None:
                gap_days = max(
                    0.0,
                    (sample.timestamp - last_timestamp).total_seconds() / 86400.0,
                )
                retention = 0.5 ** (gap_days / settings.ENSEMBLE_BMA_HALF_LIFE_DAYS)
                log_posterior = log_prior + retention * (log_posterior - log_prior)

            likelihoods = np.asarray(
                [
                    float(
                        np.clip(
                            sample.probabilities[source][sample.actual_index],
                            1e-12,
                            1.0,
                        )
                    )
                    for source in sources
                ]
            )
            log_posterior += sample.quality * np.log(likelihoods)
            posterior = cls._softmax(log_posterior)
            floored = cls._project_weight_floor(
                dict(zip(sources, posterior.tolist(), strict=True))
            )
            log_posterior = np.log(np.asarray([floored[source] for source in sources]))
            last_timestamp = sample.timestamp

        weights = dict(zip(sources, cls._softmax(log_posterior).tolist(), strict=True))
        weights = cls._project_weight_floor(weights)
        source_log_loss = {
            source: float(
                np.mean(
                    [
                        -math.log(
                            float(
                                np.clip(
                                    sample.probabilities[source][sample.actual_index],
                                    1e-12,
                                    1.0,
                                )
                            )
                        )
                        for sample in samples
                    ]
                )
            )
            for source in sources
        }
        entropy = -sum(
            weight * math.log(max(weight, 1e-12)) for weight in weights.values()
        )
        return {
            "weights": weights,
            "samples": len(samples),
            "effective_samples": effective_samples,
            "surprise_rate": surprise_rate,
            "data_quality_score": quality_score,
            "posterior_entropy": entropy,
            "source_log_loss": source_log_loss,
            "data_cutoff_at": samples[-1].timestamp.isoformat(),
        }

    @classmethod
    def _blend(cls, sample: EnsembleSample, weights: dict[str, float]) -> np.ndarray:
        sources = tuple(sample.probabilities)
        normalized = cls._normalized_weights(weights, sources)
        blended = np.zeros(len(cls.OUTCOMES), dtype=float)
        for source in sources:
            blended += normalized[source] * sample.probabilities[source]
        return blended

    @classmethod
    def _score(
        cls,
        samples: Sequence[EnsembleSample],
        profiles: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        candidate_losses: list[float] = []
        baseline_losses: list[float] = []
        candidate_briers: list[float] = []
        baseline_briers: list[float] = []
        for sample in samples:
            profile = profiles.get(sample.source_key)
            if profile is None:
                continue
            sources = tuple(sample.probabilities)
            candidate = cls._blend(sample, profile["weights"])
            baseline = cls._blend(
                sample,
                cls._normalized_weights(cls._configured_weights(), sources),
            )
            target = np.zeros(len(cls.OUTCOMES), dtype=float)
            target[sample.actual_index] = 1.0
            candidate_losses.append(
                -math.log(float(np.clip(candidate[sample.actual_index], 1e-12, 1.0)))
            )
            baseline_losses.append(
                -math.log(float(np.clip(baseline[sample.actual_index], 1e-12, 1.0)))
            )
            candidate_briers.append(float(np.sum((candidate - target) ** 2)))
            baseline_briers.append(float(np.sum((baseline - target) ** 2)))
        if not candidate_losses:
            return {
                "samples": 0.0,
                "candidate_log_loss": float("inf"),
                "baseline_log_loss": float("inf"),
                "candidate_brier": float("inf"),
                "baseline_brier": float("inf"),
            }
        return {
            "samples": float(len(candidate_losses)),
            "candidate_log_loss": float(np.mean(candidate_losses)),
            "baseline_log_loss": float(np.mean(baseline_losses)),
            "candidate_brier": float(np.mean(candidate_briers)),
            "baseline_brier": float(np.mean(baseline_briers)),
        }

    @classmethod
    def _group_profiles(
        cls, samples: Sequence[EnsembleSample]
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[EnsembleSample]] = defaultdict(list)
        for sample in samples:
            grouped[sample.source_key].append(sample)
        return {
            source_key: profile
            for source_key, group in grouped.items()
            if (profile := cls._posterior_profile(group)) is not None
        }

    @classmethod
    def _league_profiles(
        cls,
        train_samples: Sequence[EnsembleSample],
        validation_samples: Sequence[EnsembleSample],
        all_samples: Sequence[EnsembleSample],
        global_train_profiles: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        train_groups: dict[tuple[int, str], list[EnsembleSample]] = defaultdict(list)
        validation_groups: dict[tuple[int, str], list[EnsembleSample]] = defaultdict(
            list
        )
        full_groups: dict[tuple[int, str], list[EnsembleSample]] = defaultdict(list)
        for target, samples in (
            (train_groups, train_samples),
            (validation_groups, validation_samples),
            (full_groups, all_samples),
        ):
            for sample in samples:
                if sample.league_id is not None:
                    target[(sample.league_id, sample.source_key)].append(sample)

        accepted: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        minimum = settings.ENSEMBLE_BMA_MIN_LEAGUE_SAMPLES
        for key, full_group in full_groups.items():
            league_id, source_key = key
            train_group = train_groups.get(key, [])
            validation_group = validation_groups.get(key, [])
            if (
                sum(sample.quality for sample in full_group) < minimum
                or not validation_group
            ):
                continue
            train_profile = cls._posterior_profile(train_group)
            global_profile = global_train_profiles.get(source_key)
            if train_profile is None or global_profile is None:
                continue

            candidate_metrics = cls._score(
                validation_group, {source_key: train_profile}
            )
            global_metrics = cls._score(validation_group, {source_key: global_profile})
            log_loss_improvement = (
                global_metrics["candidate_log_loss"]
                - candidate_metrics["candidate_log_loss"]
            )
            brier_regression = (
                candidate_metrics["candidate_brier"] - global_metrics["candidate_brier"]
            )
            if (
                log_loss_improvement < settings.ENSEMBLE_MIN_LOG_LOSS_IMPROVEMENT
                or brier_regression > settings.ENSEMBLE_BMA_MAX_BRIER_REGRESSION
            ):
                continue

            full_profile = cls._posterior_profile(full_group)
            if full_profile is None:
                continue
            full_profile["holdout_log_loss_improvement_vs_global"] = (
                log_loss_improvement
            )
            full_profile["holdout_brier_regression_vs_global"] = brier_regression
            accepted[str(league_id)][source_key] = full_profile
        return dict(accepted)

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
        global_train_profiles = self._group_profiles(train_samples)
        metrics = self._score(validation_samples, global_train_profiles)
        improvement = metrics["baseline_log_loss"] - metrics["candidate_log_loss"]
        brier_regression = metrics["candidate_brier"] - metrics["baseline_brier"]
        if (
            metrics["samples"] <= 0
            or improvement < settings.ENSEMBLE_MIN_LOG_LOSS_IMPROVEMENT
            or brier_regression > settings.ENSEMBLE_BMA_MAX_BRIER_REGRESSION
        ):
            return {
                "status": "rejected",
                "samples": len(samples),
                "evaluated_samples": int(metrics["samples"]),
                "baseline_log_loss": metrics["baseline_log_loss"],
                "candidate_log_loss": metrics["candidate_log_loss"],
                "baseline_brier": metrics["baseline_brier"],
                "candidate_brier": metrics["candidate_brier"],
                "improvement": improvement,
            }

        global_profiles = self._group_profiles(samples)
        league_profiles = self._league_profiles(
            train_samples,
            validation_samples,
            samples,
            global_train_profiles,
        )
        artifact = {
            "schema_version": self.SCHEMA_VERSION,
            "method": self.METHOD,
            "artifact_version": datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"),
            "trained_at": datetime.now(UTC).isoformat(),
            "data_cutoff_at": samples[-1].timestamp.isoformat(),
            "samples": len(samples),
            "training_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "baseline_log_loss": metrics["baseline_log_loss"],
            "validation_log_loss": metrics["candidate_log_loss"],
            "baseline_brier": metrics["baseline_brier"],
            "validation_brier": metrics["candidate_brier"],
            "global": global_profiles,
            "leagues": league_profiles,
        }
        try:
            self._write_artifact(artifact)
        except OSError as exc:
            logger.exception("Could not persist ensemble BMA artifact")
            return {
                "status": "artifact_write_failed",
                "samples": len(samples),
                "error": type(exc).__name__,
            }
        logger.info(
            "Activated league BMA artifact %s with validation log-loss %.6f",
            artifact["artifact_version"],
            metrics["candidate_log_loss"],
        )
        return {"status": "activated", **artifact}

    @classmethod
    def _canonical_payload(cls, artifact: dict[str, Any]) -> bytes:
        unsigned = {key: value for key, value in artifact.items() if key != "signature"}
        return json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def _signature(cls, artifact: dict[str, Any]) -> str:
        return hmac.new(
            settings.MODEL_SIGNING_KEY.encode("utf-8"),
            cls.SIGNATURE_DOMAIN + cls._canonical_payload(artifact),
            hashlib.sha256,
        ).hexdigest()

    @classmethod
    def _verify_signature(cls, artifact: dict[str, Any]) -> bool:
        expected = artifact.get("signature")
        return isinstance(expected, str) and hmac.compare_digest(
            expected, cls._signature(artifact)
        )

    def _write_artifact(self, artifact: dict[str, Any]) -> None:
        path = Path(settings.ENSEMBLE_WEIGHTS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        signed_artifact = dict(artifact)
        signed_artifact["signature"] = self._signature(signed_artifact)
        temporary_path = path.with_name(
            f".{path.name}.{artifact['artifact_version']}.tmp"
        )
        temporary_path.write_text(
            json.dumps(signed_artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
        with self._lock:
            self._cached_path = path
            self._cached_mtime_ns = path.stat().st_mtime_ns
            self._cached_artifact = signed_artifact

    @classmethod
    def _validated_profile_weights(
        cls,
        profile: object,
        sources: tuple[str, ...],
    ) -> dict[str, float] | None:
        if not isinstance(profile, dict):
            return None
        weights = profile.get("weights")
        if not isinstance(weights, dict):
            return None
        try:
            return cls._normalized_weights(weights, sources)
        except ValueError:
            return None

    def _load_artifact(self) -> dict[str, Any] | None:
        path = Path(settings.ENSEMBLE_WEIGHTS_PATH)
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return None

        with self._lock:
            if (
                self._cached_path != path
                or self._cached_mtime_ns != mtime_ns
                or self._cached_artifact is None
            ):
                try:
                    artifact = json.loads(path.read_text(encoding="utf-8"))
                    schema_version = artifact.get("schema_version")
                    if schema_version == 1:
                        self._normalized_weights(artifact["weights"])
                    elif schema_version == self.SCHEMA_VERSION:
                        if artifact.get("method") != self.METHOD:
                            raise ValueError("Unsupported ensemble BMA method")
                        if not self._verify_signature(artifact):
                            raise ValueError(
                                "Ensemble BMA artifact signature verification failed"
                            )
                        if not isinstance(
                            artifact.get("global"), dict
                        ) or not isinstance(artifact.get("leagues"), dict):
                            raise ValueError("Ensemble BMA profiles are invalid")
                    else:
                        raise ValueError("Unsupported ensemble weight schema")
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    json.JSONDecodeError,
                ) as exc:
                    logger.warning("Ignoring invalid ensemble weight artifact: %s", exc)
                    return None
                self._cached_path = path
                self._cached_mtime_ns = mtime_ns
                self._cached_artifact = artifact
            return self._cached_artifact

    @classmethod
    def _low_data_weights(cls, sources: tuple[str, ...]) -> dict[str, float]:
        weights, _ = cls._quality_adjusted_prior(
            sources,
            effective_samples=0.0,
            surprise_rate=1.0,
        )
        return weights

    def get_active_weights(
        self,
        league_id: int | None = None,
        available_sources: Sequence[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        sources = tuple(
            source
            for source in self.SOURCES
            if available_sources is None or source in available_sources
        )
        if not sources:
            raise ValueError("At least one ensemble source is required")
        if len(sources) == 1:
            return {sources[0]: 1.0}, {
                "source": "single_source",
                "league_id": league_id,
                "source_set": "+".join(sources),
                "artifact_version": None,
            }

        configured = self._normalized_weights(self._configured_weights(), sources)
        artifact = self._load_artifact()
        if artifact is None:
            fallback = (
                self._low_data_weights(sources) if league_id is not None else configured
            )
            return fallback, {
                "source": ("low_data_prior" if league_id is not None else "configured"),
                "league_id": league_id,
                "source_set": "+".join(sources),
                "artifact_version": None,
            }

        if artifact.get("schema_version") == 1:
            weights = self._normalized_weights(artifact["weights"], sources)
            return weights, {
                "source": "learned_v1",
                "league_id": league_id,
                "source_set": "+".join(sources),
                "artifact_version": artifact.get("artifact_version"),
                "validation_log_loss": artifact.get("validation_log_loss"),
                "samples": artifact.get("samples"),
            }

        source_key = "+".join(sources)
        if league_id is not None:
            league_profiles = artifact["leagues"].get(str(league_id), {})
            profile = (
                league_profiles.get(source_key)
                if isinstance(league_profiles, dict)
                else None
            )
            profile_weights = self._validated_profile_weights(profile, sources)
            if profile_weights is not None and isinstance(profile, dict):
                return profile_weights, {
                    "source": "league_bma",
                    "league_id": league_id,
                    "source_set": source_key,
                    "artifact_version": artifact.get("artifact_version"),
                    "samples": profile.get("samples"),
                    "effective_samples": profile.get("effective_samples"),
                    "data_quality_score": profile.get("data_quality_score"),
                    "surprise_rate": profile.get("surprise_rate"),
                    "posterior_entropy": profile.get("posterior_entropy"),
                }
            return self._low_data_weights(sources), {
                "source": "low_data_prior",
                "league_id": league_id,
                "source_set": source_key,
                "artifact_version": artifact.get("artifact_version"),
            }

        profile = artifact["global"].get(source_key)
        profile_weights = self._validated_profile_weights(profile, sources)
        if profile_weights is not None and isinstance(profile, dict):
            return profile_weights, {
                "source": "global_bma",
                "league_id": None,
                "source_set": source_key,
                "artifact_version": artifact.get("artifact_version"),
                "samples": profile.get("samples"),
                "effective_samples": profile.get("effective_samples"),
                "data_quality_score": profile.get("data_quality_score"),
                "surprise_rate": profile.get("surprise_rate"),
                "posterior_entropy": profile.get("posterior_entropy"),
            }
        return configured, {
            "source": "configured",
            "league_id": None,
            "source_set": source_key,
            "artifact_version": artifact.get("artifact_version"),
        }


ensemble_weight_manager = EnsembleWeightManager()
