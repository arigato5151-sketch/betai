"""Signed artifact storage and inference routing for multi-tier ML models."""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.football_data_csv import FOOTBALL_DATA_LEAGUE_IDS
from app.prediction.ml.ml_pipeline import Tier1Model, Tier2Model


class TieredArtifactIntegrityError(RuntimeError):
    """Raised when a tiered artifact is missing, malformed, or has a bad HMAC."""


@dataclass(frozen=True)
class TieredModelBundle:
    """Validated models and metadata loaded from a signed active artifact."""

    tier1_model: Tier1Model
    tier2_model: Tier2Model
    artifact_version: str
    trained_at: str
    metadata: dict[str, object]
    tier2_gate: dict[str, object] | None = None
    tier1_gate: dict[str, object] | None = None


@dataclass(frozen=True)
class RoutedPrediction:
    """A normalized 1X2 prediction paired with the selected model tier."""

    tier: str
    probabilities: tuple[float, float, float]
    artifact_version: str | None
    research_only: bool = True


class TieredModelArtifactStore:
    """Persist Tier 1/2 models with the same HMAC contract as active_model.pkl."""

    SCHEMA_VERSION = 2

    def __init__(self, *, artifacts_dir: Path | str | None = None) -> None:
        root = Path(artifacts_dir or settings.MODEL_ARTIFACTS_DIR)
        self.artifacts_dir = root / "tiered"
        self.versions_dir = self.artifacts_dir / "versions"
        self.active_path = self.artifacts_dir / "tiered_active_model.joblib"
        self.previous_path = self.artifacts_dir / "tiered_previous_model.joblib"

    @staticmethod
    def signature_path(artifact_path: Path) -> Path:
        return artifact_path.with_name(f"{artifact_path.name}.sig")

    @classmethod
    def artifact_signature(cls, artifact_path: Path) -> str:
        digest = hmac.new(
            settings.MODEL_SIGNING_KEY.encode("utf-8"), digestmod=hashlib.sha256
        )
        with artifact_path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def verify(cls, artifact_path: Path) -> bool:
        try:
            expected = (
                cls.signature_path(artifact_path).read_text(encoding="ascii").strip()
            )
            actual = cls.artifact_signature(artifact_path)
        except OSError:
            return False
        return hmac.compare_digest(expected, actual)

    def export(
        self,
        tier1_model: Tier1Model,
        tier2_model: Tier2Model,
        *,
        tier1_metrics: Mapping[str, object],
        tier2_metrics: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
    ) -> TieredModelBundle:
        """Atomically promote a newly trained, signed Tier 1/2 bundle."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        trained_at = datetime.now(UTC).isoformat()
        raw_tier2_gate = tier2_metrics.get("tier2_gate")
        raw_tier1_gate = tier1_metrics.get("tier1_gate")
        bundle_metadata = {
            **dict(metadata or {}),
            "tier1_features": list(tier1_model.FEATURES),
            "tier2_features": list(tier2_model.FEATURES),
            "tier1_metrics": dict(tier1_metrics),
            "tier2_metrics": dict(tier2_metrics),
            "tier2_gate": (
                dict(raw_tier2_gate) if isinstance(raw_tier2_gate, Mapping) else {}
            ),
            "tier1_gate": (
                dict(raw_tier1_gate) if isinstance(raw_tier1_gate, Mapping) else {}
            ),
        }
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "artifact_version": version,
            "trained_at": trained_at,
            "tier1_model": tier1_model,
            "tier2_model": tier2_model,
            "metadata": bundle_metadata,
        }
        # A unique temporary name keeps concurrent trainers from clobbering
        # each other's half-written files; fsync before publishing the swap.
        temporary_path = self.active_path.with_name(
            f"{self.active_path.name}.tmp.{uuid.uuid4().hex}"
        )
        temporary_signature_path = self.signature_path(temporary_path)
        try:
            with temporary_path.open("wb") as artifact_file:
                joblib.dump(payload, artifact_file)
                artifact_file.flush()
                os.fsync(artifact_file.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        temporary_signature_path.write_text(
            self.artifact_signature(temporary_path), encoding="ascii"
        )

        active_signature_path = self.signature_path(self.active_path)
        previous_signature_path = self.signature_path(self.previous_path)
        if self.active_path.is_file() and self.verify(self.active_path):
            shutil.copy2(self.active_path, self.previous_path)
            shutil.copy2(active_signature_path, previous_signature_path)
        os.replace(temporary_path, self.active_path)
        os.replace(temporary_signature_path, active_signature_path)

        version_path = self.versions_dir / f"tiered_model_{version}.joblib"
        shutil.copy2(self.active_path, version_path)
        shutil.copy2(active_signature_path, self.signature_path(version_path))
        return TieredModelBundle(
            tier1_model=tier1_model,
            tier2_model=tier2_model,
            artifact_version=version,
            trained_at=trained_at,
            metadata=bundle_metadata,
        )

    def load_active(self) -> TieredModelBundle | None:
        """Load only a correctly signed artifact; never deserialize unsigned bytes."""
        if not self.active_path.is_file():
            return None
        if not self.verify(self.active_path):
            raise TieredArtifactIntegrityError(
                "Tiered model artifact HMAC verification failed"
            )
        try:
            payload = joblib.load(self.active_path)
        except (OSError, ValueError, TypeError) as exc:
            raise TieredArtifactIntegrityError(
                "Tiered model artifact could not be read"
            ) from exc
        return self._bundle_from_payload(payload)

    def rollback(self) -> bool:
        """Swap active/previous bundles only after both HMACs validate."""
        required_paths = (
            self.active_path,
            self.signature_path(self.active_path),
            self.previous_path,
            self.signature_path(self.previous_path),
        )
        if not all(path.is_file() for path in required_paths):
            return False
        if not self.verify(self.active_path) or not self.verify(self.previous_path):
            return False

        backup_path = self.active_path.with_name(
            f"{self.active_path.name}.rollback.bak"
        )
        backup_signature_path = self.signature_path(backup_path)
        candidate_path = self.active_path.with_name(
            f"{self.active_path.name}.rollback.candidate"
        )
        candidate_signature_path = self.signature_path(candidate_path)
        temporary_paths = (
            backup_path,
            backup_signature_path,
            candidate_path,
            candidate_signature_path,
        )
        try:
            for path in temporary_paths:
                path.unlink(missing_ok=True)
            # Copy (never move) the champion and candidate aside so a failure at
            # any point can restore the store to its pre-rollback state.
            shutil.copy2(self.active_path, backup_path)
            shutil.copy2(self.signature_path(self.active_path), backup_signature_path)
            shutil.copy2(self.previous_path, candidate_path)
            shutil.copy2(
                self.signature_path(self.previous_path), candidate_signature_path
            )

            os.replace(candidate_path, self.active_path)
            os.replace(candidate_signature_path, self.signature_path(self.active_path))
            self.load_active()
            # Promote the displaced champion as the next rollback candidate.
            os.replace(backup_path, self.previous_path)
            os.replace(backup_signature_path, self.signature_path(self.previous_path))
            return True
        except (OSError, TieredArtifactIntegrityError):
            try:
                if backup_path.is_file() and backup_signature_path.is_file():
                    os.replace(backup_path, self.active_path)
                    os.replace(
                        backup_signature_path, self.signature_path(self.active_path)
                    )
                    self.load_active()
            except (OSError, TieredArtifactIntegrityError):
                pass
            return False
        finally:
            for path in temporary_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    @classmethod
    def _bundle_from_payload(cls, payload: object) -> TieredModelBundle:
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != cls.SCHEMA_VERSION
        ):
            raise TieredArtifactIntegrityError(
                "Unsupported tiered model artifact schema"
            )
        tier1_model = payload.get("tier1_model")
        tier2_model = payload.get("tier2_model")
        if not isinstance(tier1_model, Tier1Model) or not isinstance(
            tier2_model, Tier2Model
        ):
            raise TieredArtifactIntegrityError(
                "Tiered artifact contains invalid models"
            )
        version = payload.get("artifact_version")
        trained_at = payload.get("trained_at")
        metadata = payload.get("metadata")
        if (
            not isinstance(version, str)
            or not isinstance(trained_at, str)
            or not isinstance(metadata, dict)
        ):
            raise TieredArtifactIntegrityError("Tiered artifact metadata is invalid")
        if metadata.get("tier1_features") != list(Tier1Model.FEATURES) or metadata.get(
            "tier2_features"
        ) != list(Tier2Model.FEATURES):
            raise TieredArtifactIntegrityError(
                "Tiered artifact feature schema is incompatible"
            )
        tier2_gate = metadata.get("tier2_gate")
        if not isinstance(tier2_gate, dict) or not tier2_gate:
            # Empty or absent gate evidence means the bundle predates the Tier 2
            # evidence floor and must be treated conservatively, not trusted.
            tier2_gate = None
        tier1_gate = metadata.get("tier1_gate")
        if not isinstance(tier1_gate, dict) or not tier1_gate:
            # Absent evidence means the bundle predates the closing-line gate or
            # was exported without gate evidence; expose it as unknown rather
            # than silently pretending the gate passed.
            tier1_gate = None
        return TieredModelBundle(
            tier1_model,
            tier2_model,
            version,
            trained_at,
            metadata,
            tier2_gate,
            tier1_gate,
        )


class Predictor:
    """Select Tier 1 only for data-rich supported leagues, otherwise Tier 2."""

    def __init__(
        self,
        tier1_model: Tier1Model,
        tier2_model: Tier2Model,
        *,
        artifact_version: str | None = None,
        tier1_league_ids: frozenset[int] = FOOTBALL_DATA_LEAGUE_IDS,
        tier2_gate: Mapping[str, object] | None = None,
        tier1_gate: Mapping[str, object] | None = None,
    ) -> None:
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self.artifact_version = artifact_version
        self.tier1_league_ids = tier1_league_ids
        self.tier2_gate = dict(tier2_gate) if tier2_gate else None
        self.tier1_gate = dict(tier1_gate) if tier1_gate else None

    @classmethod
    def from_active_artifact(cls, store: TieredModelArtifactStore) -> "Predictor":
        bundle = store.load_active()
        if bundle is None:
            raise TieredArtifactIntegrityError("No active tiered model artifact exists")
        return cls(
            bundle.tier1_model,
            bundle.tier2_model,
            artifact_version=bundle.artifact_version,
            tier2_gate=bundle.tier2_gate,
            tier1_gate=bundle.tier1_gate,
        )

    def predict(self, features: Mapping[str, object]) -> RoutedPrediction:
        """Route one match and return 1X2 probabilities in away/draw/home order."""
        league_id = self._league_id(features.get("league_id"))
        if league_id in self.tier1_league_ids and self._has_tier1_features(features):
            model: Tier1Model | Tier2Model = self.tier1_model
            tier = "tier1"
        else:
            model = self.tier2_model
            tier = "tier2"
        frame = self._feature_frame(features, model)
        probabilities = np.asarray(model.predict_proba(frame), dtype=float)
        if probabilities.shape != (1, 3) or not np.isfinite(probabilities).all():
            raise ValueError("Selected model returned invalid 1X2 probabilities")
        if not np.isclose(probabilities[0].sum(), 1.0, atol=1e-6):
            raise ValueError("Selected model probabilities must sum to one")
        research_only = True
        if self.tier2_gate is not None:
            # A modern bundle proves Tier 1 superiority against the closing
            # line at promotion time; Tier 2 stays research-only until the
            # evidence floors are met. Legacy bundles without the gate record
            # are treated conservatively as research-only as well.
            research_only = (
                False if tier == "tier1" else self.tier2_gate.get("passed") is not True
            )
        return RoutedPrediction(
            tier=tier,
            probabilities=(
                float(probabilities[0, 0]),
                float(probabilities[0, 1]),
                float(probabilities[0, 2]),
            ),
            artifact_version=self.artifact_version,
            research_only=research_only,
        )

    @staticmethod
    def _league_id(value: object) -> int | None:
        try:
            parsed = int(str(value))
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _has_tier1_features(features: Mapping[str, object]) -> bool:
        return all(
            value is not None and not (isinstance(value, float) and np.isnan(value))
            for name in Tier1Model.FEATURES
            if name not in Tier1Model.CATEGORICAL_FEATURES
            for value in (features.get(name),)
        )

    @staticmethod
    def _feature_frame(
        features: Mapping[str, object], model: Tier1Model | Tier2Model
    ) -> pd.DataFrame:
        row: dict[str, object] = {}
        for name in model.FEATURES:
            value = features.get(name)
            if value is None and name in model.CATEGORICAL_FEATURES:
                value = "__UNKNOWN__"
            row[name] = np.nan if value is None else value
        return pd.DataFrame([row], columns=model.FEATURES)


class TieredPredictorCache:
    """Reload the signed active bundle only when its artifact changes on disk."""

    def __init__(self, store: TieredModelArtifactStore | None = None) -> None:
        self.store = store or TieredModelArtifactStore()
        self._predictor: Predictor | None = None
        self._mtime_ns: int | None = None
        self._lock = threading.Lock()

    def get(self) -> Predictor:
        try:
            mtime_ns = self.store.active_path.stat().st_mtime_ns
        except OSError as exc:
            raise TieredArtifactIntegrityError(
                "No active tiered model artifact exists"
            ) from exc
        if self._predictor is not None and self._mtime_ns == mtime_ns:
            return self._predictor
        with self._lock:
            if self._predictor is None or self._mtime_ns != mtime_ns:
                self._predictor = Predictor.from_active_artifact(self.store)
                self._mtime_ns = mtime_ns
        return self._predictor


tiered_predictor_cache = TieredPredictorCache()


def get_active_tiered_predictor() -> Predictor:
    """Dependency-friendly access point for the current signed tiered bundle."""
    return tiered_predictor_cache.get()
