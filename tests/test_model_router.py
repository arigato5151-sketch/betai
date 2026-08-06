from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.prediction.ml.ml_pipeline import Tier1Model, Tier2Model
from app.prediction.ml.model_router import (
    Predictor,
    TieredArtifactIntegrityError,
    TieredModelArtifactStore,
)


class StubTier1Model(Tier1Model):
    def __init__(self) -> None:
        super().__init__(backend="sklearn")
        self.calls = 0

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        return np.array([[0.1, 0.2, 0.7]])


class StubTier2Model(Tier2Model):
    def __init__(self) -> None:
        super().__init__(backend="sklearn")
        self.calls = 0

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        return np.array([[0.5, 0.3, 0.2]])


def _tier1_features() -> dict[str, object]:
    model = Tier1Model(backend="sklearn")
    return {
        name: (
            "39"
            if name == "league_id"
            else "Team" if name in model.CATEGORICAL_FEATURES else 1.5
        )
        for name in model.FEATURES
    }


def _tier2_features() -> dict[str, object]:
    model = Tier2Model(backend="sklearn")
    return {
        name: (
            "2"
            if name == "league_id"
            else "Team" if name in model.CATEGORICAL_FEATURES else 1.5
        )
        for name in model.FEATURES
    }


def test_predictor_routes_data_rich_supported_league_to_tier1() -> None:
    tier1 = StubTier1Model()
    tier2 = StubTier2Model()

    prediction = Predictor(tier1, tier2, tier1_league_ids=frozenset({39})).predict(
        _tier1_features()
    )

    assert prediction.tier == "tier1"
    assert prediction.probabilities == (0.1, 0.2, 0.7)
    assert tier1.calls == 1
    assert tier2.calls == 0


def test_predictor_falls_back_to_tier2_when_rich_features_are_missing() -> None:
    tier1 = StubTier1Model()
    tier2 = StubTier2Model()
    features = _tier2_features() | {"league_id": "39"}

    prediction = Predictor(tier1, tier2, tier1_league_ids=frozenset({39})).predict(
        features
    )

    assert prediction.tier == "tier2"
    assert prediction.probabilities == (0.5, 0.3, 0.2)
    assert tier1.calls == 0
    assert tier2.calls == 1


def test_signed_bundle_export_load_and_router_smoke(tmp_path) -> None:
    tier1 = StubTier1Model()
    tier2 = StubTier2Model()
    store = TieredModelArtifactStore(artifacts_dir=tmp_path)

    bundle = store.export(
        tier1, tier2, tier1_metrics={"accuracy": 0.62}, tier2_metrics={"accuracy": 0.55}
    )
    loaded = store.load_active()
    prediction = Predictor.from_active_artifact(store).predict(_tier1_features())

    assert loaded is not None
    assert loaded.artifact_version == bundle.artifact_version
    assert store.verify(store.active_path) is True
    assert prediction.tier == "tier1"
    assert prediction.artifact_version == bundle.artifact_version


def test_tampered_signed_bundle_is_rejected(tmp_path) -> None:
    store = TieredModelArtifactStore(artifacts_dir=tmp_path)
    store.export(StubTier1Model(), StubTier2Model(), tier1_metrics={}, tier2_metrics={})
    with store.active_path.open("ab") as artifact:
        artifact.write(b"tampered")

    with pytest.raises(TieredArtifactIntegrityError, match="HMAC"):
        store.load_active()
