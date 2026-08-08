from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import endpoints
from app.prediction.ml.ml_pipeline import Tier1Model, Tier2Model
from app.prediction.ml.model_router import (
    Predictor,
    TieredModelArtifactStore,
)


class StubTier1Model(Tier1Model):
    def __init__(self) -> None:
        super().__init__(backend="sklearn")
        self.calls = 0

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        return np.array([[0.15, 0.25, 0.6]])


class StubTier2Model(Tier2Model):
    def __init__(self) -> None:
        super().__init__(backend="sklearn")
        self.calls = 0

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        return np.array([[0.55, 0.25, 0.2]])


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
            "39"
            if name == "league_id"
            else "Team" if name in model.CATEGORICAL_FEATURES else 1.5
        )
        for name in model.FEATURES
    }


def _permission_dependency(app: FastAPI):
    route = next(
        route
        for route in endpoints.router.routes
        if getattr(route, "path", None) == "/predict/tiered"
    )
    permission_dependency = route.dependant.dependencies[0].call
    app.dependency_overrides[permission_dependency] = lambda: None
    return app


def _make_client(predictor: Predictor):
    app = FastAPI()
    app.include_router(endpoints.router)
    _permission_dependency(app)
    app.dependency_overrides[endpoints.get_tiered_predictor] = lambda: predictor
    return TestClient(app)


def test_endpoint_routes_data_rich_league_to_tier1() -> None:
    tier1 = StubTier1Model()
    tier2 = StubTier2Model()
    predictor = Predictor(tier1, tier2, tier1_league_ids=frozenset({39}))

    response = _make_client(predictor).post(
        "/predict/tiered",
        json={"league_id": 39, "features": _tier1_features()},
    )

    assert response.status_code == 200
    assert response.json() == {
        "used_tier": "Tier 1",
        "confidence_scores": {"0": 0.15, "1": 0.25, "2": 0.6},
        "confidence": 0.6,
        "artifact_version": None,
    }
    assert tier1.calls == 1
    assert tier2.calls == 0


def test_endpoint_falls_back_to_tier2_when_rich_features_missing() -> None:
    tier1 = StubTier1Model()
    tier2 = StubTier2Model()
    predictor = Predictor(tier1, tier2, tier1_league_ids=frozenset({39}))

    response = _make_client(predictor).post(
        "/predict/tiered",
        json={"league_id": 39, "features": _tier2_features()},
    )

    assert response.status_code == 200
    assert response.json()["used_tier"] == "Tier 2"
    assert response.json()["confidence_scores"] == {"0": 0.55, "1": 0.25, "2": 0.2}
    assert tier1.calls == 0
    assert tier2.calls == 1


def test_endpoint_reads_signed_artifact_and_predicts_tier1(tmp_path) -> None:
    store = TieredModelArtifactStore(artifacts_dir=tmp_path)
    store.export(
        StubTier1Model(),
        StubTier2Model(),
        tier1_metrics={"accuracy": 0.62},
        tier2_metrics={"accuracy": 0.55},
    )
    predictor = Predictor.from_active_artifact(store)

    response = _make_client(predictor).post(
        "/predict/tiered",
        json={"league_id": 39, "features": _tier1_features()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["used_tier"] == "Tier 1"
    assert body["artifact_version"] == predictor.artifact_version
    assert body["confidence_scores"]["2"] == 0.6
