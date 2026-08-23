from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
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
        if name != "league_id"
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
        if name != "league_id"
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
        "decision_use": "research_only",
        "used_tier": "Tier 1",
        "confidence_scores": {"0": 0.15, "1": 0.25, "2": 0.6},
        "confidence": 0.6,
        "decision_status": "eligible",
        "decision_reasons": [],
        "confidence_tier": "high",
        "uncertainty": {
            "status": "eligible",
            "reasons": [],
            "confidence_tier": "high",
            "top_probability_pct": 60.0,
            "probability_margin_pct": 35.0,
            "normalized_entropy": 0.853474,
            "max_source_js_divergence": 0.0,
            "source_count": 0,
            "market_validation": {
                "present": False,
                "edge_pct": None,
                "implied_pct": None,
                "min_edge_pct": None,
                "passed": None,
            },
        },
        "artifact_version": None,
        "research_only": True,
        "tier2_gate": None,
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
    assert body["decision_use"] == "research_only"
    assert body["artifact_version"] == predictor.artifact_version
    assert body["confidence_scores"]["2"] == 0.6


def test_endpoint_rejects_partial_or_unknown_research_features() -> None:
    predictor = Predictor(
        StubTier1Model(), StubTier2Model(), tier1_league_ids=frozenset({39})
    )
    client = _make_client(predictor)

    partial = client.post(
        "/predict/tiered",
        json={"league_id": 39, "features": {"home_team": "Home"}},
    )
    unknown = client.post(
        "/predict/tiered",
        json={
            "league_id": 39,
            "features": {**_tier2_features(), "leaked_target": 1.0},
        },
    )

    assert partial.status_code == 422
    assert unknown.status_code == 422


def test_endpoint_rejects_nested_league_override() -> None:
    predictor = Predictor(
        StubTier1Model(), StubTier2Model(), tier1_league_ids=frozenset({39})
    )

    response = _make_client(predictor).post(
        "/predict/tiered",
        json={
            "league_id": 39,
            "features": {**_tier2_features(), "league_id": 2},
        },
    )

    assert response.status_code == 422


def test_endpoint_marks_market_inferior_tier2_as_research_only() -> None:
    predictor = Predictor(
        StubTier1Model(),
        StubTier2Model(),
        tier1_league_ids=frozenset({39}),
        tier2_gate={
            "passed": False,
            "reasons": ["insufficient_tier2_samples", "tier2_class_imbalance"],
            "training_samples": 14,
            "minimum_samples": 500,
            "min_per_class": 3,
            "minimum_per_class": 20,
        },
    )

    response = _make_client(predictor).post(
        "/predict/tiered",
        json={"league_id": 39, "features": _tier2_features()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["used_tier"] == "Tier 2"
    assert body["research_only"] is True
    assert body["tier2_gate"] == {
        "passed": False,
        "reasons": ["insufficient_tier2_samples", "tier2_class_imbalance"],
        "training_samples": 14,
        "minimum_samples": 500,
        "min_per_class": 3,
        "minimum_per_class": 20,
    }


class ConditionalTier1Model(Tier1Model):
    def __init__(self) -> None:
        super().__init__(backend="sklearn")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([[0.32, 0.33, 0.35]])


def test_endpoint_returns_conditional_for_marginal_tier1() -> None:
    tier1 = ConditionalTier1Model()
    tier2 = StubTier2Model()
    predictor = Predictor(tier1, tier2, tier1_league_ids=frozenset({39}))

    response = _make_client(predictor).post(
        "/predict/tiered",
        json={"league_id": 39, "features": _tier1_features()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision_status"] == "conditional"
    assert body["confidence_tier"] == "conditional"
    assert body["decision_use"] == "research_only"
    assert body["used_tier"] == "Tier 1"


def test_tier2_gate_passed_enables_conditional_band() -> None:
    tier1 = StubTier1Model()
    tier2 = StubTier2Model()
    predictor = Predictor(
        tier1,
        tier2,
        tier1_league_ids=frozenset({39}),
        tier2_gate={"passed": True, "reasons": []},
    )

    response = _make_client(predictor).post(
        "/predict/tiered",
        json={"league_id": 39, "features": _tier2_features()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["used_tier"] == "Tier 2"
    assert body["research_only"] is False
    assert body["decision_status"] == "eligible"


@pytest.mark.asyncio
async def test_batch_endpoint_builds_fixture_specific_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_fixture_ids: list[int] = []

    async def get_or_create(fixture_id: int, **_kwargs: object) -> dict[str, object]:
        loaded_fixture_ids.append(fixture_id)
        return {"fixture_id": fixture_id}

    def build_payload(prefill: dict[str, object]) -> object:
        return prefill

    async def compute(payload: dict[str, object], **_kwargs: object) -> dict[str, object]:
        fixture_id = int(payload["fixture_id"])
        return {
            "analysis": {
                "prediction": "HOME_WIN" if fixture_id == 42 else "AWAY_WIN",
                "all_probabilities": {
                    "HOME_WIN": 0.6 if fixture_id == 42 else 0.2,
                    "DRAW": 0.2,
                    "AWAY_WIN": 0.2 if fixture_id == 42 else 0.6,
                },
                "model": "test-model",
            },
            "ml_result": {"ready": fixture_id == 42},
        }

    monkeypatch.setattr(endpoints.fixture_context_service, "get_or_create", get_or_create)
    monkeypatch.setattr(endpoints, "_build_payload_from_prefill", build_payload)
    monkeypatch.setattr(endpoints, "_compute_analysis", compute)
    result = await endpoints.batch_predict(
        fixture_ids=[42],
        user=type("User", (), {"id": "batch-test-user"})(),
    )

    assert result == {
        "predictions": [
            {
                "fixture_id": 42,
                "status": "ok",
                "prediction": "HOME_WIN",
                "probabilities": {
                    "AWAY_WIN": 0.2,
                    "DRAW": 0.2,
                    "HOME_WIN": 0.6,
                },
                "model": "test-model",
                "ml_ready": True,
            }
        ],
        "count": 1,
    }
    assert loaded_fixture_ids == [42]
