from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import endpoints
from app.prediction.ml.model_router import RoutedPrediction


class StubPredictor:
    def predict(self, features: dict[str, object]) -> RoutedPrediction:
        assert features["league_id"] == 39
        return RoutedPrediction("tier1", (0.15, 0.25, 0.60), "test-version")


def test_tiered_prediction_endpoint_returns_selected_tier_and_scores(
    monkeypatch,
) -> None:
    app = FastAPI()
    app.include_router(endpoints.router)
    route = next(
        route
        for route in endpoints.router.routes
        if getattr(route, "path", None) == "/predict/tiered"
    )
    permission_dependency = route.dependant.dependencies[0].call
    app.dependency_overrides[permission_dependency] = lambda: None
    monkeypatch.setattr(
        endpoints,
        "get_active_tiered_predictor",
        lambda: StubPredictor(),
    )

    response = TestClient(app).post(
        "/predict/tiered",
        json={"league_id": 39, "features": {"home_avg_shots": 12.0}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "used_tier": "Tier 1",
        "confidence_scores": {"0": 0.15, "1": 0.25, "2": 0.6},
        "confidence": 0.6,
        "artifact_version": "test-version",
    }
