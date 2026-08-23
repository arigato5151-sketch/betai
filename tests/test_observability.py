from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_health_and_request_observability_endpoints() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live", headers={"X-Request-ID": "test-request"})
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request"
    assert response.json() == {"status": "ok"}
    assert metrics.status_code == 200
    assert "bet_ai_http_requests_total" in metrics.text
    assert 'path="/health/live"' in metrics.text


def test_production_metrics_require_a_bearer_token(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "METRICS_TOKEN", "m" * 32)

    with TestClient(app) as client:
        unauthorized = client.get("/metrics")
        authorized = client.get(
            "/metrics",
            headers={"Authorization": f"Bearer {'m' * 32}"},
        )

    assert unauthorized.status_code == 404
    assert authorized.status_code == 200
