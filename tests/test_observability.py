from fastapi.testclient import TestClient

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
