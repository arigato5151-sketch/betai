from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.api_mode import get_api_mode
from app.main import app


def test_known_placeholder_keys_use_demo_mode() -> None:
    for api_key in (None, "", " DEMO_KEY ", "your_api_football_key"):
        assert get_api_mode(api_key) == "demo"


def test_configured_key_uses_live_mode() -> None:
    assert get_api_mode("production-api-key") == "live"


def test_public_status_endpoint_reports_current_mode() -> None:
    response = TestClient(app).get("/api/status")

    assert response.status_code == 200
    assert response.json() == {
        "api_mode": get_api_mode(settings.API_FOOTBALL_KEY),
        "registration_enabled": settings.ALLOW_SELF_REGISTRATION,
    }
