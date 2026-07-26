import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.core.security import OriginValidationMiddleware


def production_settings(**overrides) -> Settings:
    values = {
        "ENVIRONMENT": "production",
        "DEBUG": False,
        "API_FOOTBALL_KEY": "live-api-key",
        "DATABASE_URL": "postgresql://user:password@db:5432/bet_ai",
        "ALLOW_DATABASE_FALLBACK": False,
        "JWT_SECRET_KEY": "a" * 32,
        "JWT_REFRESH_SECRET_KEY": "b" * 32,
        "MODEL_SIGNING_KEY": "c" * 32,
        "COOKIE_SECURE": True,
        "REQUIRE_ORIGIN_HEADER": True,
        "ADMIN_PASSWORD": "strong-admin-password",
        "FRONTEND_URL": "https://bets.example.com",
        "BACKEND_CORS_ORIGINS": "https://bets.example.com",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_valid_production_security_configuration() -> None:
    settings = production_settings()

    assert settings.BACKEND_CORS_ORIGINS == ["https://bets.example.com"]
    assert settings.REQUIRE_ORIGIN_HEADER is True


@pytest.mark.parametrize(
    ("override", "expected_message"),
    [
        ({"DEBUG": True}, "DEBUG must be false"),
        ({"COOKIE_SECURE": False}, "COOKIE_SECURE must be true"),
        ({"JWT_SECRET_KEY": "weak"}, "JWT secrets must be distinct"),
        ({"MODEL_SIGNING_KEY": "weak"}, "MODEL_SIGNING_KEY must be non-default"),
        (
            {"BACKEND_CORS_ORIGINS": "http://bets.example.com"},
            "production CORS origins must use HTTPS",
        ),
        ({"DATABASE_URL": "sqlite:///local.db"}, "DATABASE_URL must use PostgreSQL"),
        ({"ALLOW_DATABASE_FALLBACK": True}, "ALLOW_DATABASE_FALLBACK must be false"),
    ],
)
def test_insecure_production_configuration_fails_fast(
    override: dict[str, object], expected_message: str
) -> None:
    with pytest.raises(ValidationError, match=expected_message):
        production_settings(**override)


def test_samesite_none_requires_secure_cookie_in_every_environment() -> None:
    with pytest.raises(ValidationError, match="SameSite=None cookies require"):
        Settings(_env_file=None, COOKIE_SAMESITE="none", COOKIE_SECURE=False)


def test_cors_origins_accept_comma_separated_and_json_values() -> None:
    comma = Settings(
        _env_file=None,
        BACKEND_CORS_ORIGINS="http://localhost:3000,http://localhost:5173",
    )
    json_value = Settings(
        _env_file=None,
        BACKEND_CORS_ORIGINS='["http://localhost:3000", "http://localhost:5173"]',
    )

    assert comma.BACKEND_CORS_ORIGINS == json_value.BACKEND_CORS_ORIGINS


def make_origin_client(require_origin_header: bool = True) -> TestClient:
    app = FastAPI()
    app.add_middleware(
        OriginValidationMiddleware,
        allowed_origins=["https://bets.example.com"],
        require_origin_header=require_origin_header,
        access_cookie_name="bet_ai_access",
        refresh_cookie_name="bet_ai_refresh",
        csrf_cookie_name="bet_ai_csrf",
        csrf_header_name="X-CSRF-Token",
    )

    @app.post("/change")
    def change_state() -> dict[str, bool]:
        return {"changed": True}

    return TestClient(app)


def test_origin_middleware_allows_trusted_origin() -> None:
    response = make_origin_client().post(
        "/change", headers={"Origin": "https://bets.example.com"}
    )

    assert response.status_code == 200


def test_origin_middleware_rejects_untrusted_and_missing_origin() -> None:
    client = make_origin_client()

    assert (
        client.post("/change", headers={"Origin": "https://evil.example"}).status_code
        == 403
    )
    assert client.post("/change").status_code == 403


def test_origin_header_is_optional_outside_production() -> None:
    response = make_origin_client(require_origin_header=False).post("/change")

    assert response.status_code == 200


def test_csrf_double_submit_is_required_for_authenticated_mutations() -> None:
    client = make_origin_client(require_origin_header=False)
    client.cookies.set("bet_ai_access", "access")
    client.cookies.set("bet_ai_csrf", "csrf-token")

    missing = client.post("/change")
    valid = client.post("/change", headers={"X-CSRF-Token": "csrf-token"})

    assert missing.status_code == 403
    assert valid.status_code == 200
