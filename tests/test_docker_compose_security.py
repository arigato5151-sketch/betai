from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
COMPOSE_TEXT = COMPOSE_PATH.read_text(encoding="utf-8")
COMPOSE = yaml.safe_load(COMPOSE_TEXT)


def _env(service: str) -> dict[str, str]:
    raw = COMPOSE["services"][service].get("environment", {})
    return {k: str(v) for k, v in raw.items()}


def test_database_password_is_not_hardcoded() -> None:
    assert "betai:betai@" not in COMPOSE_TEXT
    assert "betai:betai" not in COMPOSE_TEXT


def test_postgres_password_is_required() -> None:
    assert _env("postgres")["POSTGRES_PASSWORD"] == (
        "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
    )


def test_all_database_urls_use_the_password_variable() -> None:
    for service in ("migration", "bootstrap-admin", "backend", "worker", "beat"):
        url = _env(service)["DATABASE_URL"]
        assert "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}" in url, service
        assert "betai:betai" not in url


def test_api_football_key_is_overridable_everywhere() -> None:
    for service in ("migration", "bootstrap-admin", "backend", "worker", "beat"):
        value = _env(service)["API_FOOTBALL_KEY"]
        assert value == "${API_FOOTBALL_KEY:-DEMO_KEY}", service


def test_app_services_require_explicit_environment_contract() -> None:
    for service in ("backend", "worker", "beat"):
        assert _env(service)["ENVIRONMENT"] == "${ENVIRONMENT:?ENVIRONMENT is required}"


def test_production_security_values_are_required() -> None:
    for service in ("backend", "worker", "beat"):
        env = _env(service)
        assert env["JWT_SECRET_KEY"] == "${JWT_SECRET_KEY:?JWT_SECRET_KEY is required}"
        assert env["JWT_REFRESH_SECRET_KEY"] == (
            "${JWT_REFRESH_SECRET_KEY:?JWT_REFRESH_SECRET_KEY is required}"
        )
        assert env["MODEL_SIGNING_KEY"] == (
            "${MODEL_SIGNING_KEY:?MODEL_SIGNING_KEY is required}"
        )
        assert env["COOKIE_SECURE"] == "${COOKIE_SECURE:?COOKIE_SECURE is required}"


def test_no_literal_development_secret_committed_as_secret_value() -> None:
    lines = [
        line.strip()
        for line in COMPOSE_TEXT.splitlines()
        if "SECRET" in line and "PASSWORD" not in line
    ]
    for line in lines:
        value = line.split(":", 1)[1].strip()
        assert value.startswith("${") or value == "", line


def test_redis_is_private_and_password_protected() -> None:
    redis_service = COMPOSE["services"]["redis"]

    assert "ports" not in redis_service
    assert redis_service["expose"] == ["6379"]
    assert _env("redis")["REDIS_PASSWORD"].startswith("${REDIS_PASSWORD:?")
    assert "--requirepass" in " ".join(redis_service["command"])


def test_app_services_use_authenticated_redis_urls() -> None:
    for service in ("backend", "worker", "beat"):
        redis_url = _env(service)["REDIS_URL"]
        assert redis_url.startswith("redis://:${REDIS_PASSWORD:?")
        assert redis_url.endswith("@redis:6379/0")
