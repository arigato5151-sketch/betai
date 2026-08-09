from fastapi import Response, status

from app import main


def test_readiness_rejects_degraded_database(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_database_status",
        lambda: {"status": "degraded", "fallback_active": True},
    )
    monkeypatch.setattr(main.cache, "status", lambda: {"status": "ready"})
    response = Response()

    body = main.readiness(response)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert body["status"] == "not_ready"


def test_production_readiness_requires_distributed_cache(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_database_status",
        lambda: {"status": "ready", "fallback_active": False},
    )
    monkeypatch.setattr(main.cache, "status", lambda: {"status": "degraded"})
    monkeypatch.setattr(main.settings, "ENVIRONMENT", "production")
    response = Response()

    body = main.readiness(response)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert body == {
        "status": "not_ready",
        "database": "ready",
        "cache": "degraded",
    }


def test_development_readiness_allows_local_cache(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_database_status",
        lambda: {"status": "ready", "fallback_active": False},
    )
    monkeypatch.setattr(main.cache, "status", lambda: {"status": "degraded"})
    monkeypatch.setattr(main.settings, "ENVIRONMENT", "development")
    response = Response()

    body = main.readiness(response)

    assert response.status_code == status.HTTP_200_OK
    assert body["status"] == "ready"
