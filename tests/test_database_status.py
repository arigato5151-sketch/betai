from unittest.mock import Mock

import pytest
from sqlalchemy.exc import OperationalError

from app.db import session


def test_configured_sqlite_is_not_reported_as_fallback() -> None:
    engine, status = session.initialize_database("sqlite:///:memory:")

    assert engine.dialect.name == "sqlite"
    assert status == {
        "backend": "sqlite",
        "fallback_active": False,
        "status": "ready",
        "fallback_reason": None,
    }


def test_postgres_failure_is_reported_as_degraded(monkeypatch) -> None:
    failed_engine = Mock()
    failed_engine.connect.side_effect = OperationalError("connect", {}, Exception())
    real_create_engine = session.create_engine

    def fake_create_engine(url, *args, **kwargs):
        if str(url).startswith("postgresql"):
            return failed_engine
        return real_create_engine(url, *args, **kwargs)

    monkeypatch.setattr(session, "create_engine", fake_create_engine)

    engine, status = session.initialize_database(
        "postgresql://user:password@localhost/test"
    )

    assert engine.dialect.name == "sqlite"
    assert status["status"] == "degraded"
    assert status["fallback_active"] is True
    assert status["backend"] == "sqlite"
    assert status["fallback_reason"] == "postgresql_connection_failed"


def test_postgres_failure_is_fatal_when_fallback_is_disabled(monkeypatch) -> None:
    failed_engine = Mock()
    failed_engine.connect.side_effect = OperationalError("connect", {}, Exception())
    monkeypatch.setattr(session, "create_engine", Mock(return_value=failed_engine))

    with pytest.raises(
        session.DatabaseInitializationError, match="fallback is disabled"
    ):
        session.initialize_database(
            "postgresql://user:password@localhost/test",
            allow_fallback=False,
        )

    failed_engine.dispose.assert_called_once_with()


def test_unsupported_database_scheme_is_fatal_when_fallback_is_disabled() -> None:
    with pytest.raises(session.DatabaseInitializationError, match="Unsupported"):
        session.initialize_database("mysql://localhost/test", allow_fallback=False)
