"""PostgreSQL/Redis integration gate.

The full unit suite runs on SQLite so it never proves that the production
stack actually round-trips against real PostgreSQL and Redis.  This module
is the CI gate for that gap.  Run it explicitly:

    RUN_INTEGRATION=1 pytest tests/integration -q

It is skipped by default in the local fast suite.
"""

from __future__ import annotations

import os
import uuid

import pytest
import redis as redis_lib
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Role
from app.db.session import initialize_database
from app.db.user_repository import UserRepository

pytestmark = pytest.mark.integration

_requires_integration = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to exercise live PostgreSQL and Redis",
)


@_requires_integration
def test_postgres_repository_round_trips() -> None:
    database_url = os.environ.get("INTEGRATION_DATABASE_URL") or os.environ.get(
        "DATABASE_URL"
    )
    assert database_url and database_url.startswith(
        "postgresql"
    ), "integration suite requires a postgresql INTEGRATION_DATABASE_URL"
    schema = f"integration_{uuid.uuid4().hex[:10]}"
    engine, status = initialize_database(database_url, allow_fallback=False)
    assert status["backend"] == "postgresql"
    assert status["fallback_active"] is False
    try:
        with engine.connect() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            connection.execute(text(f'SET search_path TO "{schema}", public'))
        isolated = create_engine(
            database_url,
            connect_args={
                "connect_timeout": 3,
                "options": f"-csearch_path={schema},public",
            },
        )
        Base.metadata.create_all(isolated)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=isolated)
        with session_factory() as db:
            admin_role = Role(id=1, name="admin", description="integration")
            operator_role = Role(id=2, name="operator", description="integration")
            db.add_all([admin_role, operator_role])
            db.commit()

            repo = UserRepository(db)
            username = f"int_{uuid.uuid4().hex[:8]}"
            repo = UserRepository(db)
            repo.create_user(
                username=username,
                email=f"{username}@example.com",
                password_hash="$2b$12$integrationhashtest0000000000000000000000000",
                role_names=["admin"],
            )
            fetched = repo.get_by_identifier(username)
            assert fetched is not None
            assert fetched.email == f"{username}@example.com"
            assert repo.role_names(fetched) == ["admin"]
            assert (
                repo.permission_codes(fetched) == ["admin"]
                or repo.permission_codes(fetched) == []
            )
    finally:
        with engine.connect() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        engine.dispose()


@_requires_integration
def test_redis_round_trips() -> None:
    redis_url = os.environ.get("INTEGRATION_REDIS_URL") or os.environ.get(
        "REDIS_URL", "redis://localhost:6379/0"
    )
    client = redis_lib.Redis.from_url(
        redis_url, socket_connect_timeout=3, decode_responses=True
    )
    key = f"integration:{uuid.uuid4().hex}"
    try:
        assert client.ping() is True
        assert client.set(key, "gate-open", ex=60) is True
        assert client.get(key) == "gate-open"
        assert client.incr(f"{key}:counter", 3) == 3
        assert client.incr(f"{key}:counter") == 4
    finally:
        client.delete(key, f"{key}:counter")
        client.close()
