from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.logging_config import logger

SQLITE_FALLBACK_URL = "sqlite:///./matches.db"


class DatabaseInitializationError(RuntimeError):
    """Raised when the configured database cannot be used safely."""


def _sqlite_engine(url: str) -> Engine:
    from sqlalchemy import event

    engine = create_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine


def initialize_database(
    db_url: str, *, allow_fallback: bool = True
) -> tuple[Engine, dict[str, Any]]:
    if db_url.startswith("sqlite"):
        logger.info("Using configured SQLite database.")
        return _sqlite_engine(db_url), {
            "backend": "sqlite",
            "fallback_active": False,
            "status": "ready",
            "fallback_reason": None,
        }

    if not db_url.startswith("postgresql"):
        # A wrong scheme is a configuration error, never a transient outage.
        # Falling back to SQLite here silently relocates betting data to a
        # throwaway file. Fail fast instead.
        scheme = db_url.split("://", 1)[0]
        raise DatabaseInitializationError(
            f"Unsupported DATABASE_URL scheme '{scheme}': refusing to fall back "
            "to SQLite on misconfiguration"
        )

    try:
        postgres_engine = create_engine(
            db_url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3},
        )
    except Exception as exc:
        # URL/driver errors are permanent; surface them instead of masking.
        raise DatabaseInitializationError(
            f"PostgreSQL engine could not be created: {type(exc).__name__}"
        ) from exc

    try:
        with postgres_engine.connect():
            pass
        logger.info("Connected successfully to PostgreSQL production database.")
        return postgres_engine, {
            "backend": "postgresql",
            "fallback_active": False,
            "status": "ready",
            "fallback_reason": None,
        }
    except OperationalError as exc:
        # Only genuine, transient connection failures may fall back — and even
        # then only when the operator explicitly allowed it.
        postgres_engine.dispose()
        if not allow_fallback:
            raise DatabaseInitializationError(
                "PostgreSQL is unavailable and database fallback is disabled"
            ) from exc
        logger.critical(
            "PostgreSQL connection failed (%s: %s). Falling back to local SQLite "
            "database; data written here will NOT survive a restart.",
            type(exc).__name__,
            exc,
        )
        return _sqlite_engine(SQLITE_FALLBACK_URL), {
            "backend": "sqlite",
            "fallback_active": True,
            "status": "degraded",
            "fallback_reason": "postgresql_connection_failed",
        }
    except Exception as exc:
        # Any non-transient failure (auth, driver, programming error) is fatal.
        raise DatabaseInitializationError(
            f"PostgreSQL is unusable and cannot fall back: {type(exc).__name__}"
        ) from exc


engine, database_status = initialize_database(
    settings.DATABASE_URL,
    allow_fallback=settings.ALLOW_DATABASE_FALLBACK,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=True, bind=engine)


def get_database_status() -> dict[str, Any]:
    return dict(database_status)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
