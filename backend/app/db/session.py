from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.logging_config import logger


SQLITE_FALLBACK_URL = "sqlite:///./matches.db"


class DatabaseInitializationError(RuntimeError):
    """Raised when the configured database cannot be used safely."""


def _sqlite_engine(url: str) -> Engine:
    return create_engine(url, connect_args={"check_same_thread": False})


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

    if db_url.startswith("postgresql"):
        try:
            postgres_engine = create_engine(
                db_url,
                pool_size=20,
                max_overflow=10,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 3},
            )
            with postgres_engine.connect():
                pass
            logger.info("Connected successfully to PostgreSQL production database.")
            return postgres_engine, {
                "backend": "postgresql",
                "fallback_active": False,
                "status": "ready",
                "fallback_reason": None,
            }
        except Exception as exc:
            log_reason = f"{type(exc).__name__}: {exc}"
            if "postgres_engine" in locals():
                postgres_engine.dispose()
            if not allow_fallback:
                logger.error(
                    "PostgreSQL connection failed (%s). Database fallback is disabled.",
                    log_reason,
                )
                raise DatabaseInitializationError(
                    "PostgreSQL is unavailable and database fallback is disabled"
                ) from exc
            logger.warning(
                "PostgreSQL connection failed (%s). Falling back to local SQLite database.",
                log_reason,
            )
            return _sqlite_engine(SQLITE_FALLBACK_URL), {
                "backend": "sqlite",
                "fallback_active": True,
                "status": "degraded",
                "fallback_reason": "postgresql_connection_failed",
            }

    if not allow_fallback:
        logger.error("Unsupported DATABASE_URL scheme and fallback is disabled.")
        raise DatabaseInitializationError(
            "Unsupported DATABASE_URL scheme and database fallback is disabled"
        )
    logger.warning(
        "Unsupported DATABASE_URL scheme. Falling back to local SQLite database."
    )
    return _sqlite_engine(SQLITE_FALLBACK_URL), {
        "backend": "sqlite",
        "fallback_active": True,
        "status": "degraded",
        "fallback_reason": "unsupported_database_url_scheme",
    }


engine, database_status = initialize_database(
    settings.DATABASE_URL,
    allow_fallback=settings.ALLOW_DATABASE_FALLBACK,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_database_status() -> dict[str, Any]:
    return dict(database_status)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
