from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.external_data_repository import ExternalDataRepository
from app.db.session import SessionLocal
from app.providers.sportmonks import SportmonksClient, SportmonksError

logger = logging.getLogger(__name__)


class SportmonksPlayerService:
    """Provide isolated player IDs when primary player coverage is incomplete."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
        client: SportmonksClient | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.client = client or SportmonksClient()

    async def get_team_player_ratings(
        self,
        *,
        canonical_team_id: int,
        canonical_team_name: str,
        as_of: datetime,
        canonical_source: str = "api_football",
    ) -> dict[int, dict[str, float]]:
        if (
            not settings.SPORTMONKS_ENABLED
            or not self.client.configured
            or canonical_team_id <= 0
            or not canonical_team_name.strip()
        ):
            return {}
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            as_of = as_of.replace(tzinfo=UTC)

        provider_team_key: str | None = None
        try:
            with self.session_factory() as db:
                mapping = ExternalDataRepository(db).get_mapping(
                    canonical_source=canonical_source,
                    canonical_team_id=canonical_team_id,
                    provider="sportmonks",
                )
                if mapping is not None:
                    provider_team_key = mapping.provider_team_key
        except (SQLAlchemyError, ValueError) as exc:
            logger.warning(
                "Sportmonks team mapping lookup failed",
                extra={
                    "team_id": canonical_team_id,
                    "error_type": type(exc).__name__,
                },
            )

        try:
            resolved = await self.client.get_recent_player_ratings(
                team_name=canonical_team_name,
                as_of=as_of,
                provider_team_key=provider_team_key,
            )
        except SportmonksError as exc:
            logger.warning(
                "Sportmonks player fallback unavailable",
                extra={
                    "team_id": canonical_team_id,
                    "error_type": type(exc).__name__,
                },
            )
            return {}
        if resolved is None:
            return {}

        candidate, ratings = resolved
        try:
            with self.session_factory() as db:
                ExternalDataRepository(db).upsert_mapping(
                    canonical_source=canonical_source,
                    canonical_team_id=canonical_team_id,
                    canonical_team_name=canonical_team_name,
                    provider="sportmonks",
                    candidate=candidate,
                    verified=provider_team_key is not None,
                )
        except (SQLAlchemyError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Sportmonks team mapping could not be cached",
                extra={
                    "team_id": canonical_team_id,
                    "error_type": type(exc).__name__,
                },
            )
        return ratings


sportmonks_player_service = SportmonksPlayerService()
