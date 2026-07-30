from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.external_data_repository import ExternalDataRepository
from app.db.session import SessionLocal
from app.providers.base import ExternalDataPoint
from app.providers.clubelo import ClubEloClient, ClubEloError

logger = logging.getLogger(__name__)


class ExternalFeatureService:
    """Resolve lower-priority external features without weakening primary data."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
        clubelo: ClubEloClient | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.clubelo = clubelo or ClubEloClient()

    async def get_team_elo(
        self,
        *,
        canonical_team_id: int,
        canonical_team_name: str,
        as_of: datetime,
        canonical_source: str = "api_football",
    ) -> ExternalDataPoint | None:
        if (
            not settings.CLUBELO_ENABLED
            or canonical_team_id <= 0
            or not canonical_team_name.strip()
        ):
            return None
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            as_of = as_of.replace(tzinfo=UTC)

        provider_team_key: str | None = None
        try:
            with self.session_factory() as db:
                repository = ExternalDataRepository(db)
                cached = repository.get_latest_snapshot(
                    canonical_source=canonical_source,
                    canonical_team_id=canonical_team_id,
                    feature_name="elo",
                    provider="clubelo",
                    at=datetime.now(UTC),
                )
                if cached is not None:
                    return repository.to_data_point(cached)
                mapping = repository.get_mapping(
                    canonical_source=canonical_source,
                    canonical_team_id=canonical_team_id,
                    provider="clubelo",
                )
                if mapping is not None:
                    provider_team_key = mapping.provider_team_key
        except (SQLAlchemyError, ValueError) as exc:
            logger.warning(
                "External Elo cache lookup failed",
                extra={
                    "team_id": canonical_team_id,
                    "error_type": type(exc).__name__,
                },
            )

        try:
            resolved = await self.clubelo.resolve_elo(
                team_name=canonical_team_name,
                as_of=as_of,
                provider_team_key=provider_team_key,
            )
        except ClubEloError as exc:
            logger.warning(
                "ClubElo fallback unavailable",
                extra={
                    "team_id": canonical_team_id,
                    "error_type": type(exc).__name__,
                },
            )
            return None
        if resolved is None:
            return None

        candidate, point = resolved
        try:
            with self.session_factory() as db:
                repository = ExternalDataRepository(db)
                repository.upsert_mapping(
                    canonical_source=canonical_source,
                    canonical_team_id=canonical_team_id,
                    canonical_team_name=canonical_team_name,
                    provider="clubelo",
                    candidate=candidate,
                    # Exact normalized matching is recorded but remains reviewable.
                    verified=provider_team_key is not None,
                )
                repository.save_snapshot(
                    canonical_source=canonical_source,
                    canonical_team_id=canonical_team_id,
                    feature_name="elo",
                    point=point,
                )
        except (SQLAlchemyError, RuntimeError, ValueError) as exc:
            # A transient persistence error must not discard a validated observation.
            logger.warning(
                "ClubElo observation could not be cached",
                extra={
                    "team_id": canonical_team_id,
                    "error_type": type(exc).__name__,
                },
            )
        return point


external_feature_service = ExternalFeatureService()
