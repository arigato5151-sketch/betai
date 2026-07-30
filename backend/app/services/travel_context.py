from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.player_context_repository import (
    PlayerContextRepository,
    haversine_distance_km,
)
from app.db.session import SessionLocal
from app.providers.base import ExternalDataPoint
from app.providers.geonames_city import GeoNamesCityResolver

logger = logging.getLogger(__name__)


class VenueContextClient(Protocol):
    async def get_team_venue_context(
        self,
        team_id: int,
    ) -> dict[str, Any] | None: ...


class TravelContextService:
    """Resolve missing team locations without replacing curated coordinates."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
        resolver: GeoNamesCityResolver | None = None,
    ) -> None:
        self.session_factory = session_factory
        self._resolver = resolver

    @property
    def resolver(self) -> GeoNamesCityResolver:
        """Load the relatively large offline city index only when it is needed."""
        if self._resolver is None:
            self._resolver = GeoNamesCityResolver()
        return self._resolver

    async def get_away_travel_distance(
        self,
        *,
        home_team_id: int,
        away_team_id: int,
        home_team_name: str,
        away_team_name: str,
        client: VenueContextClient,
    ) -> ExternalDataPoint | None:
        if home_team_id <= 0 or away_team_id <= 0 or home_team_id == away_team_id:
            return None

        existing = self._locations(home_team_id, away_team_id)
        if existing is None:
            return None
        home_location, away_location = existing
        missing_sides: list[str] = []
        requests = []
        if home_location is None:
            missing_sides.append("home")
            requests.append(client.get_team_venue_context(home_team_id))
        if away_location is None:
            missing_sides.append("away")
            requests.append(client.get_team_venue_context(away_team_id))

        resolved_rows: list[dict[str, object]] = []
        if requests:
            contexts = await asyncio.gather(*requests)
            for side, context in zip(missing_sides, contexts, strict=True):
                if not isinstance(context, dict):
                    continue
                resolved = self.resolver.resolve(
                    city=str(context.get("city") or ""),
                    country=str(context.get("country") or ""),
                )
                if resolved is None:
                    continue
                team_id = home_team_id if side == "home" else away_team_id
                team_name = home_team_name if side == "home" else away_team_name
                resolved_rows.append(
                    {
                        "data_source": "api_football",
                        "team_id": team_id,
                        "name": team_name,
                        "latitude": resolved.latitude,
                        "longitude": resolved.longitude,
                        "location_source": "geonames_city",
                        "confidence": resolved.confidence,
                        "details": {
                            "city": resolved.city,
                            "provider_city": context.get("city"),
                            "country_code": resolved.country_code,
                            "geoname_id": resolved.geoname_id,
                            "venue_id": context.get("venue_id"),
                            "venue_name": context.get("venue_name"),
                            "venue_metadata_source": context.get("source"),
                            "approximation": "city_centre",
                        },
                    }
                )
        if resolved_rows:
            self._persist_if_missing(resolved_rows)

        locations = self._locations(home_team_id, away_team_id)
        if locations is None:
            return None
        home_location, away_location = locations
        if (
            home_location is None
            or away_location is None
            or home_location.latitude is None
            or home_location.longitude is None
            or away_location.latitude is None
            or away_location.longitude is None
        ):
            return None

        distance = haversine_distance_km(
            away_location.latitude,
            away_location.longitude,
            home_location.latitude,
            home_location.longitude,
        )
        observed_at = datetime.now(UTC)
        confidence = min(home_location.confidence, away_location.confidence)
        sources = {
            home_location.location_source,
            away_location.location_source,
        }
        return ExternalDataPoint(
            value=distance,
            source=(
                "geonames_city"
                if "geonames_city" in sources
                else "curated_team_locations"
            ),
            captured_at=observed_at,
            confidence=confidence,
            is_fallback="geonames_city" in sources,
            details={
                "origin_team_id": away_team_id,
                "destination_team_id": home_team_id,
                "location_sources": sorted(sources),
                "method": "haversine",
            },
        )

    def _locations(
        self, home_team_id: int, away_team_id: int
    ) -> tuple[Any, Any] | None:
        try:
            with self.session_factory() as db:
                repository = PlayerContextRepository(db)
                home = repository.get_team_location(home_team_id)
                away = repository.get_team_location(away_team_id)
                if home is not None:
                    db.expunge(home)
                if away is not None:
                    db.expunge(away)
                return home, away
        except (SQLAlchemyError, ValueError) as exc:
            logger.warning(
                "Team location lookup failed",
                extra={"error_type": type(exc).__name__},
            )
            return None

    def _persist_if_missing(self, rows: list[dict[str, object]]) -> None:
        try:
            with self.session_factory() as db:
                repository = PlayerContextRepository(db)
                missing_rows: list[dict[str, object]] = []
                for row in rows:
                    team_id = row.get("team_id")
                    if (
                        isinstance(team_id, int)
                        and not isinstance(team_id, bool)
                        and repository.get_team_location(team_id) is None
                    ):
                        missing_rows.append(row)
                if missing_rows:
                    repository.upsert_team_locations(missing_rows)
        except (SQLAlchemyError, ValueError) as exc:
            logger.warning(
                "Resolved team locations could not be persisted",
                extra={"error_type": type(exc).__name__},
            )


travel_context_service = TravelContextService()
