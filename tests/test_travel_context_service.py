from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.db.player_context_repository import PlayerContextRepository
from app.providers.geonames_city import GeoNamesCityResolver
from app.services.travel_context import TravelContextService


class FakeVenueContextClient:
    def __init__(self, contexts: dict[int, dict[str, Any] | None]) -> None:
        self.contexts = contexts
        self.calls: list[int] = []

    async def get_team_venue_context(
        self,
        team_id: int,
    ) -> dict[str, Any] | None:
        self.calls.append(team_id)
        return self.contexts.get(team_id)


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.mark.asyncio
async def test_missing_locations_are_resolved_persisted_and_reused(
    session_factory: sessionmaker[Session],
) -> None:
    client = FakeVenueContextClient(
        {
            194: {
                "team_id": 194,
                "team_name": "Ajax",
                "country": "Netherlands",
                "city": "Amsterdam",
                "venue_id": 111,
                "venue_name": "Johan Cruijff Arena",
                "source": "api_football_teams",
            },
            702: {
                "team_id": 702,
                "team_name": "Vojvodina",
                "country": "Serbia",
                "city": "Novi Sad",
                "venue_id": 222,
                "venue_name": "Stadion Karađorđe",
                "source": "api_football_teams",
            },
        }
    )
    service = TravelContextService(
        session_factory=session_factory,
        resolver=GeoNamesCityResolver(),
    )

    first = await service.get_away_travel_distance(
        home_team_id=194,
        away_team_id=702,
        home_team_name="Ajax",
        away_team_name="Vojvodina",
        client=client,
    )
    second = await service.get_away_travel_distance(
        home_team_id=194,
        away_team_id=702,
        home_team_name="Ajax",
        away_team_name="Vojvodina",
        client=client,
    )

    assert first is not None
    assert first.value == pytest.approx(1347, abs=5)
    assert first.source == "geonames_city"
    assert first.confidence == pytest.approx(0.75)
    assert first.is_fallback is True
    assert second is not None
    assert second.value == pytest.approx(first.value)
    assert client.calls == [194, 702]

    with session_factory() as session:
        locations = PlayerContextRepository(session).get_all_team_locations()
        assert len(locations) == 2
        assert {location.location_source for location in locations} == {"geonames_city"}
        assert all(location.confidence == pytest.approx(0.75) for location in locations)
        ajax = next(location for location in locations if location.team_id == 194)
        assert ajax.details is not None
        assert ajax.details["city"] == "Amsterdam"
        assert ajax.details["provider_city"] == "Amsterdam"
        assert ajax.details["venue_name"] == "Johan Cruijff Arena"
        assert ajax.details["approximation"] == "city_centre"


@pytest.mark.asyncio
async def test_curated_coordinates_take_priority_over_provider_metadata(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        PlayerContextRepository(session).upsert_team_locations(
            [
                {
                    "data_source": "api_football",
                    "team_id": 1,
                    "name": "Istanbul",
                    "latitude": 41.0082,
                    "longitude": 28.9784,
                },
                {
                    "data_source": "api_football",
                    "team_id": 2,
                    "name": "Ankara",
                    "latitude": 39.9334,
                    "longitude": 32.8597,
                },
            ]
        )
    client = FakeVenueContextClient({})
    service = TravelContextService(
        session_factory=session_factory,
        resolver=GeoNamesCityResolver(),
    )

    result = await service.get_away_travel_distance(
        home_team_id=1,
        away_team_id=2,
        home_team_name="Istanbul",
        away_team_name="Ankara",
        client=client,
    )

    assert result is not None
    assert result.value == pytest.approx(351.0, abs=2.0)
    assert result.source == "curated_team_locations"
    assert result.confidence == pytest.approx(1.0)
    assert result.is_fallback is False
    assert client.calls == []


@pytest.mark.asyncio
async def test_incomplete_curated_row_is_not_silently_overwritten(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        PlayerContextRepository(session).upsert_team_locations(
            [
                {
                    "data_source": "api_football",
                    "team_id": 1,
                    "name": "Unknown",
                    "latitude": None,
                    "longitude": None,
                }
            ]
        )
    client = FakeVenueContextClient(
        {
            1: {
                "team_id": 1,
                "team_name": "Provider Name",
                "country": "Turkey",
                "city": "Istanbul",
            },
            2: {
                "team_id": 2,
                "team_name": "Ankara",
                "country": "Turkey",
                "city": "Ankara",
            },
        }
    )
    service = TravelContextService(
        session_factory=session_factory,
        resolver=GeoNamesCityResolver(),
    )

    result = await service.get_away_travel_distance(
        home_team_id=1,
        away_team_id=2,
        home_team_name="Home",
        away_team_name="Away",
        client=client,
    )

    assert result is None
    assert client.calls == [2]
    with session_factory() as session:
        preserved = PlayerContextRepository(session).get_team_location(1)
        assert preserved is not None
        assert preserved.latitude is None
        assert preserved.location_source == "manual"
