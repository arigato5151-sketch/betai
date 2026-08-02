from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.fixture_aggregator import (
    ISTANBUL,
    FixtureAggregator,
    FixtureDownloadFixtureSource,
    FootballDataOrgFixtureSource,
    SportmonksFixtureSource,
    TheSportsDBFixtureSource,
    canonical_league_id,
)


class StubSource:
    def __init__(
        self, rows: list[dict[str, object]], *, configured: bool = True
    ) -> None:
        self.rows = rows
        self.configured = configured
        self.get_fixtures = AsyncMock(return_value=rows)


def future_kickoff(days: int = 1, hours: int = 0) -> datetime:
    return datetime.now(ISTANBUL) + timedelta(days=days, hours=hours)


def fixture(
    fixture_id: int,
    source: str,
    *,
    home: str = "Arsenal",
    away: str = "Chelsea",
    kickoff: datetime | None = None,
) -> dict[str, object]:
    start = kickoff or future_kickoff()
    return {
        "fixture_id": fixture_id,
        "league": "Premier League",
        "home_team": home,
        "away_team": away,
        "home_team_id": fixture_id + 10,
        "away_team_id": fixture_id + 20,
        "league_id": 39,
        "season": start.year,
        "minute": None,
        "score": None,
        "kickoff": start.isoformat(),
        "kickoff_label": start.strftime("%d.%m %H:%M"),
        "status": "NS",
        "is_live": False,
        "is_demo": False,
        "source": source,
        "sources": [source],
    }


def test_canonical_league_mapping_disambiguates_premier_leagues() -> None:
    assert canonical_league_id("Premier League", "England") == 39
    assert canonical_league_id("Premier League", "Russia") == 235
    assert canonical_league_id("UEFA Europa Conference League") == 848


def test_fixture_download_normalization_uses_safe_positive_ids() -> None:
    row = FixtureDownloadFixtureSource._normalize(
        {
            "fixture_id": -8_000_000_000_000,
            "league_id": 62,
            "kickoff": future_kickoff(),
            "home_team_id": -7_000_000_000_000,
            "away_team_id": -6_000_000_000_000,
            "home_team": "Lorient",
            "away_team": "Reims",
        }
    )

    assert 1_750_000_000 < row["fixture_id"] < 2_000_000_000
    assert row["source"] == "fixture_download"


@pytest.mark.asyncio
async def test_football_data_org_normalizes_fixture() -> None:
    kickoff = future_kickoff()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Auth-Token"] == "test-key"
        return httpx.Response(
            200,
            json={
                "matches": [
                    {
                        "id": 123,
                        "utcDate": kickoff.astimezone().isoformat(),
                        "competition": {"code": "PL", "name": "Premier League"},
                        "homeTeam": {"id": 1, "name": "Arsenal"},
                        "awayTeam": {"id": 2, "name": "Chelsea"},
                    }
                ]
            },
        )

    source = FootballDataOrgFixtureSource(
        api_key="test-key",
        enabled=True,
        transport=httpx.MockTransport(handler),
    )
    rows = await source.get_fixtures(kickoff.date(), kickoff.date())

    assert rows[0]["fixture_id"] == 1_500_000_123
    assert rows[0]["league_id"] == 39
    assert rows[0]["source"] == "football_data_org"


@pytest.mark.asyncio
async def test_sportmonks_normalizes_fixture() -> None:
    kickoff = future_kickoff()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "test-token"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": 456,
                        "starting_at": kickoff.isoformat(),
                        "league": {
                            "name": "Premier League",
                            "country": {"name": "England"},
                        },
                        "participants": [
                            {"id": 11, "name": "Arsenal", "meta": {"location": "home"}},
                            {"id": 12, "name": "Chelsea", "meta": {"location": "away"}},
                        ],
                    }
                ]
            },
        )

    source = SportmonksFixtureSource(
        api_token="test-token",
        enabled=True,
        transport=httpx.MockTransport(handler),
    )
    rows = await source.get_fixtures(kickoff.date(), kickoff.date())

    assert rows[0]["fixture_id"] == 1_250_000_456
    assert rows[0]["home_team_id"] == 1_250_000_011


@pytest.mark.asyncio
async def test_the_sports_db_normalizes_fixture() -> None:
    kickoff = future_kickoff()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "events": [
                    {
                        "idEvent": "789",
                        "strTimestamp": kickoff.isoformat(),
                        "strLeague": "English Premier League",
                        "strCountry": "England",
                        "strHomeTeam": "Arsenal",
                        "strAwayTeam": "Chelsea",
                        "idHomeTeam": "21",
                        "idAwayTeam": "22",
                    }
                ]
            },
        )

    source = TheSportsDBFixtureSource(
        enabled=True,
        transport=httpx.MockTransport(handler),
    )
    rows = await source.get_fixtures(kickoff.date(), kickoff.date())

    assert rows[0]["fixture_id"] == 1_000_000_789
    assert rows[0]["league_id"] == 39


@pytest.mark.asyncio
async def test_aggregator_merges_duplicates_and_isolates_source_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kickoff = future_kickoff()
    primary = fixture(101, "api_football", kickoff=kickoff)
    duplicate = fixture(
        1_500_000_202,
        "football_data_org",
        home=" Arsenal ",
        away="CHELSEA",
        kickoff=kickoff + timedelta(minutes=30),
    )
    extra = fixture(
        1_000_000_303,
        "thesportsdb",
        home="Liverpool",
        away="Everton",
        kickoff=kickoff + timedelta(hours=2),
    )
    api = AsyncMock()
    api.get_upcoming_fixtures.return_value = [primary]
    failing = StubSource([])
    failing.get_fixtures.side_effect = RuntimeError("temporary failure")
    aggregator = FixtureAggregator(
        api_football=api,
        football_data=StubSource([duplicate]),
        sportmonks=failing,
        thesportsdb=StubSource([extra]),
        fixture_download=StubSource([], configured=False),
    )
    monkeypatch.setattr(
        "app.services.fixture_aggregator.cache.get", AsyncMock(return_value=None)
    )
    cache_set = AsyncMock()
    monkeypatch.setattr("app.services.fixture_aggregator.cache.set", cache_set)

    rows = await aggregator.get_upcoming_fixtures(days=7, limit=100)

    assert [row["fixture_id"] for row in rows] == [101, 1_000_000_303]
    assert rows[0]["sources"] == ["api_football", "football_data_org"]
    assert cache_set.await_count == 3


@pytest.mark.asyncio
async def test_alternative_fixture_prefill_uses_safe_neutral_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = fixture(1_000_000_789, "thesportsdb")
    api = AsyncMock()
    aggregator = FixtureAggregator(
        api_football=api,
        football_data=StubSource([], configured=False),
        sportmonks=StubSource([], configured=False),
        thesportsdb=StubSource([]),
        fixture_download=StubSource([], configured=False),
    )
    monkeypatch.setattr(
        "app.services.fixture_aggregator.cache.get", AsyncMock(return_value=row)
    )

    payload = await aggregator.get_fixture_prefill(1_000_000_789)

    assert payload is not None
    assert payload["home_stats"] == {"form": 50, "attack": 50, "defense": 50, "xg": 1.2}
    assert payload["data_quality"] == "fixture_source_fallback"
    api.get_fixture_prefill.assert_not_awaited()
