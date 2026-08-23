from __future__ import annotations

from datetime import datetime, time, timedelta
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


@pytest.mark.parametrize(
    ("name", "country", "expected_id"),
    [
        ("Scottish Premiership", "Scotland", 179),
        ("Bundesliga", "Austria", 218),
        ("Super League", "Switzerland", 207),
        ("Super League 1", "Greece", 197),
        ("Superliga", "Denmark", 119),
    ],
)
def test_canonical_mapping_supports_new_domestic_leagues(
    name: str, country: str, expected_id: int
) -> None:
    assert canonical_league_id(name, country) == expected_id


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
async def test_the_sports_db_fetches_completed_team_history() -> None:
    before = datetime(2026, 8, 14, 21, 30, tzinfo=ISTANBUL)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("searchteams.php"):
            assert request.url.params["t"] == "Çorum"
            return httpx.Response(
                200,
                json={"teams": [{"idTeam": "138951", "strTeam": "Çorum"}]},
            )
        assert request.url.path.endswith("eventslast.php")
        assert request.url.params["id"] == "138951"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "idEvent": "2470681",
                        "strTimestamp": "2026-05-15T17:00:00Z",
                        "strLeague": "Turkish 1 Lig",
                        "strHomeTeam": "Çorum",
                        "strAwayTeam": "Bodrum",
                        "idHomeTeam": "138951",
                        "idAwayTeam": "138974",
                        "intHomeScore": "1",
                        "intAwayScore": "0",
                    }
                ]
            },
        )

    source = TheSportsDBFixtureSource(
        enabled=True,
        transport=httpx.MockTransport(handler),
    )
    rows = await source.get_team_history("Çorum", before=before)

    assert len(rows) == 1
    assert rows[0]["league_id"] == 204
    assert rows[0]["actual_result"] == "HOME_WIN"
    assert rows[0]["data_source"] == "thesportsdb"


@pytest.mark.asyncio
async def test_aggregator_merges_duplicates_and_isolates_source_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kickoff = datetime.combine(
        datetime.now(ISTANBUL).date() + timedelta(days=1),
        time(12),
        tzinfo=ISTANBUL,
    )
    primary = fixture(
        101,
        "api_football",
        home="AC Horsens",
        away="Brondby",
        kickoff=kickoff,
    )
    duplicate = fixture(
        1_500_000_202,
        "football_data_org",
        home="Horsens FC",
        away="Brøndby IF",
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
        openligadb=StubSource([], configured=False),
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


def test_merge_collapses_provider_specific_team_names() -> None:
    kickoff = future_kickoff()
    canonical = fixture(
        101,
        "api_football",
        home="Clermont Foot",
        away="Dijon",
        kickoff=kickoff,
    )
    provider_variant = fixture(
        1_750_000_101,
        "fixture_download",
        home="Clermont Foot 63",
        away="Dijon FCO",
        kickoff=kickoff,
    )

    rows = FixtureAggregator._merge(
        [
            ("api_football", [canonical]),
            ("fixture_download", [provider_variant]),
        ],
        days=7,
    )

    assert len(rows) == 1
    assert rows[0]["fixture_id"] == 101
    assert rows[0]["sources"] == ["api_football", "fixture_download"]


@pytest.mark.asyncio
async def test_alternative_fixture_prefill_uses_safe_neutral_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = fixture(1_000_000_789, "thesportsdb")
    api = AsyncMock()
    odds_api = AsyncMock()
    odds_api_io = AsyncMock()
    odds_api.get_fixture_market.return_value = None
    aggregator = FixtureAggregator(
        api_football=api,
        football_data=StubSource([], configured=False),
        sportmonks=StubSource([], configured=False),
        thesportsdb=StubSource([]),
        fixture_download=StubSource([], configured=False),
        openligadb=StubSource([], configured=False),
        the_odds_api=odds_api,
        odds_api_io=odds_api_io,
    )
    monkeypatch.setattr(
        "app.services.fixture_aggregator.cache.get", AsyncMock(return_value=row)
    )

    payload = await aggregator.get_fixture_prefill(1_000_000_789)

    assert payload is not None
    assert payload["home_stats"] == {"form": 50, "attack": 50, "defense": 50, "xg": 1.2}
    assert payload["fixture"]["provider_fixture_id"] == "789"
    assert payload["data_quality"] == "fixture_source_fallback"
    api.get_fixture_prefill.assert_not_awaited()
    odds_api.get_fixture_market.assert_awaited_once()
    market_fixture = odds_api.get_fixture_market.await_args.args[0]
    assert market_fixture["fixture_id"] == row["fixture_id"]
    assert market_fixture["provider_fixture_id"] == "789"
    odds_api_io.get_fixture_market.assert_awaited_once()


@pytest.mark.asyncio
async def test_alternative_fixture_prefill_adds_the_odds_api_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = fixture(
        1_750_000_789,
        "fixture_download",
        home="Besiktas",
        away="Eyupspor",
    )
    row["league_id"] = 203
    api = AsyncMock()
    odds_api = AsyncMock()
    odds_api_io = AsyncMock()
    odds_api.get_fixture_market.return_value = {
        "raw_odds": {"HOME_WIN": 1.62, "DRAW": 4.1, "AWAY_WIN": 5.2},
        "fair_probability": {
            "HOME_WIN": 58.57,
            "DRAW": 23.14,
            "AWAY_WIN": 18.29,
        },
        "overround_pct": 5.4,
        "source": "the_odds_api",
        "captured_at": "2030-08-16T10:00:00+00:00",
    }
    aggregator = FixtureAggregator(
        api_football=api,
        football_data=StubSource([], configured=False),
        sportmonks=StubSource([], configured=False),
        thesportsdb=StubSource([]),
        fixture_download=StubSource([], configured=False),
        openligadb=StubSource([], configured=False),
        the_odds_api=odds_api,
        odds_api_io=odds_api_io,
    )
    monkeypatch.setattr(
        "app.services.fixture_aggregator.cache.get", AsyncMock(return_value=row)
    )

    payload = await aggregator.get_fixture_prefill(1_750_000_789)

    assert payload is not None
    assert payload["odd"] == 1.62
    assert payload["market_1x2"]["source"] == "the_odds_api"
    assert payload["current_odds_1x2"]["DRAW"] == 4.1
    assert payload["current_odds_at"] == "2030-08-16T10:00:00+00:00"
    assert payload["data_quality"] == "the_odds_api_market_fallback"
    assert odds_api_io.get_fixture_market.await_count == 1


@pytest.mark.asyncio
async def test_odds_api_io_runs_after_the_odds_api_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = fixture(
        1_750_000_790,
        "fixture_download",
        home="Besiktas",
        away="Eyupspor",
    )
    primary = AsyncMock()
    primary.get_fixture_market.return_value = None
    secondary = AsyncMock()
    secondary.get_fixture_market.return_value = {
        "raw_odds": {"HOME_WIN": 1.64, "DRAW": 4.2, "AWAY_WIN": 5.4},
        "fair_probability": {
            "HOME_WIN": 58.57,
            "DRAW": 22.87,
            "AWAY_WIN": 18.56,
        },
        "overround_pct": 4.79,
        "source": "odds_api_io",
        "captured_at": "2030-08-16T10:05:00+00:00",
    }
    aggregator = FixtureAggregator(
        api_football=AsyncMock(),
        football_data=StubSource([], configured=False),
        sportmonks=StubSource([], configured=False),
        thesportsdb=StubSource([]),
        fixture_download=StubSource([], configured=False),
        openligadb=StubSource([], configured=False),
        the_odds_api=primary,
        odds_api_io=secondary,
    )
    monkeypatch.setattr(
        "app.services.fixture_aggregator.cache.get", AsyncMock(return_value=row)
    )

    payload = await aggregator.get_fixture_prefill(1_750_000_790)

    assert payload is not None
    assert payload["market_1x2"]["source"] == "odds_api_io"
    assert payload["data_quality"] == "odds_api_io_market_fallback"
    assert "Odds-API.io" in payload["data_methodology"]["odds"]
    primary.get_fixture_market.assert_awaited_once()
    secondary.get_fixture_market.assert_awaited_once()


@pytest.mark.asyncio
async def test_complete_offline_league_coverage_skips_api_football(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offline = fixture(1_000_000_303, "thesportsdb")
    api = AsyncMock()
    aggregator = FixtureAggregator(
        api_football=api,
        football_data=StubSource([], configured=False),
        sportmonks=StubSource([], configured=False),
        thesportsdb=StubSource([offline]),
        fixture_download=StubSource([], configured=False),
        openligadb=StubSource([], configured=False),
    )
    monkeypatch.setattr("app.services.fixture_aggregator.ALLOWED_LEAGUE_IDS", {39})
    monkeypatch.setattr(
        "app.services.fixture_aggregator.cache.get", AsyncMock(return_value=None)
    )
    monkeypatch.setattr("app.services.fixture_aggregator.cache.set", AsyncMock())

    rows = await aggregator.get_upcoming_fixtures(days=7, limit=100)

    assert [row["fixture_id"] for row in rows] == [1_000_000_303]
    api.get_upcoming_fixtures.assert_not_awaited()
