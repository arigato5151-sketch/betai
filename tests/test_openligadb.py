from datetime import UTC, datetime

import httpx
import pytest

from app.providers.openligadb import OpenLigaDBClient, is_openligadb_fixture_id


@pytest.mark.asyncio
async def test_openligadb_normalizes_supported_upcoming_fixture() -> None:
    payload = [
        {
            "matchID": 80001,
            "matchDateTimeUTC": "2026-08-07T18:30:00Z",
            "leagueSeason": "2026",
            "leagueShortcut": "bl2",
            "matchIsFinished": False,
            "team1": {"teamId": 9, "teamName": "FC Schalke 04"},
            "team2": {"teamId": 54, "teamName": "Hertha BSC"},
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        rows = payload if "/bl2/" in request.url.path else []
        return httpx.Response(200, json=rows)

    client = OpenLigaDBClient(
        enabled=True,
        transport=httpx.MockTransport(handler),
    )
    fixtures = await client.get_upcoming_fixtures(
        datetime(2026, 8, 2, tzinfo=UTC).date(),
        datetime(2026, 8, 9, tzinfo=UTC).date(),
    )

    assert len(fixtures) == 1
    assert fixtures[0]["fixture_id"] == 500_080_001
    assert fixtures[0]["league_id"] == 79
    assert fixtures[0]["home_team_id"] == 500_000_009
    assert fixtures[0]["source"] == "openligadb"


@pytest.mark.asyncio
async def test_openligadb_is_disabled_without_network() -> None:
    transport = httpx.MockTransport(
        lambda _request: pytest.fail("disabled provider must not use the network")
    )
    client = OpenLigaDBClient(enabled=False, transport=transport)

    fixtures = await client.get_upcoming_fixtures(
        datetime(2026, 8, 2, tzinfo=UTC).date(),
        datetime(2026, 8, 9, tzinfo=UTC).date(),
    )

    assert fixtures == []


@pytest.mark.asyncio
async def test_openligadb_fetches_finished_fixture_by_namespaced_id() -> None:
    payload = {
        "matchID": 80001,
        "matchDateTimeUTC": "2026-08-07T18:30:00Z",
        "leagueSeason": "2026",
        "leagueShortcut": "bl2",
        "matchIsFinished": True,
        "team1": {"teamId": 9, "teamName": "FC Schalke 04"},
        "team2": {"teamId": 54, "teamName": "Hertha BSC"},
        "matchResults": [
            {
                "resultOrderID": 1,
                "pointsTeam1": 1,
                "pointsTeam2": 0,
            },
            {
                "resultOrderID": 2,
                "pointsTeam1": 2,
                "pointsTeam2": 1,
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/getmatchdata/80001"
        return httpx.Response(200, json=payload)

    client = OpenLigaDBClient(enabled=True, transport=httpx.MockTransport(handler))
    fixture = await client.get_fixture_by_id(500_080_001)

    assert fixture is not None
    assert fixture["status"] == "FT"
    assert fixture["score"] == "2 - 1"
    assert is_openligadb_fixture_id(fixture["fixture_id"])
    assert not is_openligadb_fixture_id(80_001)
