from datetime import UTC, datetime

import httpx
import pytest

from app.services.fixture_download import (
    FixtureDownloadClient,
    FixtureDownloadFormatError,
)


def _transport(payload: object) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == (
            "https://fixturedownload.com/feed/json/champions-league-2025"
        )
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_completed_uefa_fixtures_are_normalized() -> None:
    payload = [
        {
            "MatchNumber": 1,
            "RoundNumber": 1,
            "DateUtc": "2025-09-16 16:45:00Z",
            "HomeTeam": "PSV",
            "AwayTeam": "Union SG",
            "HomeTeamScore": 1,
            "AwayTeamScore": 3,
        },
        {
            "MatchNumber": 2,
            "RoundNumber": 1,
            "DateUtc": "2027-09-16 16:45:00Z",
            "HomeTeam": "Future Home",
            "AwayTeam": "Future Away",
            "HomeTeamScore": None,
            "AwayTeamScore": None,
        },
    ]
    client = FixtureDownloadClient(transport=_transport(payload))

    fixtures = await client.get_completed_fixtures(
        2, 2025, now=datetime(2026, 8, 1, tzinfo=UTC)
    )

    assert len(fixtures) == 1
    assert fixtures[0]["actual_result"] == "AWAY_WIN"
    assert fixtures[0]["data_source"] == "fixture_download"
    assert fixtures[0]["fixture_id"] < 0
    assert fixtures[0]["home_team_id"] < 0


@pytest.mark.asyncio
async def test_invalid_score_fails_closed() -> None:
    client = FixtureDownloadClient(
        transport=_transport(
            [
                {
                    "DateUtc": "2025-09-16 16:45:00Z",
                    "HomeTeam": "PSV",
                    "AwayTeam": "Union SG",
                    "HomeTeamScore": "1",
                    "AwayTeamScore": 3,
                }
            ]
        )
    )

    with pytest.raises(FixtureDownloadFormatError, match="score"):
        await client.get_completed_fixtures(
            2, 2025, now=datetime(2026, 8, 1, tzinfo=UTC)
        )


@pytest.mark.asyncio
async def test_unsupported_league_is_rejected_before_network() -> None:
    client = FixtureDownloadClient(transport=_transport([]))

    with pytest.raises(ValueError, match="Unsupported UEFA"):
        await client.get_completed_fixtures(39, 2025)
