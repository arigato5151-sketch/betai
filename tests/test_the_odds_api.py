from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import settings
from app.services.the_odds_api import LEAGUE_SPORT_KEYS, TheOddsApiClient


def test_supports_every_current_fixture_league() -> None:
    assert {
        39,
        40,
        61,
        62,
        79,
        88,
        94,
        135,
        140,
        144,
        203,
        218,
        235,
    }.issubset(LEAGUE_SPORT_KEYS)


@pytest.mark.asyncio
async def test_returns_best_timestamped_super_lig_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/soccer_turkey_super_league/odds")
        assert request.url.params["markets"] == "h2h"
        assert request.url.params["apiKey"] == "test-key-with-safe-length"
        return httpx.Response(
            200,
            json=[
                {
                    "id": "event-1",
                    "commence_time": "2030-08-16T18:30:00Z",
                    "home_team": "Beşiktaş JK",
                    "away_team": "Eyüpspor",
                    "bookmakers": [
                        {
                            "key": "book-a",
                            "title": "Book A",
                            "last_update": "2030-08-16T10:00:00Z",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Beşiktaş JK", "price": 1.62},
                                        {"name": "Draw", "price": 4.1},
                                        {"name": "Eyüpspor", "price": 5.2},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        )

    monkeypatch.setattr(settings, "THE_ODDS_API_ENABLED", True)
    monkeypatch.setattr(
        "app.services.the_odds_api.cache.get", AsyncMock(return_value=None)
    )
    cache_set = AsyncMock()
    monkeypatch.setattr("app.services.the_odds_api.cache.set", cache_set)
    client = TheOddsApiClient(
        api_key="test-key-with-safe-length",
        transport=httpx.MockTransport(handler),
    )

    market = await client.get_fixture_market(
        {
            "league_id": 203,
            "home_team": "Besiktas",
            "away_team": "Eyupspor",
            "kickoff": datetime(2030, 8, 16, 18, 30, tzinfo=UTC).isoformat(),
        }
    )

    assert market is not None
    assert market["raw_odds"] == {
        "HOME_WIN": 1.62,
        "DRAW": 4.1,
        "AWAY_WIN": 5.2,
    }
    assert market["source"] == "the_odds_api"
    assert market["bookmaker"] == "Book A"
    assert market["captured_at"].endswith("+00:00")
    cache_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_skips_unsupported_league_without_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "THE_ODDS_API_ENABLED", True)
    transport = httpx.MockTransport(
        lambda _request: pytest.fail("unsupported league must not use quota")
    )
    client = TheOddsApiClient(api_key="test-key-with-safe-length", transport=transport)

    market = await client.get_fixture_market(
        {
            "league_id": 999,
            "home_team": "Home",
            "away_team": "Away",
            "kickoff": "2030-08-16T18:30:00+00:00",
        }
    )

    assert market is None
