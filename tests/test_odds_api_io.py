from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import settings
from app.services.odds_api_io import OddsApiIoClient


@pytest.mark.asyncio
async def test_returns_lowest_margin_timestamped_1x2_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["apiKey"] == "safe-test-key-with-enough-length-123"
        if request.url.path.endswith("/events/search"):
            assert request.url.params["query"] == "Besiktas"
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 72723138,
                        "home": "Besiktas Istanbul",
                        "away": "Eyupspor",
                        "date": "2030-08-16T18:30:00Z",
                        "status": "pending",
                    }
                ],
            )
        assert request.url.path.endswith("/odds")
        assert request.url.params["eventId"] == "72723138"
        assert request.url.params["bookmakers"] == "Bet365,Unibet"
        return httpx.Response(
            200,
            json={
                "id": 72723138,
                "bookmakers": {
                    "Bet365": [
                        {
                            "name": "ML",
                            "updatedAt": "2030-08-16T10:00:00Z",
                            "odds": [{"home": "1.62", "draw": "4.10", "away": "5.20"}],
                        }
                    ],
                    "Unibet": [
                        {
                            "name": "ML",
                            "updatedAt": "2030-08-16T10:05:00Z",
                            "odds": [{"home": "1.64", "draw": "4.20", "away": "5.40"}],
                        }
                    ],
                },
            },
        )

    monkeypatch.setattr(settings, "ODDS_API_IO_ENABLED", True)
    monkeypatch.setattr(
        "app.services.odds_api_io.cache.get", AsyncMock(return_value=None)
    )
    cache_set = AsyncMock()
    monkeypatch.setattr("app.services.odds_api_io.cache.set", cache_set)
    client = OddsApiIoClient(
        api_key="safe-test-key-with-enough-length-123",
        bookmakers="Bet365,Unibet",
        transport=httpx.MockTransport(handler),
    )

    market = await client.get_fixture_market(
        {
            "home_team": "Besiktas",
            "away_team": "Eyupspor",
            "kickoff": datetime(2030, 8, 16, 18, 30, tzinfo=UTC).isoformat(),
        }
    )

    assert market is not None
    assert market["raw_odds"] == {
        "HOME_WIN": 1.64,
        "DRAW": 4.2,
        "AWAY_WIN": 5.4,
    }
    assert market["source"] == "odds_api_io"
    assert market["bookmaker"] == "Unibet"
    assert market["provider_event_id"] == "72723138"
    assert market["provider_last_update"] == "2030-08-16T10:05:00Z"
    assert cache_set.await_count == 2


@pytest.mark.asyncio
async def test_disabled_client_makes_no_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ODDS_API_IO_ENABLED", False)
    client = OddsApiIoClient(
        api_key="safe-test-key-with-enough-length-123",
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("disabled client must not use quota")
        ),
    )

    assert (
        await client.get_fixture_market(
            {
                "home_team": "Home",
                "away_team": "Away",
                "kickoff": "2030-08-16T18:30:00+00:00",
            }
        )
        is None
    )
