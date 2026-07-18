from unittest.mock import AsyncMock

import httpx
import pandas as pd
import pytest

from app.core.demo_data import DEMO_UPCOMING_FIXTURES
from app.services.api_football import APIFootballClient


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
        text: str = "error",
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.get_calls = 0

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str, params: dict) -> FakeResponse:
        del url, params
        response = self.responses[self.get_calls]
        self.get_calls += 1
        return response


def install_fake_http_client(
    monkeypatch: pytest.MonkeyPatch, responses: list[FakeResponse]
) -> FakeAsyncClient:
    fake_client = FakeAsyncClient(responses)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: fake_client,
    )
    return fake_client


@pytest.mark.asyncio
async def test_retry_recovers_from_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = APIFootballClient()
    fake_client = install_fake_http_client(
        monkeypatch,
        [FakeResponse(500), FakeResponse(200, payload={"response": [1]})],
    )
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.api_football.asyncio.sleep", sleep)

    result = await client._request_with_retry("fixtures", {}, base_backoff=0.01)

    assert result == {"response": [1]}
    assert fake_client.get_calls == 2
    sleep.assert_awaited_once_with(0.01)


@pytest.mark.asyncio
async def test_permanent_client_error_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = APIFootballClient()
    fake_client = install_fake_http_client(
        monkeypatch,
        [FakeResponse(401), FakeResponse(200, payload={"unexpected": True})],
    )
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.api_football.asyncio.sleep", sleep)

    result = await client._request_with_retry("fixtures", {}, retries=3)

    assert result is None
    assert fake_client.get_calls == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_retry_after_uses_bounded_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = APIFootballClient()
    install_fake_http_client(
        monkeypatch,
        [
            FakeResponse(429, headers={"Retry-After": "not-a-number"}),
            FakeResponse(200, payload={"response": []}),
        ],
    )
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.api_football.asyncio.sleep", sleep)

    result = await client._request_with_retry("fixtures", {}, retries=2)

    assert result == {"response": []}
    sleep.assert_awaited_once_with(60)


@pytest.mark.asyncio
async def test_demo_upcoming_and_prefill_never_require_network() -> None:
    client = APIFootballClient()
    client.api_key = "DEMO_KEY"
    client._request_with_retry = AsyncMock(side_effect=AssertionError("network called"))

    fixtures = await client.get_upcoming_fixtures(limit=2)
    prefill = await client.get_fixture_prefill(DEMO_UPCOMING_FIXTURES[0]["fixture_id"])

    assert len(fixtures) == 2
    assert all(fixture["is_demo"] for fixture in fixtures)
    assert prefill is not None
    assert prefill["data_quality"] == "demo"
    assert prefill["auto_filled"] is True
    assert prefill["market_1x2"]["overround_pct"] > 0
    client._request_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_demo_team_uses_local_fallback_without_network() -> None:
    client = APIFootballClient()
    client.api_key = "DEMO_KEY"
    client._request_with_retry = AsyncMock(side_effect=AssertionError("network called"))

    profile = await client.get_team_statistics(203, 2024, 999999, venue="away")

    assert profile["source"] == "fallback_default"
    assert profile["venue"] == "away"
    client._request_with_retry.assert_not_awaited()


def test_fixture_normalization_handles_live_score_and_invalid_date() -> None:
    client = APIFootballClient()
    client.api_key = "live-key"
    normalized = client._normalize_fixture(
        {
            "fixture": {
                "id": 42,
                "date": "invalid-date-value",
                "status": {"short": "2H", "elapsed": 67},
            },
            "league": {"id": 203, "name": "Super Lig", "season": 2026},
            "teams": {
                "home": {"id": 1, "name": "Home"},
                "away": {"id": 2, "name": "Away"},
            },
            "goals": {"home": 2, "away": 1},
        }
    )

    assert normalized["fixture_id"] == 42
    assert normalized["score"] == "2 - 1"
    assert normalized["is_live"] is True
    assert normalized["is_demo"] is False
    assert normalized["kickoff_label"] == "invalid-date-val"


@pytest.mark.parametrize(
    ("home_id", "home_goals", "away_goals", "expected"),
    [(10, 2, 1, "W"), (10, 1, 1, "D"), (10, 0, 1, "L"), (99, 0, 1, "W")],
)
def test_result_for_team_respects_home_and_away_perspective(
    home_id: int, home_goals: int, away_goals: int, expected: str
) -> None:
    item = {
        "teams": {"home": {"id": 10}, "away": {"id": 99}},
        "goals": {"home": home_goals, "away": away_goals},
    }

    assert APIFootballClient._result_for_team(item, home_id) == expected


def test_team_match_row_builds_model_features() -> None:
    client = APIFootballClient()
    row = client._team_match_row(
        {
            "fixture": {"date": "2026-07-13T18:00:00Z"},
            "teams": {"home": {"id": 10}, "away": {"id": 20}},
            "goals": {"home": 0, "away": 2},
        },
        team_id=20,
    )

    assert row is not None
    assert row["result"] == "W"
    assert row["goals_for"] == 2
    assert row["goals_against"] == 0
    assert row["clean_sheet"] == 1
    assert isinstance(row["match_date"], pd.Timestamp)


@pytest.mark.asyncio
async def test_prefill_marks_mixed_live_and_fallback_stats_as_fallback() -> None:
    client = APIFootballClient()
    client.api_key = "live-key"
    client.get_fixture_by_id = AsyncMock(
        return_value={
            "fixture_id": 10,
            "home_team": "Home",
            "away_team": "Away",
            "home_team_id": 1,
            "away_team_id": 2,
            "league_id": 203,
            "season": 2026,
        }
    )
    client.get_team_statistics = AsyncMock(
        side_effect=[
            {"source": "api_football_season_stats"},
            {"source": "fallback_default"},
        ]
    )
    client.get_fixture_market = AsyncMock(return_value=None)
    client.get_fixture_odds = AsyncMock(return_value=1.85)

    prefill = await client.get_fixture_prefill(10)

    assert prefill is not None
    assert prefill["data_quality"] == "fallback"
    assert prefill["odd"] == 1.85
