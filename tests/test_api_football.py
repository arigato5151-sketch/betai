from unittest.mock import AsyncMock, call

import httpx
import pandas as pd
import pytest

from app.core.demo_data import DEMO_UPCOMING_FIXTURES
from app.core.exceptions import APIDataError
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
async def test_completed_fixtures_surfaces_provider_plan_errors() -> None:
    client = APIFootballClient()
    client.api_key = "live-key"
    client._request_with_retry = AsyncMock(
        return_value={
            "errors": {"plan": "Season is unavailable"},
            "response": [],
        }
    )

    with pytest.raises(APIDataError, match="Season is unavailable"):
        await client.get_completed_fixtures(203, 2025)


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


@pytest.mark.asyncio
async def test_completed_fixtures_are_normalized_and_invalid_rows_are_skipped() -> None:
    client = APIFootballClient()
    client.api_key = "live-key"
    client._request_with_retry = AsyncMock(
        return_value={
            "response": [
                {
                    "fixture": {
                        "id": 500,
                        "date": "2026-07-18T18:00:00Z",
                        "status": {"short": "FT"},
                    },
                    "league": {"id": 203, "season": 2026},
                    "teams": {
                        "home": {"id": 1, "name": "Home"},
                        "away": {"id": 2, "name": "Away"},
                    },
                    "goals": {"home": 2, "away": 0},
                    "lineups": [
                        {
                            "team": {"id": 1},
                            "startXI": [
                                {"player": {"id": player_id}}
                                for player_id in range(1, 12)
                            ],
                        },
                        {
                            "team": {"id": 2},
                            "startXI": [
                                {"player": {"id": player_id}}
                                for player_id in range(20, 31)
                            ],
                        },
                    ],
                    "players": [
                        {
                            "team": {"id": 1},
                            "players": [
                                {
                                    "player": {"id": 1},
                                    "statistics": [
                                        {
                                            "games": {
                                                "minutes": 90,
                                                "position": "F",
                                                "rating": "8.2",
                                            },
                                            "goals": {"total": 1, "assists": 1},
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {
                    "fixture": {
                        "id": 501,
                        "date": "2026-07-19T18:00:00Z",
                        "status": {"short": "NS"},
                    }
                },
            ]
        }
    )

    fixtures = await client.get_completed_fixtures(203, 2026)

    assert len(fixtures) == 1
    assert fixtures[0]["fixture_id"] == 500
    assert fixtures[0]["actual_result"] == "HOME_WIN"
    assert fixtures[0]["home_starting_xi"] == list(range(1, 12))
    assert fixtures[0]["away_starting_xi"] == list(range(20, 31))
    assert fixtures[0]["player_performances"] == [
        {
            "fixture_id": 500,
            "league_id": 203,
            "kickoff": pd.Timestamp("2026-07-18T18:00:00Z"),
            "team_id": 1,
            "player_id": 1,
            "started": True,
            "minutes": 90,
            "rating": 8.2,
            "position": "F",
            "goals": 1,
            "assists": 1,
            "source": "api_football_fixture_players",
        }
    ]
    assert fixtures[0]["kickoff"] == pd.Timestamp("2026-07-18T18:00:00Z")
    client._request_with_retry.assert_awaited_once_with(
        "fixtures",
        {
            "league": "203",
            "season": "2026",
            "status": "FT-AET-PEN",
            "timezone": "UTC",
        },
    )


@pytest.mark.asyncio
async def test_fixture_player_context_derives_starting_xi_and_performances() -> None:
    client = APIFootballClient()
    client.api_key = "live-key"

    def team_block(team_id: int, first_player_id: int) -> dict:
        return {
            "team": {"id": team_id},
            "players": [
                {
                    "player": {"id": player_id},
                    "statistics": [
                        {
                            "games": {
                                "minutes": 90,
                                "position": "M",
                                "rating": "7.2",
                                "substitute": False,
                            },
                            "goals": {"total": 0, "assists": 0},
                        }
                    ],
                }
                for player_id in range(first_player_id, first_player_id + 11)
            ],
        }

    client._request_with_retry = AsyncMock(
        return_value={
            "response": [
                team_block(1, 1),
                team_block(2, 20),
            ]
        }
    )
    kickoff = pd.Timestamp("2026-07-18T18:00:00Z").to_pydatetime()

    context = await client.get_fixture_player_context(
        fixture_id=500,
        league_id=203,
        kickoff=kickoff,
        home_team_id=1,
        away_team_id=2,
    )

    assert context["home_starting_xi"] == list(range(1, 12))
    assert context["away_starting_xi"] == list(range(20, 31))
    performances = context["player_performances"]
    assert isinstance(performances, list)
    assert len(performances) == 22
    assert all(row["started"] is True for row in performances)
    client._request_with_retry.assert_awaited_once_with(
        "fixtures/players",
        {"fixture": "500"},
    )


@pytest.mark.asyncio
async def test_completed_fixture_ingestion_is_empty_in_demo_mode() -> None:
    client = APIFootballClient()
    client.api_key = "DEMO_KEY"
    client._request_with_retry = AsyncMock(side_effect=AssertionError("network called"))

    assert await client.get_completed_fixtures(203, 2026) == []
    client._request_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_fixture_availability_counts_missing_and_questionable_players() -> None:
    client = APIFootballClient()
    client.api_key = "live-key"
    client._request_with_retry = AsyncMock(
        return_value={
            "response": [
                {
                    "team": {"id": 1},
                    "player": {
                        "id": 11,
                        "name": "Critical Player",
                        "type": "Missing Fixture",
                        "reason": "Suspended",
                    },
                },
                {
                    "team": {"id": 1},
                    "player": {
                        "id": 12,
                        "name": "Doubtful Player",
                        "type": "Questionable",
                        "reason": "Knock",
                    },
                },
                {
                    "team": {"id": 2},
                    "player": {
                        "id": 21,
                        "name": "Away Player",
                        "type": "Missing Fixture",
                        "reason": "Injury",
                    },
                },
                {
                    "team": {"id": 2},
                    "player": {
                        "id": 21,
                        "name": "Away Player",
                        "type": "Missing Fixture",
                        "reason": "Duplicate",
                    },
                },
                {"team": {"id": 99}, "player": {"type": "Missing Fixture"}},
                {"team": {"id": 2}, "player": {"type": "Unknown"}},
            ]
        }
    )

    availability = await client.get_fixture_availability(777001, 1, 2)

    assert availability == {
        "home_missing_players": 1,
        "away_missing_players": 1,
        "home_questionable_players": 1,
        "away_questionable_players": 0,
        "availability_report_present": 1,
        "home_unavailable_players": [
            {
                "player_id": 11,
                "name": "Critical Player",
                "status": "missing",
                "reason": "Suspended",
            },
            {
                "player_id": 12,
                "name": "Doubtful Player",
                "status": "questionable",
                "reason": "Knock",
            },
        ],
        "away_unavailable_players": [
            {
                "player_id": 21,
                "name": "Away Player",
                "status": "missing",
                "reason": "Injury",
            }
        ],
        "source": "api_football_injuries",
    }
    client._request_with_retry.assert_awaited_once_with(
        "injuries", {"fixture": "777001"}
    )


@pytest.mark.asyncio
async def test_fixture_availability_is_disabled_in_demo_mode() -> None:
    client = APIFootballClient()
    client.api_key = "DEMO_KEY"
    client._request_with_retry = AsyncMock(side_effect=AssertionError("network called"))

    assert await client.get_fixture_availability(777002, 1, 2) is None
    assert await client.get_fixture_lineups(777002, 1, 2) is None
    client._request_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_team_player_ratings_normalize_and_paginate() -> None:
    client = APIFootballClient()
    client.api_key = "live-key"
    client._request_with_retry = AsyncMock(
        side_effect=[
            {
                "paging": {"current": 1, "total": 2},
                "response": [
                    {
                        "player": {"id": 101},
                        "statistics": [
                            {
                                "team": {"id": 987654},
                                "league": {"id": 203},
                                "games": {
                                    "rating": "7.4",
                                    "minutes": 900,
                                    "appearences": 12,
                                },
                                "goals": {"total": 4, "assists": 3},
                            }
                        ],
                    }
                ],
            },
            {
                "paging": {"current": 2, "total": 2},
                "response": [
                    {
                        "player": {"id": 102},
                        "statistics": [
                            {
                                "team": {"id": 987654},
                                "league": {"id": 203},
                                "games": {
                                    "rating": "not-a-rating",
                                    "minutes": 100,
                                    "appearences": 2,
                                },
                                "goals": {"total": 0, "assists": 0},
                            }
                        ],
                    },
                    {
                        "player": {"id": 103},
                        "statistics": [
                            {
                                "team": {"id": 987654},
                                "league": {"id": 203},
                                "games": {
                                    "rating": "6.8",
                                    "minutes": 450,
                                    "appearences": 7,
                                },
                                "goals": {"total": 1, "assists": 2},
                            }
                        ],
                    },
                ],
            },
        ]
    )

    ratings = await client.get_team_player_ratings(987654, 2026, league_id=203)

    assert ratings == {
        101: {
            "rating": 7.4,
            "minutes": 900.0,
            "appearances": 12.0,
            "goals": 4.0,
            "assists": 3.0,
        },
        103: {
            "rating": 6.8,
            "minutes": 450.0,
            "appearances": 7.0,
            "goals": 1.0,
            "assists": 2.0,
        },
    }
    assert client._request_with_retry.await_args_list == [
        call(
            "players",
            {
                "team": "987654",
                "season": "2026",
                "page": "1",
                "league": "203",
            },
        ),
        call(
            "players",
            {
                "team": "987654",
                "season": "2026",
                "page": "2",
                "league": "203",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_fixture_lineups_normalize_starting_player_ids() -> None:
    client = APIFootballClient()
    client.api_key = "live-key"
    client._request_with_retry = AsyncMock(
        return_value={
            "response": [
                {
                    "team": {"id": 1},
                    "startXI": [
                        {"player": {"id": player_id}} for player_id in [1, 2, 2, 3]
                    ],
                },
                {
                    "team": {"id": 2},
                    "startXI": [
                        {"player": {"id": player_id}} for player_id in range(20, 31)
                    ],
                },
            ]
        }
    )

    lineups = await client.get_fixture_lineups(778001, 1, 2)

    assert lineups == {
        "home_starting_xi": [1, 2, 3],
        "away_starting_xi": list(range(20, 31)),
        "source": "api_football_lineups",
    }
    client._request_with_retry.assert_awaited_once_with(
        "fixtures/lineups", {"fixture": "778001"}
    )
