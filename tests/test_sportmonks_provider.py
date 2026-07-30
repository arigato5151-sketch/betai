from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from app.providers.sportmonks import (
    SPORTMONKS_PLAYER_ID_OFFSET,
    SportmonksClient,
)


def _lineups(rating: float, minutes: int) -> list[dict[str, object]]:
    return [
        {
            "player_id": player_id,
            "team_id": 62,
            "type_id": 11,
            "details": [
                {"type_id": 118, "data": {"value": rating}},
                {"type_id": 119, "data": {"value": minutes}},
                {"type_id": 52, "data": {"value": int(player_id == 1)}},
                {"type_id": 79, "data": {"value": int(player_id == 2)}},
            ],
        }
        for player_id in range(1, 12)
    ]


@pytest.mark.asyncio
async def test_recent_player_ratings_use_exact_team_and_namespaced_ids() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "/teams/search/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": 62, "name": "Rangers"},
                        {"id": 999, "name": "Rangers Women"},
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "starting_at": "2030-07-20 18:00:00",
                        "lineups": _lineups(8.0, 90),
                    },
                    {
                        "starting_at": "2030-07-10 18:00:00",
                        "lineups": _lineups(6.0, 30),
                    },
                    {
                        "starting_at": "2030-08-02 18:00:00",
                        "lineups": _lineups(10.0, 90),
                    },
                ]
            },
        )

    client = SportmonksClient(
        api_token="secret-token",
        transport=httpx.MockTransport(handler),
        lookback_days=60,
        lookback_matches=10,
    )

    resolved = await client.get_recent_player_ratings(
        team_name="Rangers",
        as_of=datetime(2030, 7, 30, tzinfo=UTC),
    )

    assert resolved is not None
    candidate, ratings = resolved
    assert candidate.provider_team_key == "62"
    assert len(ratings) == 11
    assert ratings[SPORTMONKS_PLAYER_ID_OFFSET + 1] == {
        "rating": 7.5,
        "minutes": 120.0,
        "appearances": 2.0,
        "goals": 2.0,
        "assists": 0.0,
    }
    assert all(
        request.headers["Authorization"] == "secret-token" for request in requests
    )
    assert all("api_token" not in request.url.params for request in requests)
    fixture_request = requests[1]
    assert fixture_request.url.params["include"] == "lineups.details"
    assert fixture_request.url.params["filters"] == "lineupDetailTypes:52,79,118,119"


@pytest.mark.asyncio
async def test_team_resolution_fails_closed_for_ambiguous_exact_names() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": 1, "name": "United"},
                    {"id": 2, "name": "United"},
                ]
            },
        )

    client = SportmonksClient(
        api_token="secret-token",
        transport=httpx.MockTransport(handler),
    )

    assert await client.resolve_team(team_name="United") is None


@pytest.mark.asyncio
async def test_unconfigured_client_does_not_make_network_requests() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("network request was not expected")

    client = SportmonksClient(
        api_token="",
        transport=httpx.MockTransport(handler),
    )

    assert (
        await client.get_recent_player_ratings(
            team_name="Rangers",
            as_of=datetime(2030, 7, 30, tzinfo=UTC),
        )
        is None
    )
