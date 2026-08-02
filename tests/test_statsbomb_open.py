from __future__ import annotations

from datetime import UTC

import httpx
import pytest

from app.providers.statsbomb_open import StatsBombOpenDataClient, database_row

COMPETITIONS = [
    {
        "competition_id": 9,
        "season_id": 281,
        "season_name": "2023/2024",
        "competition_gender": "male",
    }
]
MATCHES = [
    {
        "match_id": 3895292,
        "match_date": "2024-04-06",
        "kick_off": "15:30:00.000",
        "competition": {"competition_id": 9},
        "season": {"season_name": "2023/2024"},
        "home_team": {"home_team_id": 190, "home_team_name": "Union Berlin"},
        "away_team": {
            "away_team_id": 904,
            "away_team_name": "Bayer Leverkusen",
        },
        "home_score": 0,
        "away_score": 1,
    }
]
EVENTS = [
    {
        "type": {"name": "Starting XI"},
        "team": {"id": 190},
        "tactics": {
            "lineup": [{"player": {"id": player_id}} for player_id in range(1, 12)]
        },
    },
    {
        "type": {"name": "Shot"},
        "team": {"id": 190},
        "shot": {"statsbomb_xg": 0.25, "outcome": {"name": "Saved"}},
    },
    {
        "type": {"name": "Shot"},
        "team": {"id": 904},
        "shot": {"statsbomb_xg": 0.65, "outcome": {"name": "Goal"}},
    },
    {
        "type": {"name": "Pass"},
        "team": {"id": 904},
        "pass": {"type": {"name": "Corner"}},
    },
    {
        "type": {"name": "Foul Committed"},
        "team": {"id": 190},
        "foul_committed": {"card": {"name": "Yellow Card"}},
    },
]


def transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/competitions.json"):
            return httpx.Response(200, json=COMPETITIONS)
        if request.url.path.endswith("/matches/9/281.json"):
            return httpx.Response(200, json=MATCHES)
        if request.url.path.endswith("/events/3895292.json"):
            return httpx.Response(200, json=EVENTS)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_statsbomb_catalog_normalizes_supported_matches() -> None:
    client = StatsBombOpenDataClient(enabled=True, transport=transport())

    fixtures = await client.get_catalog(min_season=2004)

    assert len(fixtures) == 1
    assert fixtures[0]["league_id"] == 78
    assert fixtures[0]["season"] == 2023
    assert fixtures[0]["fixture_id"] < 0
    assert fixtures[0]["actual_result"] == "AWAY_WIN"
    assert fixtures[0]["kickoff"].tzinfo == UTC


@pytest.mark.asyncio
async def test_statsbomb_events_add_xg_stats_and_starting_lineup() -> None:
    client = StatsBombOpenDataClient(enabled=True, transport=transport())
    fixtures = await client.get_catalog(min_season=2004)

    enriched, failures = await client.enrich_matches(fixtures)

    assert failures == []
    assert enriched[0]["home_xg"] == 0.25
    assert enriched[0]["away_xg"] == 0.65
    assert enriched[0]["home_shots_on_target"] == 1
    assert enriched[0]["away_corners"] == 1
    assert enriched[0]["home_yellow_cards"] == 1
    assert len(enriched[0]["home_starting_xi"]) == 11
    assert "provider_match_id" not in database_row(enriched[0])
