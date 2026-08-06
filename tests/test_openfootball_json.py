from datetime import UTC, datetime

import httpx
import pytest

from app.services.openfootball_json import (
    OpenFootballFormatError,
    OpenFootballJSONClient,
)
from app.tasks.celery_app import celery_app


def _transport(payload: object) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == (
            "https://raw.githubusercontent.com/openfootball/football.json/master/"
            "2025-26/gr.1.json"
        )
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_greek_super_league_results_are_normalized() -> None:
    payload = {
        "name": "Greek Super League 2025/26",
        "matches": [
            {
                "round": "1. Round",
                "date": "2025-08-23",
                "time": "19:00",
                "team1": "Aris Saloniki",
                "team2": "Volos NFC",
                "score": {"ft": [2, 0], "ht": [0, 0]},
            },
            {
                "round": "2. Round",
                "date": "2025-08-30",
                "time": "19:00",
                "team1": "Future Home",
                "team2": "Future Away",
            },
        ],
    }
    client = OpenFootballJSONClient(transport=_transport(payload))

    rows = await client.get_completed_fixtures(197, 2025)

    assert len(rows) == 1
    assert rows[0]["league_id"] == 197
    assert rows[0]["kickoff"] == datetime(2025, 8, 23, 16, 0, tzinfo=UTC)
    assert rows[0]["actual_result"] == "HOME_WIN"
    assert rows[0]["half_time_home_goals"] == 0
    assert rows[0]["data_source"] == "openfootball_json"
    assert rows[0]["fixture_id"] < 0


@pytest.mark.asyncio
async def test_invalid_openfootball_score_fails_closed() -> None:
    payload = {
        "matches": [
            {
                "date": "2025-08-23",
                "team1": "Home",
                "team2": "Away",
                "score": {"ft": ["2", 0]},
            }
        ]
    }

    with pytest.raises(OpenFootballFormatError, match="score"):
        await OpenFootballJSONClient(
            transport=_transport(payload)
        ).get_completed_fixtures(197, 2025)


@pytest.mark.asyncio
async def test_disabled_openfootball_client_does_not_call_network() -> None:
    client = OpenFootballJSONClient(enabled=False, transport=_transport({}))

    assert await client.get_completed_fixtures(197, 2025) == []


def test_openfootball_sync_is_scheduled_daily() -> None:
    schedule = celery_app.conf.beat_schedule["sync-openfootball-fixtures-daily"]

    assert schedule["task"] == "app.tasks.jobs.sync_openfootball_fixtures_task"
    assert schedule["schedule"] == 86400.0
