from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.core.config import settings
from app.tasks.jobs import _collect_upcoming_lineups, collect_upcoming_lineups_task


class FakeLineupClient:
    def __init__(self) -> None:
        self.lineup_calls: list[tuple[int, int, int]] = []

    async def get_upcoming_fixtures(
        self,
        days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        assert days == settings.LINEUP_COLLECTOR_HORIZON_DAYS
        assert limit == settings.LINEUP_COLLECTOR_MAX_FIXTURES
        return [
            {
                "fixture_id": 100,
                "home_team_id": 10,
                "away_team_id": 20,
                "kickoff": "2030-07-30T13:00:00+00:00",
            },
            {
                "fixture_id": 200,
                "home_team_id": 30,
                "away_team_id": 40,
                "kickoff": "2030-07-30T18:00:00+00:00",
            },
            {
                "fixture_id": "invalid",
                "home_team_id": 50,
                "away_team_id": 60,
                "kickoff": "invalid",
            },
            {
                "fixture_id": 300,
                "home_team_id": 70,
                "away_team_id": 80,
                "kickoff": "2030-07-30T13:30:00+00:00",
                "is_demo": True,
            },
        ]

    async def get_fixture_lineups(
        self,
        fixture_id: int,
        home_team_id: int,
        away_team_id: int,
    ) -> dict[str, object]:
        self.lineup_calls.append((fixture_id, home_team_id, away_team_id))
        return {
            "home_starting_xi": list(range(1, 12)),
            "away_starting_xi": list(range(20, 31)),
        }


@pytest.mark.asyncio
async def test_lineup_collector_only_fetches_pre_kickoff_candidates() -> None:
    client = FakeLineupClient()
    observed_at = datetime(2030, 7, 30, 12, tzinfo=UTC)

    result = await _collect_upcoming_lineups(
        client,  # type: ignore[arg-type]
        observed_at=observed_at,
    )

    assert result == {
        "status": "succeeded",
        "fixtures_seen": 4,
        "eligible_fixtures": 1,
        "lineups_confirmed": 1,
        "lineups_unavailable": 0,
        "outside_window": 1,
        "invalid_fixtures": 2,
        "captured_at": observed_at.isoformat(),
    }
    assert client.lineup_calls == [(100, 10, 20)]


def test_lineup_collector_task_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LINEUP_COLLECTOR_ENABLED", False)

    assert collect_upcoming_lineups_task.run() == {"status": "disabled"}


def test_lineup_collector_task_is_noop_for_demo_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.tasks import jobs

    class DemoClient:
        @staticmethod
        def _is_demo_key() -> bool:
            return True

    monkeypatch.setattr(settings, "LINEUP_COLLECTOR_ENABLED", True)
    monkeypatch.setattr(jobs, "APIFootballClient", DemoClient)

    assert collect_upcoming_lineups_task.run() == {"status": "demo_disabled"}
