from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.core.config import settings
from app.tasks.jobs import _collect_upcoming_odds, collect_upcoming_odds_task


class FakeOddsClient:
    def __init__(self) -> None:
        self.market_calls: list[int] = []

    async def get_upcoming_fixtures(
        self,
        days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        assert days == settings.ODDS_COLLECTOR_HORIZON_DAYS
        assert limit == settings.ODDS_COLLECTOR_MAX_FIXTURES
        return [
            {
                "fixture_id": 100,
                "kickoff": "2030-07-31T18:00:00+00:00",
            },
            {
                "fixture_id": 200,
                "kickoff": "2030-08-01T18:00:00+00:00",
            },
            {
                "fixture_id": "invalid",
                "kickoff": "invalid",
            },
        ]

    async def get_fixture_market(self, fixture_id: int) -> dict[str, object] | None:
        self.market_calls.append(fixture_id)
        return {
            "raw_odds": {
                "HOME_WIN": 2.1,
                "DRAW": 3.2,
                "AWAY_WIN": 3.4,
            },
            "bookmaker": "Test Book",
        }


class FakeOddsHistoryService:
    def __init__(self) -> None:
        self.enriched: list[tuple[dict[str, object], datetime | None]] = []

    def should_collect(self, **kwargs: object) -> bool:
        return kwargs["fixture_id"] == 100

    def enrich_prefill(
        self,
        prefill: dict[str, object],
        *,
        captured_at: datetime | None = None,
    ) -> dict[str, object]:
        self.enriched.append((prefill, captured_at))
        return {**prefill, "odds_history": {"status": "collecting"}}


@pytest.mark.asyncio
async def test_collector_fetches_only_due_markets_with_bounded_scope() -> None:
    client = FakeOddsClient()
    service = FakeOddsHistoryService()
    observed_at = datetime(2030, 7, 30, 12, tzinfo=UTC)

    result = await _collect_upcoming_odds(
        client,  # type: ignore[arg-type]
        service,  # type: ignore[arg-type]
        observed_at=observed_at,
    )

    assert result == {
        "status": "succeeded",
        "fixtures_seen": 3,
        "eligible_fixtures": 2,
        "snapshots_recorded": 1,
        "not_due": 1,
        "market_unavailable": 0,
        "rejected": 0,
        "invalid_fixtures": 1,
        "captured_at": observed_at.isoformat(),
    }
    assert client.market_calls == [100]
    assert len(service.enriched) == 1


def test_collector_task_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ODDS_COLLECTOR_ENABLED", False)

    assert collect_upcoming_odds_task.run() == {"status": "disabled"}


def test_collector_task_is_noop_for_demo_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.tasks import jobs

    class DemoClient:
        @staticmethod
        def _is_demo_key() -> bool:
            return True

    monkeypatch.setattr(settings, "ODDS_COLLECTOR_ENABLED", True)
    monkeypatch.setattr(jobs, "APIFootballClient", DemoClient)

    assert collect_upcoming_odds_task.run() == {"status": "demo_disabled"}
