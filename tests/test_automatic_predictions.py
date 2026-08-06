from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.core.config import settings
from app.prediction.eligibility import (
    PredictionEligibilityDecision,
    PredictionIneligibleError,
)
from app.tasks.jobs import _generate_upcoming_predictions


class FakeFixtureAggregator:
    def __init__(self, fixtures: list[dict[str, Any]]) -> None:
        self.fixtures = fixtures

    async def get_upcoming_fixtures(
        self,
        days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        assert days == settings.AUTO_PREDICTION_HORIZON_DAYS
        assert limit == settings.AUTO_PREDICTION_MAX_FIXTURES
        return self.fixtures


@pytest.mark.asyncio
async def test_automatic_predictions_skip_existing_demo_and_near_kickoff() -> None:
    now = datetime(2030, 8, 1, 12, tzinfo=UTC)
    fixtures = [
        {"fixture_id": 100, "kickoff": now + timedelta(hours=2)},
        {"fixture_id": 200, "kickoff": now + timedelta(hours=3)},
        {"fixture_id": 300, "kickoff": now + timedelta(minutes=10)},
        {
            "fixture_id": 400,
            "kickoff": now + timedelta(hours=4),
            "is_demo": True,
        },
        {"fixture_id": "invalid", "kickoff": "invalid"},
    ]
    analyzed: list[int] = []

    async def analyzer(fixture_id: int) -> object:
        analyzed.append(fixture_id)
        return {"prediction_id": fixture_id}

    result = await _generate_upcoming_predictions(
        FakeFixtureAggregator(fixtures),  # type: ignore[arg-type]
        analyzer,
        {200},
        observed_at=now,
    )

    assert analyzed == [100]
    assert result == {
        "status": "succeeded",
        "fixtures_seen": 5,
        "eligible_fixtures": 1,
        "predictions_generated": 1,
        "abstained": 0,
        "failed": 0,
        "skipped_existing": 1,
        "skipped_invalid": 3,
        "started_at": now.isoformat(),
    }


@pytest.mark.asyncio
async def test_automatic_predictions_isolate_fixture_failures() -> None:
    now = datetime(2030, 8, 1, 12, tzinfo=UTC)
    fixtures = [
        {"fixture_id": 100, "kickoff": now + timedelta(hours=2)},
        {"fixture_id": 200, "kickoff": now + timedelta(hours=3)},
    ]

    async def analyzer(fixture_id: int) -> object:
        if fixture_id == 200:
            raise RuntimeError("provider unavailable")
        return {"prediction_id": fixture_id}

    result = await _generate_upcoming_predictions(
        FakeFixtureAggregator(fixtures),  # type: ignore[arg-type]
        analyzer,
        set(),
        observed_at=now,
    )

    assert result["status"] == "partial"
    assert result["predictions_generated"] == 1
    assert result["abstained"] == 0
    assert result["failed"] == 1


@pytest.mark.asyncio
async def test_automatic_predictions_report_abstention_without_failure() -> None:
    now = datetime(2030, 8, 1, 12, tzinfo=UTC)
    fixtures = [{"fixture_id": 100, "kickoff": now + timedelta(hours=2)}]

    async def analyzer(fixture_id: int) -> object:
        raise PredictionIneligibleError(
            PredictionEligibilityDecision(
                eligible=False,
                status="abstain",
                reasons=("market_unavailable",),
                data_quality_score=55.0,
            )
        )

    result = await _generate_upcoming_predictions(
        FakeFixtureAggregator(fixtures),  # type: ignore[arg-type]
        analyzer,
        set(),
        observed_at=now,
    )

    assert result["status"] == "succeeded"
    assert result["predictions_generated"] == 0
    assert result["abstained"] == 1
    assert result["failed"] == 0
