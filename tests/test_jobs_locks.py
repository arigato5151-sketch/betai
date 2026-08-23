from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, HistoricalFixture, MatchPrediction
from app.tasks import jobs


class _FakeLock:
    def __init__(self, *, acquired: bool) -> None:
        self.acquired = acquired
        self.available = True

    def __enter__(self) -> "_FakeLock":
        return self

    def __exit__(self, *args) -> None:
        return None


def _lock_factory(acquired: bool):
    def _factory(name: str, ttl_seconds: int = 900):
        return _FakeLock(acquired=acquired)

    return _factory


@pytest.fixture(autouse=True)
def _enable_collectors(monkeypatch) -> None:
    monkeypatch.setattr(jobs.settings, "ODDS_COLLECTOR_ENABLED", True)
    monkeypatch.setattr(jobs.settings, "LINEUP_COLLECTOR_ENABLED", True)
    monkeypatch.setattr(jobs.settings, "API_FOOTBALL_HISTORICAL_SYNC_ENABLED", True)


def _patch_lock(monkeypatch, acquired: bool) -> None:
    lock = _lock_factory(acquired=acquired)
    monkeypatch.setattr("app.tasks.predictions.DistributedTaskLock", lock)
    monkeypatch.setattr("app.tasks.fixtures_sync.DistributedTaskLock", lock)
    monkeypatch.setattr("app.tasks.ml_tasks.DistributedTaskLock", lock)
    monkeypatch.setattr("app.tasks.results.DistributedTaskLock", lock)


def test_odds_collector_skips_when_lock_is_held(monkeypatch) -> None:
    _patch_lock(monkeypatch, acquired=False)
    monkeypatch.setattr(jobs.settings, "API_FOOTBALL_KEY", "live-key")

    result = jobs.collect_upcoming_odds_task()

    assert result == {"status": "locked"}


def test_lineup_collector_skips_when_lock_is_held(monkeypatch) -> None:
    _patch_lock(monkeypatch, acquired=False)
    monkeypatch.setattr(jobs.settings, "API_FOOTBALL_KEY", "live-key")

    result = jobs.collect_upcoming_lineups_task()

    assert result == {"status": "locked"}


def test_historical_sync_skips_when_lock_is_held(monkeypatch) -> None:
    _patch_lock(monkeypatch, acquired=False)

    result = jobs.sync_historical_fixtures_task()

    assert result == {"status": "locked"}


def test_collectors_still_respect_demo_key_before_work(monkeypatch) -> None:
    _patch_lock(monkeypatch, acquired=True)
    monkeypatch.setattr(jobs.settings, "API_FOOTBALL_KEY", "DEMO_KEY")

    assert jobs.collect_upcoming_odds_task() == {"status": "demo_disabled"}
    assert jobs.collect_upcoming_lineups_task() == {"status": "demo_disabled"}


def test_drift_task_writes_cooldown_before_enqueueing(monkeypatch) -> None:
    _patch_lock(monkeypatch, acquired=True)
    monkeypatch.setattr(
        jobs.ml_pipeline, "status", Mock(return_value={"artifact_version": "v1"})
    )
    monkeypatch.setattr(
        jobs.ModelMonitoringService,
        "snapshot",
        Mock(return_value={"status": "drift", "drift_detected": True}),
    )
    cache_get = AsyncMock(return_value=None)
    cache_set = AsyncMock()
    monkeypatch.setattr(jobs.cache, "get", cache_get)
    monkeypatch.setattr(jobs.cache, "set", cache_set)
    delay = Mock()
    monkeypatch.setattr(jobs.retrain_ml_model_task, "delay", delay)

    result = jobs.monitor_model_drift_task()

    assert result["retraining_queued"] is True
    assert result["retraining_suppressed_by_cooldown"] is False
    assert result["queue_contention"] is False
    delay.assert_called_once()
    cache_set.assert_awaited_once()
    assert cache_set.call_args.args[1] == "drift-retraining:v1"


def test_drift_task_respects_active_cooldown(monkeypatch) -> None:
    _patch_lock(monkeypatch, acquired=True)
    monkeypatch.setattr(
        jobs.ml_pipeline, "status", Mock(return_value={"artifact_version": "v1"})
    )
    monkeypatch.setattr(
        jobs.ModelMonitoringService,
        "snapshot",
        Mock(return_value={"status": "drift", "drift_detected": True}),
    )
    monkeypatch.setattr(
        jobs.cache, "get", AsyncMock(return_value={"artifact_version": "v1"})
    )
    cache_set = AsyncMock()
    monkeypatch.setattr(jobs.cache, "set", cache_set)
    delay = Mock()
    monkeypatch.setattr(jobs.retrain_ml_model_task, "delay", delay)

    result = jobs.monitor_model_drift_task()

    assert result["retraining_queued"] is False
    assert result["retraining_suppressed_by_cooldown"] is True
    delay.assert_not_called()
    cache_set.assert_not_awaited()


def test_drift_task_handles_queue_contention(monkeypatch) -> None:
    _patch_lock(monkeypatch, acquired=False)
    monkeypatch.setattr(
        jobs.ml_pipeline, "status", Mock(return_value={"artifact_version": "v1"})
    )
    monkeypatch.setattr(
        jobs.ModelMonitoringService,
        "snapshot",
        Mock(return_value={"status": "drift", "drift_detected": True}),
    )
    monkeypatch.setattr(jobs.cache, "get", AsyncMock(return_value=None))
    cache_set = AsyncMock()
    monkeypatch.setattr(jobs.cache, "set", cache_set)
    delay = Mock()
    monkeypatch.setattr(jobs.retrain_ml_model_task, "delay", delay)

    result = jobs.monitor_model_drift_task()

    assert result["retraining_queued"] is False
    assert result["queue_contention"] is True
    delay.assert_not_called()
    cache_set.assert_not_awaited()


def test_completed_sync_survives_retrain_enqueue_failure(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    kickoff = datetime(2026, 8, 12, 19, tzinfo=UTC)
    with session_factory() as db:
        db.add(
            MatchPrediction(
                home_team="PEC Zwolle",
                away_team="Ajax",
                league_id=88,
                home_team_id=645,
                away_team_id=11,
                kickoff=kickoff,
                fixture_id=None,
                fixture_source=None,
                provider_fixture_id=None,
                actual_result=None,
                prediction="AWAY_WIN",
                odd=1.8,
                is_value_bet=1,
                edge=2.4,
                training_eligible=True,
                result_verification_status="pending",
                created_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )
        db.add(
            HistoricalFixture(
                fixture_id=9_000_099,
                league_id=88,
                season=2026,
                kickoff=kickoff,
                home_team_id=645,
                away_team_id=11,
                home_team="PEC Zwolle",
                away_team="Ajax",
                home_goals=0,
                away_goals=2,
                actual_result="AWAY_WIN",
                status="FT",
                data_source="api_football",
            )
        )
        db.commit()

    monkeypatch.setattr("app.tasks.results.SessionLocal", session_factory)
    monkeypatch.setattr(
        "app.tasks.results.retrain_ml_model_task",
        Mock(delay=Mock(side_effect=RuntimeError("broker unreachable"))),
    )

    result = jobs._sync_completed_matches(Mock(), Mock())

    assert result["status"] == "ready"
    assert result["verified"] == 1
