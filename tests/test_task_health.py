from unittest.mock import MagicMock, Mock

import redis

from app.tasks import health
from app.tasks.celery_app import celery_app


def test_broker_failure_is_reported_and_task_is_not_queued(monkeypatch) -> None:
    client = Mock()
    client.ping.side_effect = redis.ConnectionError("broker offline")
    monkeypatch.setattr(health.redis.Redis, "from_url", Mock(return_value=client))
    delay = Mock()
    monkeypatch.setattr(health.celery_app, "send_task", delay)

    result = health.enqueue_retraining()

    assert result["status"] == "broker_unavailable"
    assert result["broker_reachable"] is False
    assert result["worker_reachable"] is False
    assert result["task_queued"] is False
    delay.assert_not_called()


def test_ready_worker_queues_retraining(monkeypatch) -> None:
    monkeypatch.setattr(
        health,
        "get_worker_health",
        Mock(
            return_value={
                "status": "ready",
                "broker_reachable": True,
                "worker_reachable": True,
                "workers": ["worker@test"],
            }
        ),
    )
    monkeypatch.setattr(
        health.celery_app,
        "send_task",
        Mock(return_value=Mock(id="task-123")),
    )

    result = health.enqueue_retraining()

    assert result["status"] == "ready"
    assert result["task_queued"] is True
    assert result["task_id"] == "task-123"


def test_celery_connection_recovery_and_delivery_guards_are_enabled() -> None:
    assert celery_app.conf.broker_connection_retry_on_startup is True
    assert celery_app.conf.broker_connection_max_retries is None
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_cancel_long_running_tasks_on_connection_loss is True
    assert set(celery_app.conf.beat_schedule) == {
        "collect-upcoming-lineups",
        "collect-upcoming-odds",
        "derive-historical-xg-daily",
        "sync-football-data-fixtures-daily",
        "sync-understat-xg-daily",
        "sync-completed-matches-daily",
        "retrain-ml-model-weekly",
        "monitor-model-drift-daily",
        "sync-current-season-primary-weekly",
        "sync-uefa-fixtures-daily",
        "sync-wikidata-team-locations-weekly",
        "sync-free-team-locations-weekly",
    }


def test_retraining_task_calibrates_ensemble_before_training(monkeypatch) -> None:
    from app.tasks import jobs

    labeled_rows = [Mock(id=1, fixture_id=1)]
    historical_fixtures = [Mock(fixture_id=2)]
    player_performances = [Mock(player_id=10)]
    team_locations = [Mock(team_id=1)]
    session_context = MagicMock()
    repository = Mock()
    repository.get_all_labeled.return_value = labeled_rows
    historical_repository = Mock()
    historical_repository.get_all.return_value = historical_fixtures
    player_context_repository = Mock()
    player_context_repository.get_all_performances.return_value = player_performances
    player_context_repository.get_all_team_locations.return_value = team_locations
    calibrate = Mock(return_value={"status": "insufficient_data"})
    train = Mock(return_value=True)
    build = Mock(return_value=[])
    monkeypatch.setattr(jobs, "SessionLocal", Mock(return_value=session_context))
    monkeypatch.setattr(
        jobs, "MatchPredictionRepository", Mock(return_value=repository)
    )
    monkeypatch.setattr(
        jobs,
        "HistoricalFixtureRepository",
        Mock(return_value=historical_repository),
    )
    monkeypatch.setattr(
        jobs,
        "PlayerContextRepository",
        Mock(return_value=player_context_repository),
    )
    monkeypatch.setattr(
        jobs.ensemble_weight_manager, "optimize_and_activate", calibrate
    )
    monkeypatch.setattr(jobs.ml_pipeline, "train_pipeline", train)
    monkeypatch.setattr(
        jobs.HistoricalTrainingDataBuilder,
        "build",
        build,
    )

    result = jobs.retrain_ml_model_task.run()

    assert result == "Retraining success."
    build.assert_called_once_with(
        historical_fixtures,
        player_performances=player_performances,
        team_locations=team_locations,
    )
    calibrate.assert_called_once_with(labeled_rows)
    train.assert_called_once_with(labeled_rows)
