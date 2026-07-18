from unittest.mock import Mock

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
        "sync-completed-matches-daily",
        "retrain-ml-model-weekly",
    }
