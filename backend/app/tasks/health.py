from __future__ import annotations

import logging
from typing import Any

import redis
from celery.exceptions import CeleryError
from kombu.exceptions import OperationalError

from app.core.config import settings
from app.tasks.celery_app import celery_app


logger = logging.getLogger("bet-ai-pro.tasks.health")


def get_worker_health() -> dict[str, Any]:
    try:
        client = redis.Redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        client.ping()
    except redis.RedisError as exc:
        logger.warning("Celery broker is unavailable: %s", exc)
        return {
            "status": "broker_unavailable",
            "broker_reachable": False,
            "worker_reachable": False,
            "workers": [],
        }

    try:
        replies = celery_app.control.inspect(timeout=1.0).ping() or {}
        workers = sorted(replies.keys())
    except (CeleryError, OSError, TimeoutError) as exc:
        logger.warning("Celery worker health check failed: %s", exc)
        workers = []

    if not workers:
        logger.warning("Celery broker is reachable but no worker responded to ping.")
        return {
            "status": "worker_unavailable",
            "broker_reachable": True,
            "worker_reachable": False,
            "workers": [],
        }

    return {
        "status": "ready",
        "broker_reachable": True,
        "worker_reachable": True,
        "workers": workers,
    }


def enqueue_retraining() -> dict[str, Any]:
    health = get_worker_health()
    if not health["broker_reachable"]:
        return {**health, "task_queued": False, "task_id": None}

    try:
        result = celery_app.send_task("app.tasks.jobs.retrain_ml_model_task")
    except (CeleryError, OperationalError, OSError, TimeoutError) as exc:
        logger.warning("ML retraining task could not be queued: %s", exc)
        return {
            **health,
            "status": "queue_failed",
            "task_queued": False,
            "task_id": None,
        }

    return {**health, "task_queued": True, "task_id": result.id}
