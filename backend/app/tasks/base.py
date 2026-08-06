from __future__ import annotations

import httpx
import redis
from celery import Task
from sqlalchemy.exc import OperationalError


class TransientTask(Task):
    """Retry transport/infrastructure failures without retrying validation bugs."""

    autoretry_for = (
        ConnectionError,
        TimeoutError,
        httpx.TimeoutException,
        redis.RedisError,
        OperationalError,
    )
    dont_autoretry_for = (TypeError, ValueError)
    max_retries = 3
    retry_backoff = 30
    retry_backoff_max = 600
    retry_jitter = True
