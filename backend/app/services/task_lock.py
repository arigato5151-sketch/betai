from __future__ import annotations

import logging
import uuid
from types import TracebackType
from typing import Protocol

import redis

from app.core.config import settings

logger = logging.getLogger("bet-ai-pro.task_lock")

RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


class SyncRedisClient(Protocol):
    def set(
        self,
        name: str,
        value: str,
        *,
        nx: bool,
        ex: int,
    ) -> object: ...

    def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object: ...

    def close(self) -> object: ...


class DistributedTaskLock:
    """Redis lease with token-checked release; fails closed when Redis is unavailable."""

    def __init__(
        self,
        name: str,
        *,
        ttl_seconds: int,
        client: SyncRedisClient | None = None,
    ) -> None:
        self.key = f"bet_ai:task_lock:{name}"
        self.ttl_seconds = ttl_seconds
        self.token = uuid.uuid4().hex
        self.client = client
        self.acquired = False
        self.available = True
        self._owns_client = client is None

    def __enter__(self) -> "DistributedTaskLock":
        try:
            if self.client is None:
                self.client = redis.Redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )
            self.acquired = bool(
                self.client.set(
                    self.key,
                    self.token,
                    nx=True,
                    ex=self.ttl_seconds,
                )
            )
        except redis.RedisError as exc:
            self.available = False
            self.acquired = False
            logger.error("Distributed task lock unavailable for %s: %s", self.key, exc)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.client is not None and self.acquired:
            try:
                self.client.eval(RELEASE_SCRIPT, 1, self.key, self.token)
            except redis.RedisError:
                logger.exception(
                    "Distributed task lock release failed for %s", self.key
                )
        if self.client is not None and self._owns_client:
            try:
                self.client.close()
            except redis.RedisError:
                logger.debug("Redis task-lock client close failed", exc_info=True)
