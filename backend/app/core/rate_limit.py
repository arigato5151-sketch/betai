from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any

import redis

from app.core.config import settings

logger = logging.getLogger("bet-ai-pro.security.rate-limit")


class LoginRateLimiter:
    """Redis-backed login guard with a process-local fail-safe fallback."""

    def __init__(self, redis_client: Any | None = None) -> None:
        self.redis_client: Any = redis_client or redis.Redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            decode_responses=True,
        )
        self._local: dict[str, tuple[int, float, float]] = {}
        self._lock = threading.Lock()
        self._use_redis = True

    @staticmethod
    def _digest(identifier: str, ip_address: str) -> str:
        value = f"{identifier.strip().lower()}|{ip_address}".encode()
        return hashlib.sha256(value).hexdigest()

    def retry_after(self, identifier: str, ip_address: str) -> int:
        key = self._digest(identifier, ip_address)
        try:
            if not self._use_redis:
                return self._local_retry_after(key)
            ttl = int(self.redis_client.ttl(f"bet_ai:login:lock:{key}"))
            return max(0, ttl)
        except redis.RedisError as exc:
            self._use_redis = False
            logger.warning(
                "Redis login limiter unavailable; using local guard: %s", exc
            )
            return self._local_retry_after(key)

    def record_failure(self, identifier: str, ip_address: str) -> None:
        key = self._digest(identifier, ip_address)
        attempts_key = f"bet_ai:login:attempts:{key}"
        lock_key = f"bet_ai:login:lock:{key}"
        try:
            if not self._use_redis:
                self._local_failure(key)
                return
            count = int(self.redis_client.incr(attempts_key))
            if count == 1:
                self.redis_client.expire(attempts_key, settings.LOGIN_WINDOW_SECONDS)
            if count >= settings.LOGIN_MAX_ATTEMPTS:
                self.redis_client.setex(lock_key, settings.LOGIN_LOCKOUT_SECONDS, "1")
            return
        except redis.RedisError as exc:
            self._use_redis = False
            logger.warning(
                "Redis login limiter write failed; using local guard: %s", exc
            )
        self._local_failure(key)

    def reset(self, identifier: str, ip_address: str) -> None:
        key = self._digest(identifier, ip_address)
        if self._use_redis:
            try:
                self.redis_client.delete(
                    f"bet_ai:login:attempts:{key}", f"bet_ai:login:lock:{key}"
                )
            except redis.RedisError:
                self._use_redis = False
                logger.debug("Redis login limiter reset failed.", exc_info=True)
        with self._lock:
            self._local.pop(key, None)

    def _local_retry_after(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            state = self._local.get(key)
            if not state:
                return 0
            _, _, locked_until = state
            if locked_until <= now:
                return 0
            return max(1, int(locked_until - now))

    def _local_failure(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            count, started_at, locked_until = self._local.get(key, (0, now, 0.0))
            if now - started_at >= settings.LOGIN_WINDOW_SECONDS:
                count, started_at = 0, now
            count += 1
            if count >= settings.LOGIN_MAX_ATTEMPTS:
                locked_until = now + settings.LOGIN_LOCKOUT_SECONDS
            self._local[key] = (count, started_at, locked_until)


login_rate_limiter = LoginRateLimiter()
