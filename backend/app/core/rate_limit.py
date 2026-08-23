from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import redis
from cachetools import TTLCache

from app.core.config import settings

logger = logging.getLogger("bet-ai-pro.security.rate-limit")


class LoginRateLimiter:
    """Redis-backed login guard with a process-local fail-safe fallback."""

    def __init__(
        self,
        redis_client: Any | None = None,
        *,
        clock: Callable[[], float] | None = None,
        redis_recovery_seconds: float | None = None,
        local_max_entries: int = 10_000,
    ) -> None:
        self.redis_client: Any = redis_client or redis.Redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            decode_responses=True,
        )
        self._clock = clock or time.monotonic
        self._redis_recovery_seconds = (
            redis_recovery_seconds
            if redis_recovery_seconds is not None
            else settings.LOGIN_REDIS_RECOVERY_SECONDS
        )
        self._local: TTLCache[str, tuple[int, float, float]] = TTLCache(
            maxsize=local_max_entries,
            ttl=max(
                settings.LOGIN_WINDOW_SECONDS,
                settings.LOGIN_LOCKOUT_SECONDS,
            ),
            timer=self._clock,
        )
        self._lock = threading.Lock()
        self._use_redis = True
        self._last_redis_retry = 0.0

    @staticmethod
    def _digest(identifier: str, ip_address: str) -> str:
        value = f"{identifier.strip().lower()}|{ip_address}".encode()
        return hashlib.sha256(value).hexdigest()

    def retry_after(self, identifier: str, ip_address: str) -> int:
        key = self._digest(identifier, ip_address)
        try:
            if not self._redis_available():
                return self._local_retry_after(key)
            ttl = int(self.redis_client.ttl(f"bet_ai:login:lock:{key}"))
            return max(0, ttl)
        except redis.RedisError as exc:
            self._mark_redis_unavailable()
            logger.warning(
                "Redis login limiter unavailable; using local guard: %s", exc
            )
            return self._local_retry_after(key)

    def record_failure(self, identifier: str, ip_address: str) -> None:
        key = self._digest(identifier, ip_address)
        attempts_key = f"bet_ai:login:attempts:{key}"
        lock_key = f"bet_ai:login:lock:{key}"
        try:
            if not self._redis_available():
                self._local_failure(key)
                return
            count = int(self.redis_client.incr(attempts_key))
            if count == 1:
                self.redis_client.expire(attempts_key, settings.LOGIN_WINDOW_SECONDS)
            if count >= settings.LOGIN_MAX_ATTEMPTS:
                self.redis_client.setex(lock_key, settings.LOGIN_LOCKOUT_SECONDS, "1")
            return
        except redis.RedisError as exc:
            self._mark_redis_unavailable()
            logger.warning(
                "Redis login limiter write failed; using local guard: %s", exc
            )
        self._local_failure(key)

    def reset(self, identifier: str, ip_address: str) -> None:
        key = self._digest(identifier, ip_address)
        if self._redis_available():
            try:
                self.redis_client.delete(
                    f"bet_ai:login:attempts:{key}", f"bet_ai:login:lock:{key}"
                )
            except redis.RedisError:
                self._mark_redis_unavailable()
                logger.debug("Redis login limiter reset failed.", exc_info=True)
        with self._lock:
            self._local.pop(key, None)

    def _redis_available(self) -> bool:
        with self._lock:
            if self._use_redis:
                return True
            now = self._clock()
            if now - self._last_redis_retry < self._redis_recovery_seconds:
                return False
            # Reserve this retry window so concurrent requests do not ping together.
            self._last_redis_retry = now

        try:
            self.redis_client.ping()
        except redis.RedisError:
            logger.debug("Redis login limiter recovery attempt failed.", exc_info=True)
            return False

        with self._lock:
            # Redis is authoritative again; stale process-local counters are discarded.
            self._local.clear()
            self._use_redis = True
        logger.info("Redis login limiter connection recovered.")
        return True

    def _mark_redis_unavailable(self) -> None:
        with self._lock:
            self._use_redis = False
            self._last_redis_retry = self._clock()

    def _local_retry_after(self, key: str) -> int:
        now = self._clock()
        with self._lock:
            state = self._local.get(key)
            if not state:
                return 0
            _, _, locked_until = state
            if locked_until <= now:
                return 0
            return max(1, int(locked_until - now))

    def _local_failure(self, key: str) -> None:
        now = self._clock()
        with self._lock:
            count, started_at, locked_until = self._local.get(key, (0, now, 0.0))
            if now - started_at >= settings.LOGIN_WINDOW_SECONDS:
                count, started_at = 0, now
            count += 1
            if count >= settings.LOGIN_MAX_ATTEMPTS:
                locked_until = now + settings.LOGIN_LOCKOUT_SECONDS
            self._local[key] = (count, started_at, locked_until)


class FixedWindowRateLimiter:
    """Redis-authoritative request quota with a bounded local outage fallback."""

    def __init__(
        self,
        namespace: str,
        *,
        redis_client: Any | None = None,
        clock: Callable[[], float] | None = None,
        local_max_entries: int = 10_000,
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> None:
        self.namespace = namespace
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self.redis_client: Any = redis_client or redis.Redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            decode_responses=True,
        )
        self._clock = clock or time.monotonic
        self._local: TTLCache[str, int] = TTLCache(
            maxsize=local_max_entries,
            ttl=self._request_window_seconds,
            timer=self._clock,
        )
        self._lock = threading.Lock()

    def consume(self, subject: str) -> tuple[bool, int]:
        key = hashlib.sha256(subject.encode("utf-8")).hexdigest()
        redis_key = f"bet_ai:{self.namespace}:{key}"
        try:
            count = int(self.redis_client.incr(redis_key))
            if count == 1:
                self.redis_client.expire(
                    redis_key, self._request_window_seconds
                )
            if count <= self._max_request_count:
                return True, 0
            ttl = int(self.redis_client.ttl(redis_key))
            return False, max(1, ttl)
        except redis.RedisError as exc:
            logger.warning("Redis quota unavailable; using local fallback: %s", exc)

        with self._lock:
            count = self._local.get(key, 0) + 1
            self._local[key] = count
        if count <= self._max_request_count:
            return True, 0
        return False, self._request_window_seconds

    @property
    def _max_request_count(self) -> int:
        return (
            self._max_requests
            if self._max_requests is not None
            else settings.BATCH_PREDICTION_MAX_REQUESTS
        )

    @property
    def _request_window_seconds(self) -> int:
        return (
            self._window_seconds
            if self._window_seconds is not None
            else settings.BATCH_PREDICTION_WINDOW_SECONDS
        )


login_rate_limiter = LoginRateLimiter()
batch_prediction_rate_limiter = FixedWindowRateLimiter("batch_prediction")
registration_rate_limiter = FixedWindowRateLimiter(
    "self_registration",
    max_requests=settings.REGISTRATION_MAX_REQUESTS,
    window_seconds=settings.REGISTRATION_WINDOW_SECONDS,
)
