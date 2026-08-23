from unittest.mock import Mock

import redis

from app.core.config import settings
from app.core.rate_limit import FixedWindowRateLimiter, LoginRateLimiter


def test_login_limiter_locks_after_configured_failures(monkeypatch) -> None:
    client = Mock()
    client.ttl.side_effect = redis.ConnectionError("offline")
    client.incr.side_effect = redis.ConnectionError("offline")
    limiter = LoginRateLimiter(redis_client=client)
    monkeypatch.setattr(settings, "LOGIN_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(settings, "LOGIN_LOCKOUT_SECONDS", 60)

    for _ in range(3):
        limiter.record_failure("user", "127.0.0.1")

    assert limiter.retry_after("user", "127.0.0.1") > 0
    limiter.reset("user", "127.0.0.1")
    assert limiter.retry_after("user", "127.0.0.1") == 0


def test_login_limiter_uses_redis_expiry_and_lock_keys(monkeypatch) -> None:
    client = Mock()
    client.ttl.return_value = 0
    client.incr.side_effect = [1, 2]
    limiter = LoginRateLimiter(redis_client=client)
    monkeypatch.setattr(settings, "LOGIN_MAX_ATTEMPTS", 2)

    limiter.record_failure("user", "127.0.0.1")
    limiter.record_failure("user", "127.0.0.1")

    client.expire.assert_called_once()
    client.setex.assert_called_once()


def test_login_limiter_recovers_redis_after_temporary_outage(monkeypatch) -> None:
    now = [100.0]
    client = Mock()
    client.ping.return_value = True
    client.ttl.return_value = 0
    client.incr.side_effect = [redis.ConnectionError("offline"), 1]
    limiter = LoginRateLimiter(
        redis_client=client,
        clock=lambda: now[0],
        redis_recovery_seconds=30,
    )
    monkeypatch.setattr(settings, "LOGIN_MAX_ATTEMPTS", 2)

    limiter.record_failure("user", "127.0.0.1")
    assert limiter._use_redis is False

    now[0] += 29
    assert limiter.retry_after("user", "127.0.0.1") == 0
    client.ping.assert_not_called()

    now[0] += 1
    assert limiter.retry_after("user", "127.0.0.1") == 0
    client.ping.assert_called_once_with()
    assert limiter._use_redis is True

    limiter.record_failure("user", "127.0.0.1")
    client.expire.assert_called_once()


def test_local_fallback_has_a_bounded_number_of_identifiers() -> None:
    client = Mock()
    client.incr.side_effect = redis.ConnectionError("offline")
    limiter = LoginRateLimiter(redis_client=client, local_max_entries=2)

    limiter.record_failure("user-1", "127.0.0.1")
    limiter.record_failure("user-2", "127.0.0.1")
    limiter.record_failure("user-3", "127.0.0.1")

    assert len(limiter._local) == 2


def test_fixed_window_limiter_is_redis_authoritative(monkeypatch) -> None:
    client = Mock()
    client.incr.side_effect = [1, 2, 3]
    client.ttl.return_value = 42
    limiter = FixedWindowRateLimiter("batch", redis_client=client)
    monkeypatch.setattr(settings, "BATCH_PREDICTION_MAX_REQUESTS", 2)
    monkeypatch.setattr(settings, "BATCH_PREDICTION_WINDOW_SECONDS", 60)

    assert limiter.consume("user-1") == (True, 0)
    assert limiter.consume("user-1") == (True, 0)
    assert limiter.consume("user-1") == (False, 42)
    client.expire.assert_called_once()


def test_fixed_window_limiter_supports_independent_registration_quota() -> None:
    client = Mock()
    client.incr.side_effect = [1, 2]
    client.ttl.return_value = 45
    limiter = FixedWindowRateLimiter(
        "registration",
        redis_client=client,
        max_requests=1,
        window_seconds=45,
    )

    assert limiter.consume("203.0.113.1") == (True, 0)
    assert limiter.consume("203.0.113.1") == (False, 45)
    client.expire.assert_called_once()
