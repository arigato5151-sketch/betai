from unittest.mock import Mock

import redis

from app.core.config import settings
from app.core.rate_limit import LoginRateLimiter


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
