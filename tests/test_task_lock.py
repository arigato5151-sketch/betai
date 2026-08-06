from unittest.mock import Mock

import redis

from app.services.task_lock import DistributedTaskLock, RELEASE_SCRIPT


def test_distributed_lock_acquires_and_releases_only_its_token() -> None:
    client = Mock()
    client.set.return_value = True

    with DistributedTaskLock("training", ttl_seconds=600, client=client) as lock:
        assert lock.acquired is True
        token = lock.token

    client.set.assert_called_once_with(
        "bet_ai:task_lock:training",
        token,
        nx=True,
        ex=600,
    )
    client.eval.assert_called_once_with(
        RELEASE_SCRIPT,
        1,
        "bet_ai:task_lock:training",
        token,
    )


def test_distributed_lock_does_not_release_unowned_lock() -> None:
    client = Mock()
    client.set.return_value = None

    with DistributedTaskLock("training", ttl_seconds=600, client=client) as lock:
        assert lock.acquired is False
        assert lock.available is True

    client.eval.assert_not_called()


def test_distributed_lock_fails_closed_when_redis_is_unavailable() -> None:
    client = Mock()
    client.set.side_effect = redis.ConnectionError("offline")

    with DistributedTaskLock("training", ttl_seconds=600, client=client) as lock:
        assert lock.acquired is False
        assert lock.available is False

    client.eval.assert_not_called()
