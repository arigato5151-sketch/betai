from unittest.mock import AsyncMock

import pytest

from app.services.cache import TieredCache


@pytest.mark.asyncio
async def test_set_falls_back_from_redis_to_memcached() -> None:
    cache = TieredCache()
    cache.redis_client = AsyncMock()
    cache.redis_client.set.side_effect = RuntimeError("redis offline")
    cache.memcached_client = AsyncMock()

    await cache.set("fixtures", "42", {"status": "live"}, ttl=60)

    cache.memcached_client.set.assert_awaited_once_with(
        b"bet_ai:fixtures:42", b'{"status": "live"}', exptime=60
    )
    assert cache.status()["active_layer"] == "memcached"


@pytest.mark.asyncio
async def test_get_falls_back_from_redis_to_memcached() -> None:
    cache = TieredCache()
    cache.redis_client = AsyncMock()
    cache.redis_client.get.side_effect = RuntimeError("redis offline")
    cache.memcached_client = AsyncMock()
    cache.memcached_client.get.return_value = b'{"status": "live"}'

    result = await cache.get("fixtures", "42")

    assert result == {"status": "live"}


@pytest.mark.asyncio
async def test_local_cache_remains_last_fallback() -> None:
    cache = TieredCache()

    await cache.set("fixtures", "42", {"status": "local"}, ttl=60)

    assert await cache.get("fixtures", "42") == {"status": "local"}
    assert cache.status() == {
        "active_layer": "local",
        "distributed": False,
        "redis_available": False,
        "memcached_available": False,
        "status": "degraded",
    }


@pytest.mark.asyncio
async def test_set_falls_back_to_local_when_both_distributed_layers_fail() -> None:
    cache = TieredCache()
    cache.redis_client = AsyncMock()
    cache.redis_client.set.side_effect = RuntimeError("redis offline")
    cache.memcached_client = AsyncMock()
    cache.memcached_client.set.side_effect = RuntimeError("memcached offline")

    await cache.set("odds", "7", {"odd": 2.1}, ttl=30)

    assert await cache.get("odds", "7") == {"odd": 2.1}
    assert cache.status()["active_layer"] == "local"


@pytest.mark.asyncio
async def test_delete_removes_all_available_cache_layers() -> None:
    cache = TieredCache()
    cache.redis_client = AsyncMock()
    cache.memcached_client = AsyncMock()
    cache.local_cache["key"] = "value"

    await cache.delete("default", "key")

    cache.redis_client.delete.assert_awaited_once_with("bet_ai:default:key")
    cache.memcached_client.delete.assert_awaited_once_with(b"bet_ai:default:key")
    assert "key" not in cache.local_cache


@pytest.mark.asyncio
async def test_close_closes_connected_clients() -> None:
    cache = TieredCache()
    cache.redis_client = AsyncMock()
    cache.memcached_client = AsyncMock()

    await cache.close()

    cache.redis_client.aclose.assert_awaited_once_with()
    cache.memcached_client.close.assert_awaited_once_with()
