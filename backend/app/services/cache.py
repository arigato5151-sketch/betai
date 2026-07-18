import asyncio
import json
import logging
from typing import Any

import aiomcache
import redis.asyncio as aioredis
from cachetools import TTLCache

from app.core.config import settings

logger = logging.getLogger("bet-ai-pro.cache")


class TieredCache:
    def __init__(self) -> None:
        self.redis_client: aioredis.Redis | None = None
        self.memcached_client: aiomcache.Client | None = None
        self.local_cache: TTLCache[str, Any] = TTLCache(maxsize=1024, ttl=86400)
        self.local_fixtures_cache: TTLCache[str, Any] = TTLCache(maxsize=128, ttl=900)
        self.local_match_data_cache: TTLCache[str, Any] = TTLCache(
            maxsize=256, ttl=1800
        )
        self.local_odds_cache: TTLCache[str, Any] = TTLCache(maxsize=256, ttl=300)
        self.local_stats_cache: TTLCache[str, Any] = TTLCache(maxsize=512, ttl=1800)
        self.local_h2h_cache: TTLCache[str, Any] = TTLCache(maxsize=256, ttl=21600)

    async def connect(self) -> None:
        await self._connect_redis()
        await self._connect_memcached()
        if not self.redis_client and not self.memcached_client:
            logger.warning(
                "Distributed cache unavailable. Using process-local TTL cache."
            )

    async def _connect_redis(self) -> None:
        try:
            client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=2.0,
            )
            await client.ping()
            self.redis_client = client
            logger.info("Connected successfully to Redis caching layer.")
        except Exception as exc:
            self.redis_client = None
            logger.warning("Redis cache connection failed: %s", exc)

    async def _connect_memcached(self) -> None:
        if not settings.MEMCACHED_HOST:
            self.memcached_client = None
            return
        try:
            client = aiomcache.Client(
                settings.MEMCACHED_HOST,
                settings.MEMCACHED_PORT,
                pool_minsize=1,
                pool_size=10,
            )
            await asyncio.wait_for(
                client.version(), timeout=settings.MEMCACHED_TIMEOUT_SECONDS
            )
            self.memcached_client = client
            logger.info("Connected successfully to Memcached fallback layer.")
        except Exception as exc:
            self.memcached_client = None
            logger.warning("Memcached cache connection failed: %s", exc)

    def status(self) -> dict[str, Any]:
        active_layer = (
            "redis"
            if self.redis_client
            else "memcached" if self.memcached_client else "local"
        )
        return {
            "active_layer": active_layer,
            "distributed": active_layer != "local",
            "redis_available": self.redis_client is not None,
            "memcached_available": self.memcached_client is not None,
            "status": "ready" if active_layer != "local" else "degraded",
        }

    async def _drop_redis(self) -> None:
        client, self.redis_client = self.redis_client, None
        if client:
            try:
                await client.aclose()
            except Exception:
                logger.debug("Redis client close failed.", exc_info=True)

    async def _drop_memcached(self) -> None:
        client, self.memcached_client = self.memcached_client, None
        if client:
            try:
                await client.close()
            except Exception:
                logger.debug("Memcached client close failed.", exc_info=True)

    def _get_local_cache(self, category: str) -> TTLCache[str, Any]:
        caches = {
            "fixtures": self.local_fixtures_cache,
            "match_data": self.local_match_data_cache,
            "odds": self.local_odds_cache,
            "stats": self.local_stats_cache,
            "h2h": self.local_h2h_cache,
        }
        return caches.get(category, self.local_cache)

    @staticmethod
    def _cache_key(category: str, key: str) -> str:
        return f"bet_ai:{category}:{key}"

    async def get(self, category: str, key: str) -> Any | None:
        cache_key = self._cache_key(category, key)
        if self.redis_client:
            try:
                data = await self.redis_client.get(cache_key)
                if data is not None:
                    return json.loads(data)
            except Exception as exc:
                logger.warning("Redis get failed for %s: %s", cache_key, exc)
                await self._drop_redis()

        if self.memcached_client:
            try:
                data = await asyncio.wait_for(
                    self.memcached_client.get(cache_key.encode()),
                    timeout=settings.MEMCACHED_TIMEOUT_SECONDS,
                )
                if data is not None:
                    return json.loads(data.decode())
            except Exception as exc:
                logger.warning("Memcached get failed for %s: %s", cache_key, exc)
                await self._drop_memcached()

        return self._get_local_cache(category).get(key)

    async def set(self, category: str, key: str, value: Any, ttl: int) -> None:
        cache_key = self._cache_key(category, key)
        serialized = json.dumps(value)
        if self.redis_client:
            try:
                await self.redis_client.set(cache_key, serialized, ex=ttl)
                return
            except Exception as exc:
                logger.warning("Redis set failed for %s: %s", cache_key, exc)
                await self._drop_redis()

        if self.memcached_client:
            try:
                await asyncio.wait_for(
                    self.memcached_client.set(
                        cache_key.encode(), serialized.encode(), exptime=ttl
                    ),
                    timeout=settings.MEMCACHED_TIMEOUT_SECONDS,
                )
                return
            except Exception as exc:
                logger.warning("Memcached set failed for %s: %s", cache_key, exc)
                await self._drop_memcached()

        self._get_local_cache(category)[key] = value

    async def delete(self, category: str, key: str) -> None:
        cache_key = self._cache_key(category, key)
        if self.redis_client:
            try:
                await self.redis_client.delete(cache_key)
            except Exception as exc:
                logger.warning("Redis delete failed for %s: %s", cache_key, exc)
                await self._drop_redis()
        if self.memcached_client:
            try:
                await asyncio.wait_for(
                    self.memcached_client.delete(cache_key.encode()),
                    timeout=settings.MEMCACHED_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                logger.warning("Memcached delete failed for %s: %s", cache_key, exc)
                await self._drop_memcached()
        self._get_local_cache(category).pop(key, None)

    async def close(self) -> None:
        if self.redis_client:
            await self.redis_client.aclose()
        if self.memcached_client:
            await self.memcached_client.close()


cache = TieredCache()
