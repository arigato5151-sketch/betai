from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from fastapi.encoders import jsonable_encoder

from app.core.config import settings
from app.services.cache import cache


class ContextCache(Protocol):
    async def get(self, category: str, key: str) -> Any | None: ...

    async def set(
        self,
        category: str,
        key: str,
        value: Any,
        ttl: int,
    ) -> None: ...


class FixtureContextService:
    """Coalesce provider reads into a reusable, auditable fixture snapshot."""

    SNAPSHOT_VERSION = "fixture_context_v1"

    def __init__(self, cache_backend: ContextCache = cache) -> None:
        self.cache = cache_backend
        self._locks: dict[int, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _lock_for(self, fixture_id: int) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(fixture_id, asyncio.Lock())

    async def get_or_create(
        self,
        fixture_id: int,
        *,
        loader: Callable[[int], Awaitable[dict[str, Any] | None]],
        enricher: Callable[[dict[str, Any]], dict[str, Any]],
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        key = f"{self.SNAPSHOT_VERSION}:{fixture_id}"
        cached = await self.cache.get("fixture_context", key)
        if isinstance(cached, dict):
            return self._mark_cache_hit(cached)

        fixture_lock = await self._lock_for(fixture_id)
        try:
            async with fixture_lock:
                # A concurrent request may have populated the distributed cache.
                cached = await self.cache.get("fixture_context", key)
                if isinstance(cached, dict):
                    return self._mark_cache_hit(cached)

                prefill = await loader(fixture_id)
                if not isinstance(prefill, dict):
                    return None
                enriched = enricher(prefill)
                generated_at = (now or datetime.now(UTC)).astimezone(UTC)
                ttl = settings.FIXTURE_CONTEXT_SNAPSHOT_TTL_SECONDS
                fixture = enriched.get("fixture")
                source = fixture.get("source") if isinstance(fixture, dict) else None
                snapshot = jsonable_encoder(
                    {
                        **enriched,
                        "context_snapshot": {
                            "version": self.SNAPSHOT_VERSION,
                            "generated_at": generated_at,
                            "expires_at": generated_at + timedelta(seconds=ttl),
                            "source": source,
                            "cached": False,
                        },
                    }
                )
                await self.cache.set("fixture_context", key, snapshot, ttl)
                return deepcopy(snapshot)
        finally:
            async with self._locks_guard:
                if not fixture_lock.locked():
                    self._locks.pop(fixture_id, None)

    @staticmethod
    def _mark_cache_hit(snapshot: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(snapshot)
        metadata = result.get("context_snapshot")
        if isinstance(metadata, dict):
            metadata["cached"] = True
        return result


fixture_context_service = FixtureContextService()
