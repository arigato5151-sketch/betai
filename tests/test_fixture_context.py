from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.fixture_context import FixtureContextService


class MemoryContextCache:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], Any] = {}
        self.set_calls = 0

    async def get(self, category: str, key: str) -> Any | None:
        return self.values.get((category, key))

    async def set(
        self,
        category: str,
        key: str,
        value: Any,
        ttl: int,
    ) -> None:
        assert ttl > 0
        self.set_calls += 1
        self.values[(category, key)] = value


@pytest.mark.asyncio
async def test_fixture_context_reuses_timestamped_snapshot() -> None:
    context_cache = MemoryContextCache()
    service = FixtureContextService(context_cache)
    loader = AsyncMock(
        return_value={
            "fixture": {"fixture_id": 10, "source": "api_football"},
            "home_team": "Home",
        }
    )
    enricher = Mock(side_effect=lambda payload: {**payload, "odd": 2.1})
    now = datetime(2030, 8, 1, 12, tzinfo=UTC)

    first = await service.get_or_create(10, loader=loader, enricher=enricher, now=now)
    second = await service.get_or_create(10, loader=loader, enricher=enricher, now=now)

    assert first is not None and second is not None
    assert first["context_snapshot"]["cached"] is False
    assert second["context_snapshot"]["cached"] is True
    assert first["context_snapshot"]["generated_at"] == now.isoformat()
    assert first["context_snapshot"]["source"] == "api_football"
    loader.assert_awaited_once_with(10)
    enricher.assert_called_once()
    assert context_cache.set_calls == 1


@pytest.mark.asyncio
async def test_fixture_context_coalesces_concurrent_provider_reads() -> None:
    context_cache = MemoryContextCache()
    service = FixtureContextService(context_cache)

    async def load(fixture_id: int) -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"fixture": {"fixture_id": fixture_id}}

    loader = AsyncMock(side_effect=load)
    enricher = Mock(side_effect=lambda payload: payload)

    results = await asyncio.gather(
        *(service.get_or_create(10, loader=loader, enricher=enricher) for _ in range(5))
    )

    assert all(result is not None for result in results)
    loader.assert_awaited_once_with(10)
    assert context_cache.set_calls == 1


@pytest.mark.asyncio
async def test_fixture_context_does_not_cache_missing_fixture() -> None:
    context_cache = MemoryContextCache()
    service = FixtureContextService(context_cache)
    loader = AsyncMock(return_value=None)
    enricher = Mock()

    result = await service.get_or_create(10, loader=loader, enricher=enricher)

    assert result is None
    enricher.assert_not_called()
    assert context_cache.set_calls == 0
