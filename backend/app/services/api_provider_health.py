from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from app.core.config import settings
from app.services.cache import cache

_HEALTH_KEY = "api_football"


def _now() -> datetime:
    return datetime.now(UTC)


def _header_int(headers: Mapping[str, str], name: str) -> int | None:
    raw = next(
        (value for key, value in headers.items() if key.lower() == name.lower()), None
    )
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


class APIFootballHealthTracker:
    """Tracks quota and circuit state; Redis makes it visible across processes."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._state = self._empty_state()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "provider": "api_football",
            "status": "unknown",
            "circuit_open_until": None,
            "consecutive_failures": 0,
            "daily_limit": None,
            "daily_remaining": None,
            "minute_limit": None,
            "minute_remaining": None,
            "last_status_code": None,
            "last_success_at": None,
            "last_failure_at": None,
            "last_error": None,
            "updated_at": None,
        }

    async def _load_shared(self) -> None:
        shared = await cache.get("operations", _HEALTH_KEY)
        if isinstance(shared, dict):
            self._state.update(shared)

    async def _persist(self) -> None:
        self._state["updated_at"] = _now().isoformat()
        await cache.set("operations", _HEALTH_KEY, self._state, 172800)

    async def allow_request(self) -> bool:
        async with self._lock:
            await self._load_shared()
            open_until_raw = self._state.get("circuit_open_until")
            if not open_until_raw:
                return True
            try:
                open_until = datetime.fromisoformat(str(open_until_raw))
            except ValueError:
                self._state["circuit_open_until"] = None
                await self._persist()
                return True
            if open_until > _now():
                self._state["status"] = "circuit_open"
                return False
            self._state.update(
                status="half_open",
                circuit_open_until=None,
                consecutive_failures=0,
            )
            await self._persist()
            return True

    async def record_response(
        self, status_code: int, headers: Mapping[str, str], error: str | None = None
    ) -> None:
        async with self._lock:
            quota = {
                "daily_limit": _header_int(headers, "x-ratelimit-requests-limit"),
                "daily_remaining": _header_int(
                    headers, "x-ratelimit-requests-remaining"
                ),
                "minute_limit": _header_int(headers, "x-ratelimit-limit"),
                "minute_remaining": _header_int(headers, "x-ratelimit-remaining"),
            }
            self._state.update(
                {key: value for key, value in quota.items() if value is not None}
            )
            self._state["last_status_code"] = status_code
            if status_code == 200:
                self._state.update(
                    status="ready",
                    consecutive_failures=0,
                    circuit_open_until=None,
                    last_success_at=_now().isoformat(),
                    last_error=None,
                )
            else:
                failures = int(self._state.get("consecutive_failures") or 0) + 1
                self._state.update(
                    status="degraded",
                    consecutive_failures=failures,
                    last_failure_at=_now().isoformat(),
                    last_error=error,
                )
                if failures >= settings.API_FOOTBALL_CIRCUIT_FAILURE_THRESHOLD:
                    self._open_circuit(settings.API_FOOTBALL_CIRCUIT_OPEN_SECONDS)
            await self._persist()

    async def record_rate_limit(
        self, headers: Mapping[str, str], retry_after: int
    ) -> None:
        async with self._lock:
            self._state.update(
                daily_limit=_header_int(headers, "x-ratelimit-requests-limit"),
                daily_remaining=_header_int(headers, "x-ratelimit-requests-remaining"),
                minute_limit=_header_int(headers, "x-ratelimit-limit"),
                minute_remaining=_header_int(headers, "x-ratelimit-remaining"),
                last_status_code=429,
                last_failure_at=_now().isoformat(),
                last_error="rate_limited",
                consecutive_failures=int(self._state.get("consecutive_failures") or 0)
                + 1,
            )
            self._open_circuit(
                max(retry_after, settings.API_FOOTBALL_CIRCUIT_OPEN_SECONDS)
            )
            await self._persist()

    async def record_transport_failure(self, error: str) -> None:
        await self.record_response(0, {}, error)

    def _open_circuit(self, seconds: int) -> None:
        self._state["status"] = "circuit_open"
        self._state["circuit_open_until"] = (
            _now() + timedelta(seconds=seconds)
        ).isoformat()

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            await self._load_shared()
            return dict(self._state)

    def reset_for_test(self) -> None:
        self._state = self._empty_state()
        cache.local_cache.pop(_HEALTH_KEY, None)


api_football_health = APIFootballHealthTracker()
