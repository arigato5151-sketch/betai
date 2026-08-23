from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import httpx

from app.core.config import settings
from app.core.team_identity import normalize_team_name
from app.prediction.value_calc import ValueCalc
from app.services.cache import cache

logger = logging.getLogger("bet-ai-pro.odds_api_io")


def _datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


class OddsApiIoClient:
    """Broad-coverage, quota-conscious fallback for pre-match 1X2 markets."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        bookmakers: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = (
            settings.ODDS_API_IO_KEY if api_key is None else api_key
        ).strip()
        self.base_url = (base_url or settings.ODDS_API_IO_BASE_URL).rstrip("/")
        configured_bookmakers = bookmakers or settings.ODDS_API_IO_BOOKMAKERS
        self.bookmakers = ",".join(
            item.strip() for item in configured_bookmakers.split(",") if item.strip()
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.ODDS_API_IO_TIMEOUT_SECONDS
        )
        self.transport = transport

    @property
    def configured(self) -> bool:
        return (
            settings.ODDS_API_IO_ENABLED
            and bool(self.api_key)
            and bool(self.bookmakers)
        )

    async def get_fixture_market(
        self, fixture: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if not self.configured:
            return None
        kickoff = _datetime(fixture.get("kickoff"))
        home_name = str(fixture.get("home_team") or "").strip()
        away_name = str(fixture.get("away_team") or "").strip()
        home_key = normalize_team_name(home_name)
        away_key = normalize_team_name(away_name)
        if kickoff is None or len(home_name) < 3 or not home_key or not away_key:
            return None

        events = await self._search_events(home_name, home_key)
        candidates = [
            event
            for event in events
            if normalize_team_name(str(event.get("home") or "")) == home_key
            and normalize_team_name(str(event.get("away") or "")) == away_key
            and (event_kickoff := _datetime(event.get("date"))) is not None
            and abs((event_kickoff - kickoff).total_seconds()) <= 6 * 3600
        ]
        if len(candidates) != 1:
            return None

        event_id = candidates[0].get("id")
        if not isinstance(event_id, (int, str)) or not str(event_id).strip():
            return None
        odds_payload = await self._event_odds(str(event_id))
        if not isinstance(odds_payload, dict):
            return None
        return self._best_market(odds_payload, captured_at=datetime.now(UTC))

    async def _search_events(
        self, query: str, normalized_query: str
    ) -> list[dict[str, Any]]:
        cache_key = f"odds-api-io:events:v1:{normalized_query}"
        cached = await cache.get("odds", cache_key)
        if isinstance(cached, list):
            return [row for row in cached if isinstance(row, dict)]

        payload = await self._get("/events/search", {"query": query})
        events = (
            [row for row in payload if isinstance(row, dict)]
            if isinstance(payload, list)
            else []
        )
        await cache.set("odds", cache_key, events, settings.ODDS_API_IO_CACHE_SECONDS)
        return events

    async def _event_odds(self, event_id: str) -> dict[str, Any] | None:
        bookmaker_key = normalize_team_name(self.bookmakers)
        cache_key = f"odds-api-io:market:v1:{event_id}:{bookmaker_key}"
        cached = await cache.get("odds", cache_key)
        if isinstance(cached, dict):
            return cached

        payload = await self._get(
            "/odds", {"eventId": event_id, "bookmakers": self.bookmakers}
        )
        if not isinstance(payload, dict):
            return None
        await cache.set("odds", cache_key, payload, settings.ODDS_API_IO_CACHE_SECONDS)
        return payload

    async def _get(self, path: str, params: dict[str, str]) -> object | None:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
                headers={"Accept": "application/json"},
            ) as client:
                response = await client.get(
                    f"{self.base_url}{path}",
                    params={"apiKey": self.api_key, **params},
                )
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "Odds-API.io request failed",
                extra={"path": path, "error_type": type(exc).__name__},
            )
            return None

    @staticmethod
    def _best_market(
        event: Mapping[str, Any], *, captured_at: datetime
    ) -> dict[str, Any] | None:
        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, Mapping):
            return None
        best: dict[str, Any] | None = None
        for bookmaker, markets in bookmakers.items():
            if not isinstance(markets, list):
                continue
            for market_payload in markets:
                if (
                    not isinstance(market_payload, Mapping)
                    or market_payload.get("name") != "ML"
                ):
                    continue
                rows = market_payload.get("odds")
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    try:
                        home = float(row["home"])
                        draw = float(row["draw"])
                        away = float(row["away"])
                        market = ValueCalc.devig_1x2(home, draw, away)
                    except (KeyError, TypeError, ValueError):
                        continue
                    candidate = {
                        **market,
                        "bookmaker": str(bookmaker)[:100],
                        "source": "odds_api_io",
                        "provider_event_id": str(event.get("id") or "")[:100] or None,
                        "provider_last_update": market_payload.get("updatedAt"),
                        "captured_at": captured_at.isoformat(),
                    }
                    if best is None or float(candidate["overround_pct"]) < float(
                        best["overround_pct"]
                    ):
                        best = candidate
        return best
