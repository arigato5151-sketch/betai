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

logger = logging.getLogger("bet-ai-pro.the_odds_api")

LEAGUE_SPORT_KEYS: dict[int, str] = {
    39: "soccer_epl",
    40: "soccer_efl_champ",
    61: "soccer_france_ligue_one",
    62: "soccer_france_ligue_two",
    79: "soccer_germany_bundesliga2",
    88: "soccer_netherlands_eredivisie",
    94: "soccer_portugal_primeira_liga",
    135: "soccer_italy_serie_a",
    140: "soccer_spain_la_liga",
    144: "soccer_belgium_first_div",
    203: "soccer_turkey_super_league",
    218: "soccer_austria_bundesliga",
    235: "soccer_russia_premier_league",
}


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


class TheOddsApiClient:
    """Quota-conscious fallback for timestamped pre-match 1X2 markets."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = (
            settings.THE_ODDS_API_KEY if api_key is None else api_key
        ).strip()
        self.base_url = (base_url or settings.THE_ODDS_API_BASE_URL).rstrip("/")
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.THE_ODDS_API_TIMEOUT_SECONDS
        )
        self.transport = transport

    @property
    def configured(self) -> bool:
        return settings.THE_ODDS_API_ENABLED and bool(self.api_key)

    async def get_fixture_market(
        self, fixture: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if not self.configured:
            return None
        league_id = fixture.get("league_id")
        if not isinstance(league_id, int) or isinstance(league_id, bool):
            return None
        sport_key = LEAGUE_SPORT_KEYS.get(league_id)
        kickoff = _datetime(fixture.get("kickoff"))
        home_key = normalize_team_name(str(fixture.get("home_team") or ""))
        away_key = normalize_team_name(str(fixture.get("away_team") or ""))
        if not sport_key or kickoff is None or not home_key or not away_key:
            return None

        events = await self._league_events(sport_key)
        candidates = [
            event
            for event in events
            if normalize_team_name(str(event.get("home_team") or "")) == home_key
            and normalize_team_name(str(event.get("away_team") or "")) == away_key
            and (event_kickoff := _datetime(event.get("commence_time"))) is not None
            and abs((event_kickoff - kickoff).total_seconds()) <= 6 * 3600
        ]
        if len(candidates) != 1:
            return None
        return self._best_market(candidates[0], captured_at=datetime.now(UTC))

    async def _league_events(self, sport_key: str) -> list[dict[str, Any]]:
        cache_key = f"the-odds-api:v1:{sport_key}"
        cached = await cache.get("odds", cache_key)
        if isinstance(cached, list):
            return [row for row in cached if isinstance(row, dict)]

        url = f"{self.base_url}/v4/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
                headers={"Accept": "application/json"},
            ) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "The Odds API market request failed",
                extra={"sport_key": sport_key, "error_type": type(exc).__name__},
            )
            return []

        events = (
            [row for row in payload if isinstance(row, dict)]
            if isinstance(payload, list)
            else []
        )
        await cache.set(
            "odds",
            cache_key,
            events,
            settings.THE_ODDS_API_CACHE_SECONDS,
        )
        return events

    @staticmethod
    def _best_market(
        event: Mapping[str, Any], *, captured_at: datetime
    ) -> dict[str, Any] | None:
        home_name = str(event.get("home_team") or "")
        away_name = str(event.get("away_team") or "")
        home_key = normalize_team_name(home_name)
        away_key = normalize_team_name(away_name)
        best: dict[str, Any] | None = None

        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, list):
            return None
        for bookmaker in bookmakers:
            if not isinstance(bookmaker, Mapping):
                continue
            markets = bookmaker.get("markets")
            if not isinstance(markets, list):
                continue
            for market_payload in markets:
                if (
                    not isinstance(market_payload, Mapping)
                    or market_payload.get("key") != "h2h"
                ):
                    continue
                outcomes = market_payload.get("outcomes")
                if not isinstance(outcomes, list):
                    continue
                odds: dict[str, float] = {}
                for outcome in outcomes:
                    if not isinstance(outcome, Mapping):
                        continue
                    name = str(outcome.get("name") or "")
                    key = normalize_team_name(name)
                    if key == home_key:
                        outcome_key = "HOME_WIN"
                    elif key == away_key:
                        outcome_key = "AWAY_WIN"
                    elif key == "draw":
                        outcome_key = "DRAW"
                    else:
                        continue
                    try:
                        odds[outcome_key] = float(outcome["price"])
                    except (KeyError, TypeError, ValueError):
                        continue
                if len(odds) != 3:
                    continue
                try:
                    market = ValueCalc.devig_1x2(
                        odds["HOME_WIN"], odds["DRAW"], odds["AWAY_WIN"]
                    )
                except ValueError:
                    continue
                candidate = {
                    **market,
                    "bookmaker": str(
                        bookmaker.get("title") or bookmaker.get("key") or "unknown"
                    )[:100],
                    "source": "the_odds_api",
                    "provider_event_id": str(event.get("id") or "")[:100] or None,
                    "provider_last_update": market_payload.get("last_update")
                    or bookmaker.get("last_update"),
                    "captured_at": captured_at.isoformat(),
                }
                if best is None or float(candidate["overround_pct"]) < float(
                    best["overround_pct"]
                ):
                    best = candidate
        return best
