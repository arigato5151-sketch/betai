from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Mapping

import httpx

from app.core.config import settings

MAX_PAYLOAD_BYTES = 4 * 1024 * 1024

UNDERSTAT_LEAGUES: Mapping[int, str] = MappingProxyType(
    {
        39: "EPL",
        61: "Ligue_1",
        78: "Bundesliga",
        135: "Serie_A",
        140: "La_liga",
    }
)


class UnderstatError(RuntimeError):
    """Base error for Understat enrichment."""


class UnderstatDownloadError(UnderstatError):
    """Raised when the provider endpoint cannot be downloaded."""


class UnderstatFormatError(UnderstatError):
    """Raised when the provider response violates its expected schema."""


@dataclass(frozen=True, slots=True)
class UnderstatFixtureXG:
    provider_match_id: str
    kickoff: datetime
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float


class UnderstatClient:
    """Small, isolated adapter around Understat's league JSON endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or settings.UNDERSTAT_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.UNDERSTAT_TIMEOUT_SECONDS
        self.transport = transport

    @property
    def supported_league_ids(self) -> frozenset[int]:
        return frozenset(UNDERSTAT_LEAGUES)

    async def get_completed_fixture_xg(
        self,
        league_id: int,
        season: int,
    ) -> list[UnderstatFixtureXG]:
        league = UNDERSTAT_LEAGUES.get(league_id)
        if league is None:
            raise ValueError(f"Unsupported Understat league_id: {league_id}")
        if season < 2014 or season > datetime.now(UTC).year + 1:
            raise ValueError(f"Invalid Understat season: {season}")

        payload = await self._download(league, season)
        return self._parse(payload)

    async def _download(self, league: str, season: int) -> object:
        page_url = f"{self.base_url}/league/{league}/{season}"
        endpoint = f"{self.base_url}/getLeagueData/{league}/{season}"
        headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": page_url,
            "User-Agent": "BetAIPlatform/1.0 xg-enrichment",
            "X-Requested-With": "XMLHttpRequest",
        }
        last_error: Exception | None = None
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=False,
            transport=self.transport,
            headers=headers,
        ) as client:
            for attempt in range(3):
                try:
                    response = await client.get(endpoint)
                    response.raise_for_status()
                    if len(response.content) > MAX_PAYLOAD_BYTES:
                        raise UnderstatFormatError("Understat payload is too large")
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise UnderstatFormatError(
                            "Understat payload is not valid JSON"
                        ) from exc
                except UnderstatFormatError:
                    raise
                except httpx.HTTPError as exc:
                    last_error = exc
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (2**attempt))
        raise UnderstatDownloadError(
            "Understat league data is unavailable"
        ) from last_error

    @classmethod
    def _parse(cls, payload: object) -> list[UnderstatFixtureXG]:
        if not isinstance(payload, dict) or not isinstance(payload.get("dates"), list):
            raise UnderstatFormatError("Understat payload must contain a dates list")

        fixtures: list[UnderstatFixtureXG] = []
        seen_ids: set[str] = set()
        for index, raw in enumerate(payload["dates"], start=1):
            if not isinstance(raw, dict) or raw.get("isResult") is not True:
                continue
            try:
                match_id = cls._text(raw.get("id"), "id")
                home = cls._team_name(raw.get("h"), "home team")
                away = cls._team_name(raw.get("a"), "away team")
                goals = raw.get("goals")
                xg = raw.get("xG")
                if not isinstance(goals, dict) or not isinstance(xg, dict):
                    raise ValueError("goals and xG must be objects")
                home_goals = cls._non_negative_int(goals.get("h"), "home goals")
                away_goals = cls._non_negative_int(goals.get("a"), "away goals")
                home_xg = cls._xg(xg.get("h"), "home xG")
                away_xg = cls._xg(xg.get("a"), "away xG")
                kickoff = datetime.strptime(
                    cls._text(raw.get("datetime"), "datetime"),
                    "%Y-%m-%d %H:%M:%S",
                ).replace(tzinfo=UTC)
            except (TypeError, ValueError) as exc:
                raise UnderstatFormatError(
                    f"Invalid Understat fixture at row {index}"
                ) from exc
            if match_id in seen_ids:
                raise UnderstatFormatError(f"Duplicate Understat match id: {match_id}")
            if home == away:
                raise UnderstatFormatError(f"Invalid teams at row {index}")
            seen_ids.add(match_id)
            fixtures.append(
                UnderstatFixtureXG(
                    provider_match_id=match_id,
                    kickoff=kickoff,
                    home_team=home,
                    away_team=away,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    home_xg=home_xg,
                    away_xg=away_xg,
                )
            )
        return fixtures

    @staticmethod
    def _text(value: object, label: str) -> str:
        text = str(value or "").strip()
        if not text or len(text) > 150:
            raise ValueError(f"invalid {label}")
        return text

    @classmethod
    def _team_name(cls, value: object, label: str) -> str:
        if not isinstance(value, dict):
            raise ValueError(f"invalid {label}")
        return cls._text(value.get("title"), label)

    @staticmethod
    def _non_negative_int(value: object, label: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"invalid {label}")
        parsed = int(str(value))
        if parsed < 0:
            raise ValueError(f"invalid {label}")
        return parsed

    @staticmethod
    def _xg(value: object, label: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"invalid {label}")
        parsed = float(str(value))
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 15.0:
            raise ValueError(f"invalid {label}")
        return parsed
