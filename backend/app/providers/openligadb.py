from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

import httpx

from app.core.config import settings

MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
ID_OFFSET = 500_000_000
MAX_SOURCE_ID = 199_999_999
LEAGUE_SHORTCUTS = {78: "bl1", 79: "bl2", 2: "ucl"}
LEAGUE_NAMES = {78: "Bundesliga", 79: "2. Bundesliga", 2: "UEFA Champions League"}


class OpenLigaDBError(RuntimeError):
    """Raised when OpenLigaDB cannot provide a valid fixture payload."""


def _namespace_id(value: object) -> int | None:
    try:
        source_id = int(str(value))
    except (TypeError, ValueError):
        return None
    if source_id <= 0 or source_id > MAX_SOURCE_ID:
        return None
    return ID_OFFSET + source_id


def is_openligadb_fixture_id(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and ID_OFFSET < value <= ID_OFFSET + MAX_SOURCE_ID
    )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


class OpenLigaDBClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.base_url = (base_url or settings.OPENLIGADB_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.OPENLIGADB_TIMEOUT_SECONDS
        self.transport = transport
        self.enabled = settings.OPENLIGADB_ENABLED if enabled is None else enabled

    @property
    def configured(self) -> bool:
        return self.enabled

    async def get_upcoming_fixtures(
        self, start: date, end: date
    ) -> list[dict[str, Any]]:
        if not self.configured:
            return []
        if end < start or (end - start).days > 31:
            raise ValueError("Invalid OpenLigaDB date range")
        season = start.year if start.month >= 7 else start.year - 1
        results = await asyncio.gather(
            *(self._get_league(league_id, season) for league_id in LEAGUE_SHORTCUTS),
            return_exceptions=True,
        )
        fixtures: list[dict[str, Any]] = []
        failures = 0
        for result in results:
            if isinstance(result, BaseException):
                failures += 1
                continue
            fixtures.extend(
                row
                for raw in result
                if (row := self._normalize(raw))
                and start <= row["kickoff"].date() <= end
                and row["status"] == "NS"
            )
        if failures == len(results):
            raise OpenLigaDBError("All OpenLigaDB league requests failed")
        return fixtures

    async def get_fixture_by_id(self, fixture_id: int) -> dict[str, Any] | None:
        if not self.configured or not is_openligadb_fixture_id(fixture_id):
            return None
        payload = await self._get_json(f"getmatchdata/{fixture_id - ID_OFFSET}")
        row = self._normalize(payload)
        return row or None

    async def _get_league(self, league_id: int, season: int) -> list[object]:
        shortcut = LEAGUE_SHORTCUTS.get(league_id)
        if shortcut is None:
            raise ValueError(f"Unsupported OpenLigaDB league_id: {league_id}")
        payload = await self._get_json(f"getmatchdata/{shortcut}/{season}")
        if not isinstance(payload, list):
            raise OpenLigaDBError("OpenLigaDB payload must be a list")
        return payload

    async def _get_json(self, path: str) -> object:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "BetAIPlatform/1.0 fixture-aggregator",
                },
            ) as client:
                response = await client.get(f"{self.base_url}/{path.lstrip('/')}")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OpenLigaDBError(f"OpenLigaDB request failed: {path}") from exc
        if len(response.content) > MAX_PAYLOAD_BYTES:
            raise OpenLigaDBError("OpenLigaDB payload exceeds safety limit")
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenLigaDBError("OpenLigaDB returned invalid JSON") from exc
        return payload

    @staticmethod
    def _normalize(raw: object) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        team1 = raw.get("team1")
        team2 = raw.get("team2")
        if not isinstance(team1, dict) or not isinstance(team2, dict):
            return {}
        fixture_id = _namespace_id(raw.get("matchID"))
        home_team_id = _namespace_id(team1.get("teamId"))
        away_team_id = _namespace_id(team2.get("teamId"))
        kickoff = _parse_datetime(raw.get("matchDateTimeUTC"))
        league_id = next(
            (
                candidate
                for candidate, shortcut in LEAGUE_SHORTCUTS.items()
                if shortcut == str(raw.get("leagueShortcut") or "").lower()
            ),
            None,
        )
        home = str(team1.get("teamName") or "").strip()
        away = str(team2.get("teamName") or "").strip()
        if (
            fixture_id is None
            or home_team_id is None
            or away_team_id is None
            or kickoff is None
            or league_id is None
            or not home
            or not away
        ):
            return {}
        final_score = OpenLigaDBClient._final_score(raw.get("matchResults"))
        return {
            "fixture_id": fixture_id,
            "league_id": league_id,
            "league": LEAGUE_NAMES[league_id],
            "season": int(raw.get("leagueSeason") or kickoff.year),
            "kickoff": kickoff,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_team": home[:100],
            "away_team": away[:100],
            "status": "FT" if raw.get("matchIsFinished") is True else "NS",
            "score": (f"{final_score[0]} - {final_score[1]}" if final_score else None),
            "source": "openligadb",
        }

    @staticmethod
    def _final_score(value: object) -> tuple[int, int] | None:
        if not isinstance(value, list):
            return None
        candidates = [
            row
            for row in value
            if isinstance(row, dict)
            and isinstance(row.get("pointsTeam1"), int)
            and not isinstance(row.get("pointsTeam1"), bool)
            and isinstance(row.get("pointsTeam2"), int)
            and not isinstance(row.get("pointsTeam2"), bool)
        ]
        if not candidates:
            return None
        final = max(candidates, key=lambda row: int(row.get("resultOrderID") or 0))
        return int(final["pointsTeam1"]), int(final["pointsTeam2"])
