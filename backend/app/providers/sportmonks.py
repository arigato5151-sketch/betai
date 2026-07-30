from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx

from app.core.config import settings
from app.core.team_identity import stable_team_name_key
from app.providers.base import ProviderTeamCandidate

MAX_SPORTMONKS_PAYLOAD_BYTES = 8 * 1024 * 1024
SPORTMONKS_PLAYER_ID_OFFSET = 10_000_000_000
RATING_TYPE_ID = 118
MINUTES_TYPE_ID = 119
GOALS_TYPE_ID = 52
ASSISTS_TYPE_ID = 79


class SportmonksError(RuntimeError):
    """Raised when the optional Sportmonks fallback cannot be used safely."""


class SportmonksDownloadError(SportmonksError):
    """Raised when the provider request fails."""


class SportmonksFormatError(SportmonksError):
    """Raised when the provider response violates the expected schema."""


class SportmonksClient:
    """Resolve exact team identities and aggregate recent player performance."""

    def __init__(
        self,
        *,
        api_token: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        lookback_days: int | None = None,
        lookback_matches: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_token = (
            settings.SPORTMONKS_API_TOKEN if api_token is None else api_token
        ).strip()
        self.base_url = (base_url or settings.SPORTMONKS_BASE_URL).strip().rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.SPORTMONKS_TIMEOUT_SECONDS
        self.lookback_days = lookback_days or settings.SPORTMONKS_PLAYER_LOOKBACK_DAYS
        self.lookback_matches = (
            lookback_matches or settings.SPORTMONKS_PLAYER_LOOKBACK_MATCHES
        )
        self.transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.api_token)

    async def get_recent_player_ratings(
        self,
        *,
        team_name: str,
        as_of: datetime,
        provider_team_key: str | None = None,
    ) -> tuple[ProviderTeamCandidate, dict[int, dict[str, float]]] | None:
        if not self.configured or not team_name.strip():
            return None
        effective_at = self._aware_utc(as_of)
        candidate = await self.resolve_team(
            team_name=team_name,
            provider_team_key=provider_team_key,
        )
        if candidate is None:
            return None

        end_date = min(effective_at.date(), datetime.now(UTC).date())
        start_date = end_date - timedelta(days=self.lookback_days)
        payload = await self._request(
            (
                f"fixtures/between/{start_date.isoformat()}/"
                f"{end_date.isoformat()}/{candidate.provider_team_key}"
            ),
            params={
                "include": "lineups.details",
                "filters": "lineupDetailTypes:52,79,118,119",
                "order": "desc",
                "per_page": "50",
            },
        )
        ratings = self._aggregate_player_ratings(
            payload.get("data"),
            team_id=int(candidate.provider_team_key),
            before=effective_at,
            match_limit=self.lookback_matches,
        )
        return (candidate, ratings) if ratings else None

    async def resolve_team(
        self,
        *,
        team_name: str,
        provider_team_key: str | None = None,
    ) -> ProviderTeamCandidate | None:
        if provider_team_key is not None:
            try:
                team_id = int(provider_team_key)
            except (TypeError, ValueError):
                return None
            if team_id <= 0:
                return None
            payload = await self._request(f"teams/{team_id}")
            rows: object = [payload.get("data")]
        else:
            normalized = stable_team_name_key(team_name)
            if not normalized:
                return None
            payload = await self._request(f"teams/search/{quote(team_name.strip())}")
            rows = payload.get("data")

        matches: list[dict[str, object]] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                raw_team_id = row.get("id")
                name = row.get("name")
                if (
                    isinstance(raw_team_id, int)
                    and not isinstance(raw_team_id, bool)
                    and raw_team_id > 0
                    and isinstance(name, str)
                    and stable_team_name_key(name) == stable_team_name_key(team_name)
                ):
                    matches.append(row)
        if len(matches) != 1:
            return None

        match = matches[0]
        return ProviderTeamCandidate(
            provider_team_key=str(match["id"]),
            provider_team_name=str(match["name"]),
            confidence=1.0,
        )

    async def _request(
        self,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
                headers={
                    "Authorization": self.api_token,
                    "Accept": "application/json",
                    "User-Agent": "BetAIPlatform/1.0 sportmonks-fallback",
                },
            ) as client:
                response = await client.get(
                    f"{self.base_url}/{path.lstrip('/')}",
                    params=params,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SportmonksDownloadError(
                "Sportmonks player fallback request failed"
            ) from exc
        if len(response.content) > MAX_SPORTMONKS_PAYLOAD_BYTES:
            raise SportmonksFormatError("Sportmonks payload exceeds the safety limit")
        try:
            payload = response.json()
        except ValueError as exc:
            raise SportmonksFormatError("Sportmonks payload is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise SportmonksFormatError("Sportmonks payload must be an object")
        return payload

    @classmethod
    def _aggregate_player_ratings(
        cls,
        fixtures: object,
        *,
        team_id: int,
        before: datetime,
        match_limit: int,
    ) -> dict[int, dict[str, float]]:
        if not isinstance(fixtures, list):
            return {}
        eligible: list[tuple[datetime, dict[str, object]]] = []
        for raw_fixture in fixtures:
            if not isinstance(raw_fixture, dict):
                continue
            kickoff = cls._parse_kickoff(raw_fixture.get("starting_at"))
            lineups = raw_fixture.get("lineups")
            if kickoff is None or kickoff >= before or not isinstance(lineups, list):
                continue
            eligible.append((kickoff, raw_fixture))
        eligible.sort(key=lambda item: item[0], reverse=True)

        totals: dict[int, dict[str, float]] = defaultdict(
            lambda: {
                "weighted_rating": 0.0,
                "rating_weight": 0.0,
                "minutes": 0.0,
                "appearances": 0.0,
                "goals": 0.0,
                "assists": 0.0,
            }
        )
        for _kickoff, fixture in eligible[:match_limit]:
            lineups = fixture["lineups"]
            if not isinstance(lineups, list):
                continue
            for lineup in lineups:
                if not isinstance(lineup, dict) or lineup.get("team_id") != team_id:
                    continue
                player_id = lineup.get("player_id")
                if (
                    isinstance(player_id, bool)
                    or not isinstance(player_id, int)
                    or player_id <= 0
                ):
                    continue
                details = cls._detail_values(lineup.get("details"))
                rating = details.get(RATING_TYPE_ID)
                if rating is None or not 1.0 <= rating <= 10.0:
                    continue
                minutes = max(0.0, details.get(MINUTES_TYPE_ID, 0.0))
                weight = minutes if minutes > 0 else 1.0
                namespaced_id = SPORTMONKS_PLAYER_ID_OFFSET + player_id
                total = totals[namespaced_id]
                total["weighted_rating"] += rating * weight
                total["rating_weight"] += weight
                total["minutes"] += minutes
                total["appearances"] += 1.0
                total["goals"] += max(0.0, details.get(GOALS_TYPE_ID, 0.0))
                total["assists"] += max(0.0, details.get(ASSISTS_TYPE_ID, 0.0))

        result: dict[int, dict[str, float]] = {}
        for player_id, total in totals.items():
            weight = total["rating_weight"]
            if weight <= 0:
                continue
            result[player_id] = {
                "rating": round(total["weighted_rating"] / weight, 4),
                "minutes": total["minutes"],
                "appearances": total["appearances"],
                "goals": total["goals"],
                "assists": total["assists"],
            }
        return result

    @staticmethod
    def _detail_values(details: object) -> dict[int, float]:
        values: dict[int, float] = {}
        if not isinstance(details, list):
            return values
        for detail in details:
            if not isinstance(detail, dict):
                continue
            type_id = detail.get("type_id")
            data = detail.get("data")
            raw_value = data.get("value") if isinstance(data, dict) else None
            if isinstance(type_id, bool) or not isinstance(type_id, int):
                continue
            if isinstance(raw_value, bool) or not isinstance(
                raw_value, (int, float, str)
            ):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values[type_id] = value
        return values

    @staticmethod
    def _parse_kickoff(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @staticmethod
    def _aware_utc(value: datetime | date) -> datetime:
        if isinstance(value, date) and not isinstance(value, datetime):
            return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
